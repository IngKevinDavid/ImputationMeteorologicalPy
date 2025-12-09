"""
Módulo de orquestación para imputación multi-variable.

Este módulo contiene funciones de alto nivel para procesar
múltiples variables meteorológicas (prcp, tmax, tmin, flow)
con diferentes métodos de imputación.
"""

import os
import numpy as np
import xarray as xr
from sklearn.neighbors import BallTree

import data_processing
import dem_processing
import near_stn_search
import weight_calculation
import regression


def get_var_config(config, var_index):
    """
    Extrae la configuración de una variable específica desde listas paralelas.
    
    Args:
        config: Diccionario de configuración global con listas paralelas
        var_index: Índice de la variable en las listas
        
    Returns:
        dict: Configuración específica de la variable
    """
    return {
        'var_name': config['variables'][var_index],
        'csv_path': config['csv_path'][var_index],
        'use_boxcox': config['use_boxcox'][var_index],
        'exp': config['exp'][var_index],
        'minRange': config['minRange'][var_index],
        'maxRange': config['maxRange'][var_index],
        'nearstn_max': config['nearstn_max'][var_index],
        'predictors': config['predictors'][var_index],
        'results_nc': config['results_nc'][var_index],
    }


def load_and_prepare_data(vcfg, dem_path):
    """
    Carga datos, aplica clamping y añade predictores DEM.
    
    Args:
        vcfg: Configuración de la variable
        dem_path: Ruta al archivo DEM
        
    Returns:
        xr.Dataset: Dataset preparado con predictores DEM
    """
    var_name = vcfg['var_name']
    
    # Cargar datos
    ds = data_processing.load_meteorological_data(vcfg['csv_path'], var_name)
    
    # Clamping
    clamp_config = {'minRange_vars': vcfg['minRange'], 'maxRange_vars': vcfg['maxRange']}
    ds = data_processing.check_and_clamp_data(ds, clamp_config, var_name)
    
    # Añadir predictores DEM
    ds = dem_processing.add_dem_predictors_to_dataset(ds, dem_path)
    
    return ds


def apply_boxcox_transform(ds, var_name, exp, use_boxcox):
    """
    Aplica transformación Box-Cox si corresponde.
    
    Args:
        ds: Dataset con la variable
        var_name: Nombre de la variable
        exp: Exponente de Box-Cox
        use_boxcox: Si aplicar transformación
        
    Returns:
        xr.Dataset: Dataset transformado (o copia sin transformar)
    """
    ds_trans = ds.copy()
    if use_boxcox:
        ds_trans[var_name].values = data_processing.boxcox_transform(
            ds[var_name].values, exp
        )
    return ds_trans


def find_neighbors_self(ds, config, vcfg):
    """
    Encuentra vecinos de la misma variable.
    
    Returns:
        tuple: (near_indices, weights)
    """
    near_indices, near_distances = near_stn_search.find_nearstn_for_InStn(
        ds['lat'].values, ds['lon'].values,
        config['try_radius'], config['nearstn_min'],
        vcfg['nearstn_max'], config['initial_distance']
    )
    weights = weight_calculation.calculate_weights_from_distance(
        near_distances, config['initial_distance']
    )
    return near_indices, weights


def find_neighbors_prcp_flow(ds, ds_prcp, config, n_prcp=3, n_flow=2):
    """
    Encuentra vecinos de precipitación Y caudal para imputación de flow.
    
    Args:
        ds: Dataset de caudal
        ds_prcp: Dataset de precipitación imputada
        config: Configuración global
        n_prcp: Número de estaciones de prcp cercanas
        n_flow: Número de estaciones de flow cercanas
        
    Returns:
        tuple: (near_indices_dict, weights_dict)
    """
    lat_flow = ds['lat'].values
    lon_flow = ds['lon'].values
    lat_prcp = ds_prcp['lat'].values
    lon_prcp = ds_prcp['lon'].values
    
    # BallTree para vecinos de prcp
    coords_flow = np.deg2rad(np.column_stack([lat_flow, lon_flow]))
    coords_prcp = np.deg2rad(np.column_stack([lat_prcp, lon_prcp]))
    
    tree_prcp = BallTree(coords_prcp, metric='haversine')
    dist_prcp, near_prcp_indices = tree_prcp.query(coords_flow, k=n_prcp)
    near_prcp_distances = dist_prcp * 6371  # km
    
    # BallTree para vecinos de flow (excluyendo self)
    tree_flow = BallTree(coords_flow, metric='haversine')
    dist_flow, near_flow_indices = tree_flow.query(coords_flow, k=n_flow + 1)
    near_flow_indices = near_flow_indices[:, 1:]  # Excluir self
    near_flow_distances = dist_flow[:, 1:] * 6371
    
    near_indices = {
        'prcp': near_prcp_indices,
        'flow': near_flow_indices
    }
    
    weights_prcp = weight_calculation.calculate_weights_from_distance(
        near_prcp_distances, config['initial_distance']
    )
    weights_flow = weight_calculation.calculate_weights_from_distance(
        near_flow_distances, config['initial_distance']
    )
    weights = {
        'prcp': weights_prcp,
        'flow': weights_flow
    }
    
    return near_indices, weights


def find_neighbors(ds, config, vcfg, all_results=None):
    """
    Encuentra vecinos según tipo de predictor.
    
    Args:
        ds: Dataset de la variable
        config: Configuración global
        vcfg: Configuración de la variable
        all_results: Resultados previos (para flow)
        
    Returns:
        tuple: (near_indices, weights)
    """
    if vcfg['predictors'] == 'self':
        return find_neighbors_self(ds, config, vcfg)
    
    elif vcfg['predictors'] == 'prcp+flow':
        if all_results is None or 'prcp' not in all_results:
            raise ValueError("Precipitación debe procesarse antes que caudales")
        
        # Usar primer método disponible de prcp
        prcp_method = list(all_results['prcp']['results'].keys())[0]
        ds_prcp = all_results['prcp']['results'][prcp_method]['ds']
        
        n_prcp = vcfg.get('n_prcp_neighbors', 3)
        n_flow = vcfg.get('n_flow_neighbors', 2)
        
        return find_neighbors_prcp_flow(ds, ds_prcp, config, n_prcp, n_flow)
    
    else:
        raise ValueError(f"Tipo de predictor desconocido: {vcfg['predictors']}")


def impute_single_method(ds_trans, near_indices, weights, method, vcfg, config, all_results=None):
    """
    Ejecuta imputación con un método específico.
    
    Returns:
        xr.Dataset: Dataset imputado (en espacio transformado si Box-Cox)
    """
    var_name = vcfg['var_name']
    
    if vcfg['predictors'] == 'prcp+flow':
        # Imputación especial para flow
        prcp_method = list(all_results['prcp']['results'].keys())[0]
        ds_prcp = all_results['prcp']['results'][prcp_method]['ds']
        
        ds_imp_trans = regression.impute_with_combined_predictors(
            ds_trans, ds_prcp, near_indices, weights, method, var_name, vcfg
        )
    else:
        # Imputación estándar
        ds_imp_trans = regression.impute_data(
            ds_trans, near_indices, weights,
            method=method,
            var_name=var_name,
            min_valid_neighbors=config['min_valid_neighbors']
        )
    
    return ds_imp_trans


def apply_boxcox_retransform(ds_imp_trans, ds_original, vcfg):
    """
    Aplica retransformación Box-Cox y clamping final.
    
    Returns:
        xr.Dataset: Dataset en escala original con clamping
    """
    var_name = vcfg['var_name']
    
    if vcfg['use_boxcox']:
        # Clamping en espacio transformado
        max_t = data_processing.boxcox_transform(np.array([vcfg['maxRange']]), vcfg['exp'])[0]
        min_t = data_processing.boxcox_transform(np.array([vcfg['minRange']]), vcfg['exp'])[0]
        ds_imp_trans[var_name].values = np.clip(ds_imp_trans[var_name].values, min_t, max_t)
        
        # Retransformar
        values_final = data_processing.boxcox_retransform(
            ds_imp_trans[var_name].values, vcfg['exp']
        )
        values_final = np.clip(values_final, vcfg['minRange'], vcfg['maxRange'])
        
        ds_imp = ds_original.copy()
        ds_imp[var_name].values = values_final
    else:
        ds_imp = ds_imp_trans.copy()
        ds_imp[var_name].values = np.clip(
            ds_imp[var_name].values, vcfg['minRange'], vcfg['maxRange']
        )
    
    return ds_imp


def impute_variable(ds, ds_trans, near_indices, weights, config, vcfg, all_results=None):
    """
    Ejecuta imputación para todos los métodos de una variable.
    
    Args:
        ds: Dataset original
        ds_trans: Dataset transformado
        near_indices: Índices de vecinos
        weights: Pesos
        config: Configuración global
        vcfg: Configuración de la variable
        all_results: Resultados previos (para flow)
        
    Returns:
        dict: Resultados por método {'method': {'ds': ..., 'ds_trans': ...}}
    """
    var_name = vcfg['var_name']
    results = {}
    
    for method in config['methods']:
        print(f"\n  --- {var_name}: {method} ---")
        
        # Imputar
        ds_imp_trans = impute_single_method(
            ds_trans, near_indices, weights, method, vcfg, config, all_results
        )
        
        # Retransformar
        ds_imp = apply_boxcox_retransform(ds_imp_trans, ds, vcfg)
        
        results[method] = {
            'ds': ds_imp,
            'ds_trans': ds_imp_trans if vcfg['use_boxcox'] else None
        }
    
    return results


def save_results_to_netcdf(results, ds, var_name, nc_path):
    """
    Guarda resultados imputados en NetCDF.
    
    Args:
        results: Diccionario de resultados por método
        ds: Dataset original (para coordenadas)
        var_name: Nombre de la variable
        nc_path: Ruta del archivo NetCDF
    """
    data_vars = {}
    for method in results:
        data_vars[f'{var_name}_{method}'] = (
            ['time', 'stn'], 
            results[method]['ds'][var_name].values
        )
    
    ds_save = xr.Dataset(
        data_vars=data_vars,
        coords={
            'time': ds['time'].values,
            'stn': ds['stn'].values,
            'lat': ('stn', ds['lat'].values),
            'lon': ('stn', ds['lon'].values),
        }
    )
    ds_save.to_netcdf(nc_path)
    print(f"  Guardado: {nc_path}")


def load_results_from_netcdf(nc_path, ds, config, vcfg):
    """
    Carga resultados previamente guardados desde NetCDF.
    
    Returns:
        tuple: (results_dict, ds_trans)
    """
    var_name = vcfg['var_name']
    
    ds_saved = xr.open_dataset(nc_path)
    
    # Crear ds_trans si aplica
    ds_trans = None
    if vcfg['use_boxcox']:
        ds_trans = ds.copy()
        ds_trans[var_name].values = data_processing.boxcox_transform(
            ds[var_name].values, vcfg['exp']
        )
    
    # Reconstruir resultados
    results = {}
    for method in config['methods']:
        key = f'{var_name}_{method}'
        if key in ds_saved:
            ds_imp = ds.copy()
            ds_imp[var_name].values = ds_saved[key].values
            results[method] = {'ds': ds_imp, 'ds_trans': None}
            print(f"    {method}: Cargado")
    
    ds_saved.close()
    return results, ds_trans


def process_single_variable(config, var_index, all_results):
    """
    Procesa una sola variable (carga, imputación, guardado).
    
    Args:
        config: Configuración global
        var_index: Índice de la variable
        all_results: Diccionario con resultados de variables anteriores
        
    Returns:
        dict: Resultados de esta variable
    """
    vcfg = get_var_config(config, var_index)
    var_name = vcfg['var_name']
    
    print(f"\n{'='*70}")
    print(f"  PROCESANDO: {var_name.upper()}")
    print(f"{'='*70}")
    
    # Verificar si hay resultados guardados
    if os.path.exists(vcfg['results_nc']):
        print(f"  Cargando desde: {vcfg['results_nc']}")
        
        ds = load_and_prepare_data(vcfg, config['dem_path'])
        results, ds_trans = load_results_from_netcdf(vcfg['results_nc'], ds, config, vcfg)
        
        # Calcular vecinos para evaluación
        near_indices, weights = None, None
        if vcfg['predictors'] == 'self':
            near_indices, weights = find_neighbors_self(ds, config, vcfg)
        
        return {
            'ds_original': ds,
            'ds_trans': ds_trans,
            'results': results,
            'near_indices': near_indices,
            'weights': weights,
        }
    
    # Cargar y preparar datos
    print("  Cargando datos...")
    ds = load_and_prepare_data(vcfg, config['dem_path'])
    print(f"\n    Estaciones: {ds.sizes['stn']}, Tiempo: {ds.sizes['time']}")
    
    # Transformación Box-Cox
    if vcfg['use_boxcox']:
        print(f"\n  Aplicando Box-Cox (exp={vcfg['exp']})...")
    ds_trans = apply_boxcox_transform(ds, var_name, vcfg['exp'], vcfg['use_boxcox'])
    
    # Encontrar vecinos
    print(f"  Buscando vecinos (predictors={vcfg['predictors']})...")
    near_indices, weights = find_neighbors(ds, config, vcfg, all_results)
    
    if not isinstance(near_indices, dict):
        print(f"    Vecinos: {near_indices.shape}")
    
    # Imputar para todos los métodos
    results = impute_variable(ds, ds_trans, near_indices, weights, config, vcfg, all_results)
    
    # Guardar en NetCDF
    save_results_to_netcdf(results, ds, var_name, vcfg['results_nc'])
    
    return {
        'ds_original': ds,
        'ds_trans': ds_trans if vcfg['use_boxcox'] else None,
        'results': results,
        'near_indices': near_indices,
        'weights': weights,
    }


def process_all_variables(config):
    """
    Función principal que procesa todas las variables secuencialmente.
    
    Args:
        config: Diccionario de configuración global
        
    Returns:
        dict: Resultados de todas las variables
    """
    os.makedirs(config['out_dir'], exist_ok=True)
    
    print("\n=== INICIANDO PROCESAMIENTO MULTI-VARIABLE ===\n")
    print(f"Variables: {config['variables']}")
    print(f"Métodos: {config['methods']}")
    
    all_results = {}
    
    for i, var_name in enumerate(config['variables']):
        all_results[var_name] = process_single_variable(config, i, all_results)
    
    print(f"\n{'='*70}")
    print("  ¡PROCESAMIENTO COMPLETO!")
    print(f"  Variables: {list(all_results.keys())}")
    print(f"  Métodos: {config['methods']}")
    print(f"{'='*70}")
    
    return all_results


def export_results_to_csv(config, all_results):
    """
    Exporta todos los resultados imputados a archivos CSV.
    
    Args:
        config: Configuración global
        all_results: Diccionario con todos los resultados
    """
    print("\n=== EXPORTANDO RESULTADOS ===\n")
    
    for var_name in config['variables']:
        for method in config['methods']:
            if method not in all_results[var_name]['results']:
                continue
            
            ds_imp = all_results[var_name]['results'][method]['ds']
            csv_path = os.path.join(config['out_dir'], f'DATOS_IMPUTADOS_{var_name}_{method}.csv')
            ds_imp[var_name].to_pandas().to_csv(csv_path)
            print(f"Exportado: {csv_path}")
    
    print("\n¡Exportación completa!")


def evaluate_all_variables(config, all_results):
    """
    Evalúa todas las variables con validación cruzada.
    Incluye flow con evaluación adaptada (hold-out).
    
    Args:
        config: Configuración global
        all_results: Diccionario con todos los resultados
        
    Returns:
        pd.DataFrame: Tabla con métricas de evaluación
    """
    import pandas as pd
    import evaluate as eval_module
    
    print("\n=== EVALUACIÓN CON VALIDACIÓN CRUZADA ===\n")
    
    metrics_csv_path = os.path.join(config['out_dir'], 'COMPARACION_METODOS_MULTI.csv')
    
    if os.path.exists(metrics_csv_path):
        print(f"Cargando métricas desde: {metrics_csv_path}")
        return pd.read_csv(metrics_csv_path)
    
    all_metrics = []
    
    for i, var_name in enumerate(config['variables']):
        vcfg = get_var_config(config, i)
        
        print(f"\n  Evaluando {var_name}...")
        
        if vcfg['predictors'] == 'self':
            # CV estándar para prcp, tmax, tmin
            if all_results[var_name].get('near_indices') is None:
                print(f"    Saltando CV (sin datos de vecinos)")
                continue
            
            ds_eval = all_results[var_name]['ds_original']
            if vcfg['use_boxcox'] and all_results[var_name].get('ds_trans') is not None:
                ds_eval = all_results[var_name]['ds_trans']
            
            near_idx = all_results[var_name]['near_indices']
            wts = all_results[var_name]['weights']
            
            for method in config['methods']:
                print(f"    {method}...")
                obs_true, pred_values = regression.cross_validate_imputation(
                    ds_eval, near_idx, wts,
                    method=method,
                    var_name=var_name,
                    sample_fraction=config['cv_sample_fraction'],
                    min_valid_neighbors=config['min_valid_neighbors']
                )
                
                if len(obs_true) < 10:
                    continue
                
                # Retransformar si aplica
                if vcfg['use_boxcox']:
                    obs_true = data_processing.boxcox_retransform(obs_true, vcfg['exp'])
                    pred_values = data_processing.boxcox_retransform(pred_values, vcfg['exp'])
                
                metrics, names = eval_module.evaluate(obs_true, pred_values)
                
                for name, val in zip(names, metrics):
                    all_metrics.append({
                        'Variable': var_name,
                        'Method': method,
                        'Metric': name,
                        'Value': val
                    })
        
        else:
            # Evaluación para flow usando hold-out
            print(f"    Evaluación hold-out para {var_name}...")
            
            for method in config['methods']:
                if method not in all_results[var_name]['results']:
                    continue
                
                print(f"    {method}...")
                
                obs_true, pred_values = cross_validate_flow(
                    config, all_results, var_name, method, vcfg,
                    sample_fraction=config['cv_sample_fraction']
                )
                
                if len(obs_true) < 10:
                    print(f"      Datos insuficientes para CV")
                    continue
                
                # Retransformar si aplica
                if vcfg['use_boxcox']:
                    obs_true = data_processing.boxcox_retransform(obs_true, vcfg['exp'])
                    pred_values = data_processing.boxcox_retransform(pred_values, vcfg['exp'])
                
                metrics, names = eval_module.evaluate(obs_true, pred_values)
                
                for name, val in zip(names, metrics):
                    all_metrics.append({
                        'Variable': var_name,
                        'Method': method,
                        'Metric': name,
                        'Value': val
                    })
    
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv(metrics_csv_path, index=False)
    print(f"\nGuardado: {metrics_csv_path}")
    
    return df_metrics


def cross_validate_flow(config, all_results, var_name, method, vcfg, sample_fraction=0.1):
    """
    Validación cruzada para flow usando hold-out con RE-IMPUTACIÓN REAL.
    Oculta una fracción de datos conocidos, re-imputa con el modelo, y compara.
    
    Args:
        config: Configuración global
        all_results: Resultados previos
        var_name: Nombre de la variable (flow)
        method: Método de imputación
        vcfg: Configuración de la variable
        sample_fraction: Fracción de datos a ocultar
        
    Returns:
        tuple: (obs_true, pred_values)
    """
    from sklearn.linear_model import Ridge
    
    np.random.seed(42)
    
    # Obtener datos originales
    ds_obs = all_results[var_name]['ds_original']
    obs_data = ds_obs[var_name].values.copy()
    
    # Datos de prcp imputados
    prcp_method = list(all_results['prcp']['results'].keys())[0]
    ds_prcp = all_results['prcp']['results'][prcp_method]['ds']
    prcp_data = ds_prcp['prcp'].values
    
    # Obtener índices de vecinos
    near_indices = all_results[var_name].get('near_indices')
    if near_indices is None or not isinstance(near_indices, dict):
        return np.array([]), np.array([])
    
    # Encontrar posiciones con datos válidos
    valid_mask = ~np.isnan(obs_data)
    valid_positions = np.argwhere(valid_mask)
    
    # Seleccionar muestra para hold-out
    n_sample = int(len(valid_positions) * sample_fraction)
    n_sample = max(10, min(n_sample, len(valid_positions)))
    sample_idx = np.random.choice(len(valid_positions), n_sample, replace=False)
    holdout_positions = valid_positions[sample_idx]
    
    print(f"      CV: {n_sample} puntos hold-out")
    
    obs_true = []
    pred_values = []
    
    # Para cada estación, entrenar modelo excluyendo puntos hold-out y predecir
    nstn = obs_data.shape[1]
    
    for s in range(nstn):
        # Posiciones hold-out para esta estación
        stn_holdout = holdout_positions[holdout_positions[:, 1] == s]
        if len(stn_holdout) == 0:
            continue
        
        # Índices de vecinos para esta estación
        prcp_idx = near_indices['prcp'][s]
        flow_idx = near_indices['flow'][s]
        
        # Construir predictores
        X_prcp = prcp_data[:, prcp_idx]  # (ntime, n_prcp_neighbors)
        X_flow = obs_data[:, flow_idx]   # (ntime, n_flow_neighbors)
        X = np.hstack([X_prcp, X_flow])  # (ntime, n_prcp + n_flow)
        y = obs_data[:, s]
        
        # Para cada punto hold-out de esta estación
        for pos in stn_holdout:
            t = pos[0]
            
            # Valor observado real (guardamos antes de ocultar)
            obs_val = y[t]
            
            # Crear máscara de entrenamiento (excluyendo el punto actual)
            train_mask = ~np.isnan(y.copy())
            train_mask[t] = False  # Excluir el punto hold-out
            
            if train_mask.sum() < 10:
                continue
            
            X_train = X[train_mask]
            y_train = y[train_mask]
            
            # Manejar NaN en predictores
            X_train = np.nan_to_num(X_train, nan=0)
            X_pred = np.nan_to_num(X[t:t+1], nan=0)
            
            # Verificar que hay datos
            if np.std(X_train) < 1e-6:
                continue
            
            try:
                # Entrenar modelo y predecir
                model = Ridge(alpha=0.1)
                model.fit(X_train, y_train)
                pred_val = model.predict(X_pred)[0]
                
                if not np.isnan(obs_val) and not np.isnan(pred_val):
                    obs_true.append(obs_val)
                    pred_values.append(pred_val)
            except Exception:
                continue
    
    print(f"      CV: {len(obs_true)} predicciones exitosas")
    return np.array(obs_true), np.array(pred_values)


def compute_basic_statistics(config, all_results):
    """
    Calcula estadísticas básicas para todas las variables y métodos.
    Útil para variables como flow que no tienen CV tradicional.
    
    Args:
        config: Configuración global
        all_results: Diccionario con todos los resultados
        
    Returns:
        pd.DataFrame: Tabla con estadísticas
    """
    import pandas as pd
    
    print("\n=== ESTADÍSTICAS BÁSICAS ===\n")
    
    all_stats = []
    
    for i, var_name in enumerate(config['variables']):
        vcfg = get_var_config(config, i)
        
        print(f"  {var_name}...")
        
        ds_obs = all_results[var_name]['ds_original']
        obs_data = ds_obs[var_name].values
        
        # Estadísticas de datos originales
        n_obs = np.sum(~np.isnan(obs_data))
        n_missing = np.sum(np.isnan(obs_data))
        pct_missing = 100 * n_missing / obs_data.size
        
        all_stats.append({
            'Variable': var_name,
            'Tipo': 'Original',
            'Método': '-',
            'N_Obs': n_obs,
            'N_Missing': n_missing,
            'Pct_Missing': round(pct_missing, 2),
            'Mean': round(np.nanmean(obs_data), 4),
            'Std': round(np.nanstd(obs_data), 4),
            'Min': round(np.nanmin(obs_data), 4),
            'Max': round(np.nanmax(obs_data), 4),
        })
        
        # Estadísticas por método
        for method in config['methods']:
            if method not in all_results[var_name]['results']:
                continue
            
            ds_imp = all_results[var_name]['results'][method]['ds']
            imp_data = ds_imp[var_name].values
            
            n_imputed = np.sum(~np.isnan(imp_data))
            
            all_stats.append({
                'Variable': var_name,
                'Tipo': 'Imputado',
                'Método': method,
                'N_Obs': n_imputed,
                'N_Missing': np.sum(np.isnan(imp_data)),
                'Pct_Missing': round(100 * np.sum(np.isnan(imp_data)) / imp_data.size, 2),
                'Mean': round(np.nanmean(imp_data), 4),
                'Std': round(np.nanstd(imp_data), 4),
                'Min': round(np.nanmin(imp_data), 4),
                'Max': round(np.nanmax(imp_data), 4),
            })
    
    df_stats = pd.DataFrame(all_stats)
    
    # Guardar
    stats_path = os.path.join(config['out_dir'], 'ESTADISTICAS_BASICAS.csv')
    df_stats.to_csv(stats_path, index=False)
    print(f"\nGuardado: {stats_path}")
    
    return df_stats


def compute_obs_vs_imp_metrics(config, all_results):
    """
    Calcula métricas comparando datos originales vs imputados.
    SOLO donde hay observaciones (para evaluar calidad de imputación).
    Incluye TODAS las variables, incluyendo flow.
    
    Args:
        config: Configuración global
        all_results: Diccionario con todos los resultados
        
    Returns:
        pd.DataFrame: Tabla con métricas por variable/método
    """
    import pandas as pd
    import evaluate as eval_module
    
    print("\n=== MÉTRICAS OBS VS IMPUTADO ===\n")
    
    all_metrics = []
    
    for i, var_name in enumerate(config['variables']):
        vcfg = get_var_config(config, i)
        
        print(f"  {var_name}...")
        
        ds_obs = all_results[var_name]['ds_original']
        obs_data = ds_obs[var_name].values
        
        for method in config['methods']:
            if method not in all_results[var_name]['results']:
                continue
            
            ds_imp = all_results[var_name]['results'][method]['ds']
            imp_data = ds_imp[var_name].values
            
            # Comparar SOLO donde hay observaciones originales
            valid_mask = ~np.isnan(obs_data)
            obs_valid = obs_data[valid_mask]
            imp_valid = imp_data[valid_mask]
            
            # Filtrar también NaN en imputados (si quedaron algunos)
            both_valid = ~np.isnan(imp_valid)
            obs_final = obs_valid[both_valid]
            imp_final = imp_valid[both_valid]
            
            if len(obs_final) < 10:
                print(f"    {method}: Datos insuficientes")
                continue
            
            # Calcular métricas
            metrics_vals, metric_names = eval_module.evaluate(obs_final, imp_final)
            
            for name, val in zip(metric_names, metrics_vals):
                all_metrics.append({
                    'Variable': var_name,
                    'Método': method,
                    'Métrica': name,
                    'Valor': round(val, 4) if not np.isnan(val) else np.nan,
                })
            
            # Añadir métricas adicionales
            bias = np.mean(imp_final - obs_final)
            mae = np.mean(np.abs(imp_final - obs_final))
            
            all_metrics.append({
                'Variable': var_name,
                'Método': method,
                'Métrica': 'BIAS',
                'Valor': round(bias, 4),
            })
            all_metrics.append({
                'Variable': var_name,
                'Método': method,
                'Métrica': 'MAE',
                'Valor': round(mae, 4),
            })
            
            print(f"    {method}: CC={metrics_vals[0]:.4f}, RMSE={metrics_vals[1]:.4f}, BIAS={bias:.4f}")
    
    df_metrics = pd.DataFrame(all_metrics)
    
    # Guardar
    metrics_path = os.path.join(config['out_dir'], 'METRICAS_OBS_VS_IMP.csv')
    df_metrics.to_csv(metrics_path, index=False)
    print(f"\nGuardado: {metrics_path}")
    
    return df_metrics
