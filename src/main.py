"""
Script Principal de Imputación de Precipitación

Compara cuatro métodos: SLR, MLR, LWR, RF
Usa validación cruzada para métricas reales de rendimiento
"""

import os
import sys
import numpy as np
import xarray as xr
import pandas as pd

sys.path.append('./src')

import data_processing
import dem_processing
import near_stn_search
import weight_calculation
import regression
import evaluate


def run_imputation(ds_trans, near_indices, weights, config, method):
    """Ejecuta imputación con un método específico."""
    ds_imp_trans = regression.impute_data(
        ds_trans, near_indices, weights,
        method=method,
        min_valid_neighbors=config['min_valid_neighbors']
    )
    
    # Clamping
    max_t = data_processing.boxcox_transform(np.array([config['maxRange_vars']]), config['exp'])[0]
    min_t = data_processing.boxcox_transform(np.array([config['minRange_vars']]), config['exp'])[0]
    ds_imp_trans['prcp'].values = np.clip(ds_imp_trans['prcp'].values, min_t, max_t)
    
    # Transformación inversa
    prcp_final = data_processing.boxcox_retransform(ds_imp_trans['prcp'].values, config['exp'])
    prcp_final = np.clip(prcp_final, 0, config['maxRange_vars'])
    
    ds_imp = ds_trans.copy()
    ds_imp['prcp'].values = prcp_final
    
    return ds_imp, ds_imp_trans


def calculate_cv_metrics(ds_trans, near_indices, weights, method, config):
    """
    Calcula métricas usando validación cruzada.
    Esto evalúa el verdadero rendimiento de la imputación.
    """
    obs_true, pred_values = regression.cross_validate_imputation(
        ds_trans, near_indices, weights,
        method=method,
        sample_fraction=0.05,  # 5% de muestras para velocidad
        min_valid_neighbors=config['min_valid_neighbors']
    )
    
    if len(obs_true) < 10:
        return {'CC': np.nan, 'MAE': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan, 'NSE': np.nan, 'KGE_G2009': np.nan}
    
    # Retransformar para métricas en escala original
    obs_original = data_processing.boxcox_retransform(obs_true, config['exp'])
    pred_original = data_processing.boxcox_retransform(pred_values, config['exp'])
    
    metrics, names = evaluate.evaluate(obs_original, pred_original)
    return dict(zip(names, metrics))


def main():
    config = {
        'csv_path': '01_prcp.csv',
        'dem_path': 'DEM/DEM.tif',
        'out_dir': 'output',
        'nearstn_min': 1,
        'nearstn_max': 25,
        'try_radius': 150,
        'initial_distance': 100,
        'exp': 0.25,
        'min_valid_neighbors': 1,
        'minRange_vars': 0,
        'maxRange_vars': 300,
    }
    
    os.makedirs(config['out_dir'], exist_ok=True)
    
    # 1. Cargar datos
    print("\n" + "="*60)
    print("CARGANDO DATOS")
    print("="*60)
    ds = data_processing.load_precipitation_data(config['csv_path'])
    ds = data_processing.check_and_clamp_data(ds, config)
    ds = dem_processing.add_dem_predictors_to_dataset(ds, config['dem_path'])
    
    # 2. Transformación Box-Cox
    print("\n" + "="*60)
    print("TRANSFORMACIÓN BOX-COX")
    print("="*60)
    prcp_trans = data_processing.boxcox_transform(ds['prcp'].values, config['exp'])
    ds_trans = ds.copy()
    ds_trans['prcp'].values = prcp_trans
    
    # 3. Vecinos y pesos
    print("\n" + "="*60)
    print("VECINOS Y PESOS")
    print("="*60)
    near_indices, near_distances = near_stn_search.find_nearstn_for_InStn(
        ds['lat'].values, ds['lon'].values,
        config['try_radius'], config['nearstn_min'], 
        config['nearstn_max'], config['initial_distance']
    )
    weights = weight_calculation.calculate_weights_from_distance(
        near_distances, config['initial_distance']
    )
    
    # 4. Imputación y validación cruzada con los 4 métodos
    print("\n" + "="*60)
    print("IMPUTACIÓN CON CUATRO MÉTODOS")
    print("="*60)
    
    methods = ['SLR', 'MLR', 'LWR', 'RF']
    results = {}
    
    for method in methods:
        print(f"\n--- {method} ---")
        
        # Imputación completa
        ds_imp, ds_imp_trans = run_imputation(ds_trans, near_indices, weights, config, method)
        
        # Métricas con validación cruzada (rendimiento REAL)
        cv_metrics = calculate_cv_metrics(ds_trans, near_indices, weights, method, config)
        
        results[method] = {
            'ds': ds_imp,
            'ds_trans': ds_imp_trans,
            'metrics': cv_metrics
        }
        
        # Guardar CSV
        csv_path = os.path.join(config['out_dir'], f'DATOS_IMPUTADOS_{method}.csv')
        ds_imp['prcp'].to_pandas().to_csv(csv_path)
        print(f"  Guardado: {csv_path}")
    
    # 5. Tabla comparativa
    print("\n" + "="*60)
    print("COMPARACIÓN DE MÉTODOS (Validación Cruzada)")
    print("="*60)
    
    metrics_keys = ['CC', 'MAE', 'RMSE', 'PBIAS', 'NSE', 'KGE_G2009']
    
    print(f"\n{'Métrica':<12}", end='')
    for m in methods:
        print(f"{m:>12}", end='')
    print()
    print("-" * 60)
    
    for key in metrics_keys:
        print(f"{key:<12}", end='')
        for m in methods:
            val = results[m]['metrics'].get(key, np.nan)
            if not np.isnan(val):
                print(f"{val:>12.4f}", end='')
            else:
                print(f"{'N/A':>12}", end='')
        print()
    
    # 6. Guardar tabla como CSV
    df_compare = pd.DataFrame({
        'Metric': metrics_keys,
        'SLR': [results['SLR']['metrics'].get(k, np.nan) for k in metrics_keys],
        'MLR': [results['MLR']['metrics'].get(k, np.nan) for k in metrics_keys],
        'LWR': [results['LWR']['metrics'].get(k, np.nan) for k in metrics_keys],
        'RF': [results['RF']['metrics'].get(k, np.nan) for k in metrics_keys],
    })
    df_compare.to_csv(os.path.join(config['out_dir'], 'COMPARACION_METODOS.csv'), index=False)
    
    print("\n✓ PROCESO COMPLETADO")
    print(f"Archivos en: {config['out_dir']}/")


if __name__ == "__main__":
    main()
