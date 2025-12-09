"""
Módulo de Regresión para Imputación de Datos

Implementa tres métodos:
- SLR: Simple Linear Regression (mejor vecino)
- MLR: Multiple Linear Regression (múltiples vecinos como predictores)
- LWR: Locally Weighted Regression (predictores DEM)
"""

import numpy as np
import xarray as xr
from sklearn.linear_model import Ridge, LinearRegression


def normalize_predictors(X, X_target=None):
    """Normaliza predictores usando z-score."""
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std < 1e-10] = 1.0
    
    X_norm = (X - mean) / std
    
    if X_target is not None:
        X_target_norm = (X_target - mean) / std
        return X_norm, X_target_norm
    return X_norm


def idw_estimate(values, weights):
    """Promedio ponderado inverso (IDW)."""
    weights_norm = weights / np.sum(weights)
    return np.sum(values * weights_norm)


def slr_estimate(y_target_history, y_neighbor_history, y_neighbor_current):
    """
    Simple Linear Regression: Y = β₀ + β₁·X_best
    
    Usa el historial de la estación target y el mejor vecino para
    ajustar una regresión simple.
    """
    try:
        # Filtrar datos válidos (ambos no-NaN)
        valid = ~np.isnan(y_target_history) & ~np.isnan(y_neighbor_history)
        if np.sum(valid) < 10:
            return np.nan
        
        y = y_target_history[valid]
        x = y_neighbor_history[valid].reshape(-1, 1)
        
        model = LinearRegression()
        model.fit(x, y)
        
        return model.predict([[y_neighbor_current]])[0]
    except:
        return np.nan


def mlr_estimate(y_target_history, neighbors_history, neighbors_current, weights):
    """
    Multiple Linear Regression: Y = β₀ + Σβᵢ·Xᵢ
    
    Usa valores de múltiples vecinos como predictores.
    
    Args:
        y_target_history: Historial de la estación target (ntime,)
        neighbors_history: Historial de vecinos (ntime, n_neighbors)
        neighbors_current: Valores actuales de vecinos (n_neighbors,)
        weights: Pesos de vecinos (n_neighbors,)
    """
    try:
        # Filtrar tiempos donde target y al menos 3 vecinos tienen datos
        valid_times = ~np.isnan(y_target_history)
        valid_neighbors = ~np.isnan(neighbors_history)
        
        # Contar vecinos válidos por tiempo
        n_valid_per_time = np.sum(valid_neighbors, axis=1)
        valid_times = valid_times & (n_valid_per_time >= 3)
        
        if np.sum(valid_times) < 20:
            return np.nan
        
        y = y_target_history[valid_times]
        X = neighbors_history[valid_times, :]
        
        # Rellenar NaN en X con la media de cada vecino
        for j in range(X.shape[1]):
            col = X[:, j]
            if np.all(np.isnan(col)):
                # Columna completamente NaN - MLR fallará
                return np.nan
            col_mean = np.nanmean(col)
            X[np.isnan(X[:, j]), j] = col_mean
        
        # Ajustar modelo con regularización
        model = Ridge(alpha=1.0)
        model.fit(X, y, sample_weight=None)
        
        # Rellenar NaN en current con media
        neighbors_current_filled = neighbors_current.copy()
        neighbors_current_filled[np.isnan(neighbors_current_filled)] = np.nanmean(neighbors_current_filled)
        
        return model.predict(neighbors_current_filled.reshape(1, -1))[0]
    except:
        return np.nan


def rf_estimate(y_target_history, neighbors_history, neighbors_current, weights, n_estimators=100):
    """
    Random Forest Regression para imputación de precipitación.
    
    Configuración optimizada para datos de precipitación:
    - n_estimators=100: suficientes árboles para estabilidad
    - max_depth=10: captura patrones complejos
    - min_samples_leaf=3: evita sobreajuste
    
    Args:
        y_target_history: Historial de la estación target (ntime,)
        neighbors_history: Historial de vecinos (ntime, n_neighbors)
        neighbors_current: Valores actuales de vecinos (n_neighbors,)
        weights: Pesos de vecinos (n_neighbors,)
        n_estimators: Número de árboles en el bosque
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        
        # Filtrar tiempos donde target y al menos 3 vecinos tienen datos
        valid_times = ~np.isnan(y_target_history)
        valid_neighbors = ~np.isnan(neighbors_history)
        
        # Contar vecinos válidos por tiempo
        n_valid_per_time = np.sum(valid_neighbors, axis=1)
        valid_times = valid_times & (n_valid_per_time >= 3)
        
        n_valid = np.sum(valid_times)
        if n_valid < 30:
            return np.nan
        
        # Usar hasta 1000 muestras para mejor calidad
        valid_indices = np.where(valid_times)[0]
        if len(valid_indices) > 1000:
            np.random.seed(42)
            valid_indices = np.random.choice(valid_indices, 1000, replace=False)
        
        y = y_target_history[valid_indices]
        X = neighbors_history[valid_indices, :]
        
        # Rellenar NaN en X con la media de cada vecino
        for j in range(X.shape[1]):
            col = X[:, j]
            if np.all(np.isnan(col)):
                return np.nan
            col_mean = np.nanmean(col)
            X[np.isnan(X[:, j]), j] = col_mean
        
        # RF optimizado para precipitación
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_leaf=3,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,  # Usar todos los cores
            max_features= 0.9
        )
        model.fit(X, y)
        
        # Rellenar NaN en current con media
        neighbors_current_filled = neighbors_current.copy()
        neighbors_current_filled[np.isnan(neighbors_current_filled)] = np.nanmean(neighbors_current_filled)
        
        return model.predict(neighbors_current_filled.reshape(1, -1))[0]
    except Exception:
        return np.nan


def rf_estimate_with_spatial(y_target_history, neighbors_history, neighbors_current, 
                              spatial_predictors_neighbors, spatial_predictors_target,
                              weights, n_estimators=100):
    """
    Random Forest Regression MEJORADO con predictores espaciales.
    
    Incluye los mismos predictores que LWR:
    - Series temporales de vecinos
    - Predictores espaciales: elev, slp_n, slp_e, lat, lon
    
    Args:
        y_target_history: Historial de la estación target (ntime,)
        neighbors_history: Historial de vecinos (ntime, n_neighbors)
        neighbors_current: Valores actuales de vecinos (n_neighbors,)
        spatial_predictors_neighbors: Predictores espaciales de vecinos (n_neighbors, 5)
        spatial_predictors_target: Predictores espaciales del target (5,)
        weights: Pesos de vecinos (n_neighbors,)
        n_estimators: Número de árboles en el bosque
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        
        # Filtrar tiempos donde target y al menos 3 vecinos tienen datos
        valid_times = ~np.isnan(y_target_history)
        valid_neighbors = ~np.isnan(neighbors_history)
        
        n_valid_per_time = np.sum(valid_neighbors, axis=1)
        valid_times = valid_times & (n_valid_per_time >= 3)
        
        n_valid = np.sum(valid_times)
        if n_valid < 30:
            return np.nan
        
        # Usar hasta 1500 muestras (más porque tenemos más features)
        valid_indices = np.where(valid_times)[0]
        if len(valid_indices) > 1500:
            np.random.seed(42)
            valid_indices = np.random.choice(valid_indices, 1500, replace=False)
        
        y = y_target_history[valid_indices]
        X_temporal = neighbors_history[valid_indices, :]
        
        # Rellenar NaN en X_temporal con la media de cada vecino
        for j in range(X_temporal.shape[1]):
            col = X_temporal[:, j]
            if np.all(np.isnan(col)):
                col_mean = 0
            else:
                col_mean = np.nanmean(col)
            X_temporal[np.isnan(X_temporal[:, j]), j] = col_mean
        
        # Expandir predictores espaciales para cada timestep
        # Usamos la diferencia entre target y vecinos (más informativo)
        n_samples = len(valid_indices)
        n_spatial = spatial_predictors_target.shape[0]
        
        # Diferencia de predictores espaciales: target - media(vecinos)
        spatial_diff = spatial_predictors_target - np.mean(spatial_predictors_neighbors, axis=0)
        X_spatial = np.tile(spatial_diff, (n_samples, 1))  # Repetir para cada timestep
        
        # Combinar: [temporal] + [spatial]
        X = np.hstack([X_temporal, X_spatial])
        
        # RF optimizado con más features
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=20,  # Aumentado para manejar más features
            min_samples_leaf=3,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            max_features='sqrt'  # Mejor para muchos features
        )
        model.fit(X, y)
        
        # Preparar X para predicción
        neighbors_current_filled = neighbors_current.copy()
        neighbors_current_filled[np.isnan(neighbors_current_filled)] = np.nanmean(neighbors_current_filled)
        
        X_pred = np.hstack([neighbors_current_filled, spatial_diff])
        
        return model.predict(X_pred.reshape(1, -1))[0]
    except Exception:
        return np.nan


def impute_with_combined_predictors(ds_target, ds_prcp, near_indices, weights, 
                                     method, var_name, vcfg):
    """
    Imputa caudales usando AMBOS predictores: precipitación + caudales vecinos.
    
    Args:
        ds_target: Dataset con la variable a imputar (flow)
        ds_prcp: Dataset con precipitación imputada
        near_indices: dict con 'prcp' y 'flow' indices
        weights: dict con 'prcp' y 'flow' weights
        method: Método (no usado aún, para compatibilidad futura)
        var_name: Nombre de la variable target
        vcfg: Configuración de la variable
        
    Returns:
        xr.Dataset: Dataset con valores imputados
    """
    from sklearn.linear_model import Ridge
    
    target = ds_target[var_name].values.copy()
    prcp_data = ds_prcp['prcp'].values
    
    ntime, nstn = target.shape
    
    for s in range(nstn):
        # Índices de estaciones vecinas
        prcp_idx = near_indices['prcp'][s]
        flow_idx = near_indices['flow'][s]
        
        # Series de predictores
        X_prcp = prcp_data[:, prcp_idx]
        X_flow = target[:, flow_idx]
        
        # Combinar predictores
        X = np.hstack([X_prcp, X_flow])
        y = target[:, s]
        
        # Máscaras
        train_mask = ~np.isnan(y)
        pred_mask = np.isnan(y)
        
        if train_mask.sum() < 10 or pred_mask.sum() == 0:
            continue
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_pred = X[pred_mask]
        
        # Manejar NaN en predictores
        X_mean = np.nanmean(X_train, axis=0)
        X_mean = np.where(np.isnan(X_mean), 0, X_mean)
        X_train = np.nan_to_num(X_train, nan=0)
        X_pred = np.nan_to_num(X_pred, nan=0)
        
        # Verificar varianza
        if np.std(X_train) < 1e-6:
            continue
        
        try:
            model = Ridge(alpha=0.1)
            model.fit(X_train, y_train)
            predictions = model.predict(X_pred)
            target[pred_mask, s] = predictions
        except Exception:
            continue
    
    ds_out = ds_target.copy()
    ds_out[var_name].values = target
    return ds_out

def impute_data(ds, near_indices, weights, method='LWR', var_name='prcp', min_valid_neighbors=3):
    """
    Imputa datos faltantes usando el método especificado.
    Utiliza paralelización con joblib para todos los métodos.
    
    Args:
        ds: Dataset con la variable, lat, lon, elev_dem, slp_n, slp_e
        near_indices: Índices de vecinos (nstn, near)
        weights: Pesos de vecinos (nstn, near)
        method: 'SLR', 'MLR', 'LWR', 'RF' o 'IDW'
        var_name: Nombre de la variable a imputar ('prcp', 'tmax', 'tmin', 'flow')
        min_valid_neighbors: Mínimo de vecinos requeridos
        
    Returns:
        xr.Dataset: Dataset con datos imputados
    """
    from joblib import Parallel, delayed
    import multiprocessing
    
    print(f"Imputación de {var_name} usando método: {method}")
    
    prcp = ds[var_name].values  # (time, stn)
    lat = ds['lat'].values
    lon = ds['lon'].values
    
    # Predictores DEM (solo para LWR)
    elev = ds['elev_dem'].values if 'elev_dem' in ds else ds['elev'].values
    slp_n = ds['slp_n'].values if 'slp_n' in ds else np.zeros(len(lat))
    slp_e = ds['slp_e'].values if 'slp_e' in ds else np.zeros(len(lat))
    
    imputed_prcp = prcp.copy()
    ntime, nstn = prcp.shape
    
    # Recopilar todos los puntos faltantes
    missing_points = []
    for s in range(nstn):
        curr_idx = near_indices[s, :]
        curr_weights = weights[s, :]
        valid_mask = (curr_idx >= 0) & (~np.isnan(curr_weights)) & (curr_weights > 0)
        curr_idx_valid = curr_idx[valid_mask].astype(int)
        curr_weights_valid = curr_weights[valid_mask]
        
        if len(curr_idx_valid) < min_valid_neighbors:
            continue
        
        missing_times = np.where(np.isnan(prcp[:, s]))[0]
        for t in missing_times:
            datanear = prcp[t, curr_idx_valid]
            valid_data = ~np.isnan(datanear)
            if np.sum(valid_data) >= 1:
                missing_points.append({
                    't': t, 's': s,
                    'curr_idx': curr_idx_valid,
                    'curr_weights': curr_weights_valid,
                    'idx_valid': curr_idx_valid[valid_data],
                    'y_valid': datanear[valid_data],
                    'w_valid': curr_weights_valid[valid_data],
                    'n_valid': np.sum(valid_data)
                })
    
    n_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"  Procesando {len(missing_points):,} puntos en paralelo con {n_cores} cores...")
    
    def _impute_single(point):
        """Imputa un solo punto usando el método especificado."""
        t, s = point['t'], point['s']
        curr_idx = point['curr_idx']
        curr_weights = point['curr_weights']
        idx_valid = point['idx_valid']
        y_valid = point['y_valid']
        w_valid = point['w_valid']
        n_valid = point['n_valid']
        
        # IDW como baseline/fallback
        idw_val = idw_estimate(y_valid, w_valid)
        
        if method == 'IDW':
            return (t, s, idw_val, 'idw')
        
        elif method == 'SLR':
            best_idx = idx_valid[np.argmax(w_valid)]
            est = slr_estimate(prcp[:, s], prcp[:, best_idx], prcp[t, best_idx])
            if np.isnan(est) or np.isinf(est):
                return (t, s, idw_val, 'idw')
            return (t, s, est, 'method')
        
        elif method == 'MLR':
            max_neighbors = min(10, len(curr_idx))
            top_idx = curr_idx[:max_neighbors]
            est = mlr_estimate(prcp[:, s], prcp[:, top_idx], prcp[t, top_idx], curr_weights[:max_neighbors])
            if np.isnan(est) or np.isinf(est):
                return (t, s, idw_val, 'idw')
            return (t, s, est, 'method')
        
        elif method == 'LWR':
            if n_valid < 5:
                return (t, s, idw_val, 'idw')
            
            tar_pred = np.array([elev[s], slp_n[s], slp_e[s], lat[s], lon[s]])
            near_pred = np.column_stack([
                elev[idx_valid], slp_n[idx_valid], slp_e[idx_valid],
                lat[idx_valid], lon[idx_valid]
            ])
            
            X_norm, tar_norm = normalize_predictors(near_pred, tar_pred)
            
            try:
                model = Ridge(alpha=0.1)
                model.fit(X_norm, y_valid, sample_weight=w_valid)
                lwr_val = model.predict(tar_norm.reshape(1, -1))[0]
                
                if np.isnan(lwr_val) or np.isinf(lwr_val):
                    return (t, s, idw_val, 'idw')
                
                # Mezcla adaptativa
                diff = abs(lwr_val - idw_val) / (abs(idw_val) + 0.1)
                alpha = 0.4 if diff > 1.5 else (0.6 if diff > 0.8 else 0.85)
                return (t, s, alpha * lwr_val + (1 - alpha) * idw_val, 'method')
            except Exception:
                return (t, s, idw_val, 'idw')
        
        elif method == 'RF':
            max_neighbors = min(10, len(curr_idx))
            top_idx = curr_idx[:max_neighbors]
            
            # Preparar predictores espaciales para vecinos y target
            spatial_neighbors = np.column_stack([
                elev[top_idx], slp_n[top_idx], slp_e[top_idx],
                lat[top_idx], lon[top_idx]
            ])
            spatial_target = np.array([elev[s], slp_n[s], slp_e[s], lat[s], lon[s]])
            
            # Usar RF mejorado con predictores espaciales
            est = rf_estimate_with_spatial(
                prcp[:, s], prcp[:, top_idx], prcp[t, top_idx],
                spatial_neighbors, spatial_target,
                curr_weights[:max_neighbors]
            )
            if np.isnan(est) or np.isinf(est):
                return (t, s, idw_val, 'idw')
            return (t, s, est, 'method')
        
        return (t, s, idw_val, 'idw')
    
    # Ejecutar en paralelo
    results = Parallel(n_jobs=n_cores, verbose=1)(
        delayed(_impute_single)(point) for point in missing_points
    )
    
    # Aplicar resultados
    counts = {'method': 0, 'idw': 0, 'failed': 0}
    for t, s, val, status in results:
        if not np.isnan(val):
            imputed_prcp[t, s] = val
        counts[status] += 1
    
    print(f"  {method}: {counts['method']}, IDW fallback: {counts['idw']}, Fallidas: {counts['failed']}")
    
    ds_out = ds.copy()
    ds_out[var_name].values = imputed_prcp
    return ds_out


def cross_validate_imputation(ds, near_indices, weights, method='LWR', var_name='prcp', sample_fraction=0.20, min_valid_neighbors=3):
    """
    Validación cruzada leave-one-out para evaluar métodos de imputación.
    Utiliza paralelización con joblib para todos los métodos.
    
    En lugar de evaluar valores ya imputados contra valores ya imputados,
    esta función:
    1. Selecciona valores observados (no NaN)
    2. Los enmascara temporalmente como NaN
    3. Aplica la imputación para predecirlos
    4. Compara predicciones con valores reales
    
    Args:
        ds: Dataset con la variable, lat, lon, etc.
        near_indices: Índices de vecinos (nstn, near)
        weights: Pesos de vecinos (nstn, near)
        method: 'SLR', 'MLR', 'LWR', 'RF', o 'IDW'
        var_name: Nombre de la variable a validar
        sample_fraction: Fracción de observaciones a usar para validación
        min_valid_neighbors: Mínimo de vecinos requeridos
        
    Returns:
        obs_true: Array de valores originales "ocultos"
        pred_values: Array de valores predichos
    """
    from joblib import Parallel, delayed
    import multiprocessing
    
    print(f"Validación cruzada de {var_name} para método: {method}")
    
    prcp = ds[var_name].values.copy()  # (time, stn)
    lat = ds['lat'].values
    lon = ds['lon'].values
    
    # Predictores DEM (solo para LWR)
    elev = ds['elev_dem'].values if 'elev_dem' in ds else ds['elev'].values
    slp_n = ds['slp_n'].values if 'slp_n' in ds else np.zeros(len(lat))
    slp_e = ds['slp_e'].values if 'slp_e' in ds else np.zeros(len(lat))
    
    ntime, nstn = prcp.shape
    
    # Encontrar todas las posiciones con datos observados
    obs_mask = ~np.isnan(prcp)
    obs_positions = np.argwhere(obs_mask)
    
    # Muestrear una fracción para validación
    n_samples = max(100, int(len(obs_positions) * sample_fraction))
    np.random.seed(42)
    sample_idx = np.random.choice(len(obs_positions), size=min(n_samples, len(obs_positions)), replace=False)
    
    n_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"  Procesando {len(sample_idx):,} muestras en paralelo con {n_cores} cores...")
    
    def _cv_single(idx):
        """Procesa una sola muestra de validación cruzada."""
        t, s = obs_positions[idx]
        true_val = prcp[t, s]
        
        # Obtener vecinos
        curr_idx = near_indices[s, :]
        curr_weights = weights[s, :]
        
        valid_mask = (curr_idx >= 0) & (~np.isnan(curr_weights)) & (curr_weights > 0)
        curr_idx = curr_idx[valid_mask].astype(int)
        curr_weights = curr_weights[valid_mask]
        
        if len(curr_idx) < min_valid_neighbors:
            return None
        
        # Enmascarar temporalmente este valor
        prcp_masked = prcp.copy()
        prcp_masked[t, s] = np.nan
        
        datanear = prcp_masked[t, curr_idx]
        valid_data = ~np.isnan(datanear)
        n_valid = np.sum(valid_data)
        
        if n_valid < 1:
            return None
        
        y_valid = datanear[valid_data]
        w_valid = curr_weights[valid_data]
        idx_valid = curr_idx[valid_data]
        
        # IDW como baseline
        idw_val = idw_estimate(y_valid, w_valid)
        
        if method == 'IDW':
            pred = idw_val
            
        elif method == 'SLR':
            best_idx = idx_valid[np.argmax(w_valid)]
            pred = slr_estimate(prcp_masked[:, s], prcp_masked[:, best_idx], prcp_masked[t, best_idx])
            if np.isnan(pred) or np.isinf(pred):
                pred = idw_val
                
        elif method == 'MLR':
            max_neighbors = min(10, len(curr_idx))
            top_idx = curr_idx[:max_neighbors]
            pred = mlr_estimate(prcp_masked[:, s], prcp_masked[:, top_idx], prcp_masked[t, top_idx], curr_weights[:max_neighbors])
            if np.isnan(pred) or np.isinf(pred):
                pred = idw_val
                
        elif method == 'LWR':
            if n_valid < 5:
                pred = idw_val
            else:
                tar_pred = np.array([elev[s], slp_n[s], slp_e[s], lat[s], lon[s]])
                near_pred = np.column_stack([
                    elev[idx_valid], slp_n[idx_valid], slp_e[idx_valid],
                    lat[idx_valid], lon[idx_valid]
                ])
                
                X_norm, tar_norm = normalize_predictors(near_pred, tar_pred)
                
                try:
                    model = Ridge(alpha=0.1)
                    model.fit(X_norm, y_valid, sample_weight=w_valid)
                    lwr_val = model.predict(tar_norm.reshape(1, -1))[0]
                    
                    if np.isnan(lwr_val) or np.isinf(lwr_val):
                        pred = idw_val
                    else:
                        diff = abs(lwr_val - idw_val) / (abs(idw_val) + 0.1)
                        alpha = 0.4 if diff > 1.5 else (0.6 if diff > 0.8 else 0.85)
                        pred = alpha * lwr_val + (1 - alpha) * idw_val
                except Exception:
                    pred = idw_val
                    
        elif method == 'RF':
            max_neighbors = min(10, len(curr_idx))
            top_idx = curr_idx[:max_neighbors]
            pred = rf_estimate(prcp_masked[:, s], prcp_masked[:, top_idx], prcp_masked[t, top_idx], curr_weights[:max_neighbors])
            if np.isnan(pred) or np.isinf(pred):
                pred = idw_val
        else:
            pred = idw_val
        
        if not np.isnan(pred) and not np.isinf(pred):
            return (true_val, pred)
        return None
    
    # Ejecutar en paralelo
    results = Parallel(n_jobs=n_cores, verbose=1)(
        delayed(_cv_single)(idx) for idx in sample_idx
    )
    
    # Filtrar resultados válidos
    valid_results = [r for r in results if r is not None]
    obs_true = [r[0] for r in valid_results]
    pred_values = [r[1] for r in valid_results]
    
    print(f"  Validación completada: {len(obs_true)} muestras evaluadas")
    
    return np.array(obs_true), np.array(pred_values)
