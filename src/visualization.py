"""
Módulo de Visualización para Evaluación de Imputación

Estilo basado en demo_PyGMET_outputs.ipynb (celdas 19, 23-24)
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path


def plot_regression_comparison(ds_obs, ds_reg, varnames=['prcp_boxcox'], save=False, output_path=None):
    """
    Genera panel de comparación observado vs regresión al estilo PyGMET.
    Similar a celda 19 del demo notebook.
    
    Args:
        ds_obs: Dataset con observaciones
        ds_reg: Dataset con regresión/imputación
        varnames: Variables a comparar
        save: Si guardar figura
        output_path: Ruta de salida
        
    Returns:
        Figure si save=False
    """
    lat_stn = ds_obs.lat.values
    lon_stn = ds_obs.lon.values
    
    fig, axs = plt.subplots(len(varnames), 3, figsize=[12, 4*len(varnames)])
    
    if len(varnames) == 1:
        axs = axs[np.newaxis, :]
    
    for i, var in enumerate(varnames):
        # Calcular rango completo (sin filtrar por percentiles)
        obs_mean = ds_obs[var].mean(dim='time').values
        reg_mean = ds_reg[var].mean(dim='time').values
        
        d = obs_mean[~np.isnan(obs_mean)]
        if len(d) > 0:
            vmin = np.min(d)
            vmax = np.max(d)
        else:
            vmin, vmax = 0, 1
        
        # Panel 1: Observación
        ax = axs[i, 0]
        sc = ax.scatter(lon_stn, lat_stn, 30, obs_mean, 
                       vmin=vmin, vmax=vmax, cmap='viridis',
                       edgecolors='k', linewidths=0.3)
        plt.colorbar(sc, ax=ax)
        ax.set_title(f'{var}: observation')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Panel 2: Regresión
        ax = axs[i, 1]
        sc = ax.scatter(lon_stn, lat_stn, 30, reg_mean,
                       vmin=vmin, vmax=vmax, cmap='viridis',
                       edgecolors='k', linewidths=0.3)
        plt.colorbar(sc, ax=ax)
        ax.set_title(f'{var}: regression')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        # Panel 3: Scatter
        ax = axs[i, 2]
        ax.scatter(obs_mean, reg_mean)
        ax.plot([vmin, vmax], [vmin, vmax])
        ax.set_title(f'{var}: reg vs obs')
        ax.set_xlabel('Observation')
        ax.set_ylabel('Regression')
        ax.set_xlim([vmin, vmax])
        ax.set_ylim([vmin, vmax])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save and output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    return fig


def plot_metrics_panel(ds_metrics, varnames=['prcp'], metrics=['CC', 'RMSE', 'KGE_K2012'], 
                       save=False, output_path=None):
    """
    Genera panel de métricas espaciales al estilo PyGMET.
    Similar a celda 23-24 del demo notebook.
    
    Args:
        ds_metrics: Dataset con métricas por estación
        varnames: Variables
        metrics: Métricas a graficar
        save: Si guardar
        output_path: Ruta
        
    Returns:
        Figure si save=False
    """
    lat_stn = ds_metrics.lat.values
    lon_stn = ds_metrics.lon.values
    
    fig, axs = plt.subplots(len(varnames), len(metrics), 
                           figsize=[4*len(metrics), 4*len(varnames)])
    
    if len(varnames) == 1 and len(metrics) == 1:
        axs = np.array([[axs]])
    elif len(varnames) == 1:
        axs = axs[np.newaxis, :]
    elif len(metrics) == 1:
        axs = axs[:, np.newaxis]
    
    for i, var in enumerate(varnames):
        for j, met in enumerate(metrics):
            ax = axs[i, j]
            
            # Obtener datos de métrica
            if 'met' in ds_metrics.dims:
                met_idx = np.where(ds_metrics.met.values == met)[0]
                if len(met_idx) == 0:
                    ax.set_title(f'{var}: {met} (no data)')
                    continue
                d = ds_metrics[f'{var}_metric'].values[:, met_idx[0]]
            else:
                var_name = f'{var}_{met}'
                if var_name in ds_metrics:
                    d = ds_metrics[var_name].values
                else:
                    ax.set_title(f'{var}: {met} (no data)')
                    continue
            
            d_valid = d[~np.isnan(d)]
            if len(d_valid) == 0:
                ax.set_title(f'{var}: {met} (no valid)')
                continue
            
            # Usar rango completo (min/max) en lugar de percentiles
            vmin = np.min(d_valid)
            vmax = np.max(d_valid)
            
            sc = ax.scatter(lon_stn, lat_stn, 30, d,
                           vmin=vmin, vmax=vmax, cmap='RdYlGn',
                           edgecolors='k', linewidths=0.3)
            plt.colorbar(sc, ax=ax)
            ax.set_title(f'{var}: {met}')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
    
    plt.tight_layout()
    
    if save and output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    return fig


def plot_boxcox_comparison(ds_obs, ds_obs_trans, ds_imp, ds_imp_trans, 
                           save=False, output_path=None):
    """
    Compara prcp y prcp_boxcox en panel 2x3.
    
    Fila 0: prcp_boxcox
    Fila 1: prcp (mm)
    """
    lat_stn = ds_obs.lat.values
    lon_stn = ds_obs.lon.values
    
    # Preparar datasets para el bucle
    # varnames[0] = prcp_boxcox (ds_trans), varnames[1] = prcp (ds original)
    varnames = ['prcp_boxcox', 'prcp']
    ds_stn_list = [ds_obs_trans, ds_obs]
    ds_reg_list = [ds_imp_trans, ds_imp]
    
    fig, axs = plt.subplots(2, 3, figsize=[14, 8])
    
    for i in range(2):
        # Obtener datos para mapas (promedios por estación)
        if i == 0:  # prcp_boxcox
            obs_data = ds_obs_trans['prcp'].mean(dim='time').values
            reg_data = ds_imp_trans['prcp'].mean(dim='time').values
            # Datos individuales para scatter (TODOS los valores)
            obs_all = ds_obs_trans['prcp'].values.flatten()
            reg_all = ds_imp_trans['prcp'].values.flatten()
        else:  # prcp
            obs_data = ds_obs['prcp'].mean(dim='time').values
            reg_data = ds_imp['prcp'].mean(dim='time').values
            # Datos individuales para scatter (TODOS los valores)
            obs_all = ds_obs['prcp'].values.flatten()
            reg_all = ds_imp['prcp'].values.flatten()
        
        # Filtrar valores válidos para scatter
        valid_mask = ~np.isnan(obs_all) & ~np.isnan(reg_all)
        obs_scatter = obs_all[valid_mask]
        reg_scatter = reg_all[valid_mask]
        
        # Calcular rango completo (min/max) para mapas
        d = obs_data[~np.isnan(obs_data)]
        if len(d) > 0:
            vmin = np.min(d)
            vmax = np.max(d)
        else:
            vmin, vmax = 0, 1
        
        # Rango para scatter (usando todos los valores)
        if len(obs_scatter) > 0:
            vmin_s = np.min(obs_scatter)
            vmax_s = np.max(obs_scatter)
        else:
            vmin_s, vmax_s = vmin, vmax
        
        # Panel 1: Observation (promedios)
        axi = axs[i, 0]
        p = axi.scatter(lon_stn, lat_stn, 10, obs_data, vmin=vmin, vmax=vmax)
        plt.colorbar(p, ax=axi)
        axi.set_title(varnames[i] + ': observation')
        
        # Panel 2: Regression (promedios)
        axi = axs[i, 1]
        p = axi.scatter(lon_stn, lat_stn, 10, reg_data, vmin=vmin, vmax=vmax)
        plt.colorbar(p, ax=axi)
        axi.set_title(varnames[i] + ': regression')
        
        # Panel 3: Scatter obs vs reg (TODOS los datos individuales)
        axi = axs[i, 2]
        p = axi.scatter(obs_scatter, reg_scatter)
        # axi.plot([vmin_s, vmax_s], [vmin_s, vmax_s]) # Línea eliminada a petición del usuario
        axi.set_title(f'{varnames[i]}: reg vs ob')
        axi.set_xlabel('Observation')
        axi.set_ylabel('Regression')
    
    plt.tight_layout()
    
    if save and output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    return fig


def plot_time_series(ds_obs, ds_imp, station_indices, var='prcp', save=False, output_path=None):
    """
    Series temporales de estaciones seleccionadas.
    Muestra 2 columnas por estación: Original (azul) e Imputado (naranja).
    """
    n = len(station_indices)
    fig, axs = plt.subplots(n, 2, figsize=[16, 3*n])
    
    if n == 1:
        axs = axs[np.newaxis, :]
    
    for i, stn in enumerate(station_indices):
        obs = ds_obs[var].isel(stn=stn).values
        imp = ds_imp[var].isel(stn=stn).values
        time = ds_obs['time'].values
        stn_name = ds_obs.stn.values[stn]
        
        # Columna 1: Datos Originales (azul)
        ax = axs[i, 0]
        ax.plot(time, obs, 'b-', linewidth=0.8, alpha=0.9)
        ax.set_title(f'Estación {stn_name}: Observado')
        ax.set_ylabel(var)
        ax.grid(True, alpha=0.3)
        
        # Columna 2: Datos Imputados/Regresión (naranja)
        ax = axs[i, 1]
        ax.plot(time, imp, color='orange', linewidth=0.8, alpha=0.9)
        ax.set_title(f'Estación {stn_name}: Regresión')
        ax.set_ylabel(var)
        ax.grid(True, alpha=0.3)
    
    axs[-1, 0].set_xlabel('Tiempo')
    axs[-1, 1].set_xlabel('Tiempo')
    plt.tight_layout()
    
    if save and output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return None
    return fig


def plot_variable_comparison(ds_obs, ds_imp, ds_trans, ds_imp_trans, var_name, 
                              method_name, vcfg, figsize=(14, 8)):
    """
    Genera panel de comparación para una variable y método.
    
    Args:
        ds_obs: Dataset observado
        ds_imp: Dataset imputado
        ds_trans: Dataset observado transformado (Box-Cox)
        ds_imp_trans: Dataset imputado transformado
        var_name: Nombre de la variable
        method_name: Nombre del método
        vcfg: Configuración de la variable
        figsize: Tamaño de figura
    """
    import data_processing
    
    n_rows = 2 if vcfg['use_boxcox'] else 1
    fig, axs = plt.subplots(n_rows, 3, figsize=[figsize[0], 4*n_rows])
    if n_rows == 1:
        axs = axs.reshape(1, -1)
    
    lat = ds_obs['lat'].values
    lon = ds_obs['lon'].values
    
    # Fila 1: Variable original
    obs_mean = ds_obs[var_name].mean(dim='time').values
    imp_mean = ds_imp[var_name].mean(dim='time').values
    
    d = obs_mean[~np.isnan(obs_mean)]
    vmin, vmax = (np.min(d), np.max(d)) if len(d) > 0 else (0, 1)
    
    sc1 = axs[0, 0].scatter(lon, lat, 10, obs_mean, vmin=vmin, vmax=vmax, cmap='viridis')
    plt.colorbar(sc1, ax=axs[0, 0])
    axs[0, 0].set_title(f'{var_name}: Observación')
    axs[0, 0].set_xlabel('Longitud')
    axs[0, 0].set_ylabel('Latitud')
    
    sc2 = axs[0, 1].scatter(lon, lat, 10, imp_mean, vmin=vmin, vmax=vmax, cmap='viridis')
    plt.colorbar(sc2, ax=axs[0, 1])
    axs[0, 1].set_title(f'{var_name}: Imputado')
    axs[0, 1].set_xlabel('Longitud')
    axs[0, 1].set_ylabel('Latitud')
    
    # Scatter
    obs_all = ds_obs[var_name].values.flatten()
    imp_all = ds_imp[var_name].values.flatten()
    valid = ~np.isnan(obs_all) & ~np.isnan(imp_all)
    axs[0, 2].scatter(obs_all[valid], imp_all[valid])
    axs[0, 2].plot([vmin, vmax], [vmin, vmax])
    axs[0, 2].set_title(f'{var_name}: Imputado vs Obs')
    axs[0, 2].set_xlabel('Observación')
    axs[0, 2].set_ylabel('Imputado')
    
    # Fila 2: Box-Cox
    if vcfg['use_boxcox'] and ds_trans is not None:
        obs_t_mean = ds_trans[var_name].mean(dim='time').values
        
        if ds_imp_trans is None:
            ds_imp_trans = ds_imp.copy()
            ds_imp_trans[var_name].values = data_processing.boxcox_transform(
                ds_imp[var_name].values, vcfg['exp']
            )
        
        imp_t_mean = ds_imp_trans[var_name].mean(dim='time').values
        
        d_t = obs_t_mean[~np.isnan(obs_t_mean)]
        vmin_t, vmax_t = (np.min(d_t), np.max(d_t)) if len(d_t) > 0 else (-4, 4)
        
        sc3 = axs[1, 0].scatter(lon, lat, 10, obs_t_mean, vmin=vmin_t, vmax=vmax_t, cmap='viridis')
        plt.colorbar(sc3, ax=axs[1, 0])
        axs[1, 0].set_title(f'{var_name}_boxcox: Observación')
        axs[1, 0].set_xlabel('Longitud')
        axs[1, 0].set_ylabel('Latitud')
        
        sc4 = axs[1, 1].scatter(lon, lat, 10, imp_t_mean, vmin=vmin_t, vmax=vmax_t, cmap='viridis')
        plt.colorbar(sc4, ax=axs[1, 1])
        axs[1, 1].set_title(f'{var_name}_boxcox: Regresión')
        axs[1, 1].set_xlabel('Longitud')
        axs[1, 1].set_ylabel('Latitud')
        
        obs_t_all = ds_trans[var_name].values.flatten()
        imp_t_all = ds_imp_trans[var_name].values.flatten()
        valid_t = ~np.isnan(obs_t_all) & ~np.isnan(imp_t_all)
        axs[1, 2].scatter(obs_t_all[valid_t], imp_t_all[valid_t])
        axs[1, 2].plot([vmin_t, vmax_t], [vmin_t, vmax_t])
        axs[1, 2].set_title(f'{var_name}_boxcox: Reg vs Obs')
        axs[1, 2].set_xlabel('Observación')
        axs[1, 2].set_ylabel('Regresión')
    
    plt.tight_layout()
    plt.suptitle(f'{var_name.upper()} - Método: {method_name}', y=1.02, fontsize=14)
    plt.show()
    
    return fig


def plot_all_results(config, all_results):
    """
    Visualiza todas las variables y métodos.
    
    Args:
        config: Configuración global
        all_results: Diccionario con todos los resultados
    """
    import imputation_pipeline as pipeline
    
    print("\n=== VISUALIZACIÓN POR VARIABLE ===\n")
    
    for i, var_name in enumerate(config['variables']):
        vcfg = pipeline.get_var_config(config, i)
        
        ds_obs = all_results[var_name]['ds_original']
        ds_trans = all_results[var_name].get('ds_trans')
        
        for method in config['methods']:
            if method not in all_results[var_name]['results']:
                continue
            
            print(f"\n--- {var_name}: {method} ---")
            
            ds_imp = all_results[var_name]['results'][method]['ds']
            ds_imp_trans = all_results[var_name]['results'][method].get('ds_trans')
            
            plot_variable_comparison(
                ds_obs, ds_imp, ds_trans, ds_imp_trans,
                var_name, method, vcfg
            )


def plot_all_time_series(config, all_results, n_stations=3):
    """
    Genera series temporales para todas las variables.
    
    Args:
        config: Configuración global
        all_results: Diccionario con todos los resultados
        n_stations: Número de estaciones por variable
    """
    print("\n=== SERIES TEMPORALES ===\n")
    
    for var_name in config['variables']:
        ds_obs = all_results[var_name]['ds_original']
        n_stns = ds_obs.sizes['stn']
        sample_idx = np.linspace(0, n_stns-1, min(n_stations, n_stns), dtype=int)
        
        for method in config['methods']:
            if method not in all_results[var_name]['results']:
                continue
            
            print(f"\n--- {var_name}: {method} ---")
            
            ds_imp = all_results[var_name]['results'][method]['ds']
            
            plot_time_series(ds_obs, ds_imp, sample_idx, var=var_name)
            plt.suptitle(f'{var_name.upper()} - {method}', y=1.02)
            plt.show()


if __name__ == "__main__":
    print("Módulo de visualización estilo PyGMET")

