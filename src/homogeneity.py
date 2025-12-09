"""
Módulo de Pruebas de Homogeneidad para Series Temporales.

Implementa pruebas de Buishand U-test y SNHT para detectar
puntos de quiebre en series de datos meteorológicos.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import pyhomogeneity as hg
    HAS_PYHOMOGENEITY = True
except ImportError:
    HAS_PYHOMOGENEITY = False
    print("Warning: pyhomogeneity no instalado. Instalar con: pip install pyhomogeneity")


def test_homogeneity(series, alpha=0.05):
    """
    Realiza pruebas de homogeneidad en una serie temporal.
    
    Args:
        series: pd.Series con índice temporal
        alpha: Nivel de significancia (default 0.05)
        
    Returns:
        dict: Resultados de las pruebas
    """
    if not HAS_PYHOMOGENEITY:
        return {'error': 'pyhomogeneity no instalado'}
    
    series_clean = series.dropna()
    if len(series_clean) < 10:
        return {'error': 'Datos insuficientes', 'n_samples': len(series_clean)}
    
    # Pruebas
    buishand = hg.buishand_u_test(series_clean, alpha)
    snht = hg.snht_test(series_clean)
    
    # Extraer resultados
    is_homogeneous = not (buishand.h or snht.h)
    
    result = {
        'is_homogeneous': is_homogeneous,
        'buishand_h': buishand.h,
        'buishand_p': buishand.p,
        'buishand_cp': buishand.cp,
        'snht_h': snht.h,
        'snht_p': snht.p if hasattr(snht, 'p') else None,
        'n_samples': len(series_clean),
        'mean': np.nanmean(series_clean),
        'std': np.nanstd(series_clean),
    }
    
    # Medias antes/después del punto de quiebre
    if hasattr(buishand.avg, 'mu1'):
        result['mu1'] = buishand.avg.mu1
        result['mu2'] = buishand.avg.mu2
    
    return result


def get_ylabel(var_name):
    """Retorna etiqueta Y según variable."""
    labels = {
        'prcp': 'Precipitación (mm)',
        'tmax': 'Temperatura Máxima (°C)',
        'tmin': 'Temperatura Mínima (°C)',
        'flow': 'Caudal (m³/s)',
    }
    return labels.get(var_name, var_name)


def plot_homogeneity_panel(series_list, stn_names, title, freq_label, var_name='prcp',
                            save_path=None, figsize_per_stn=(14, 3)):
    """
    Grafica pruebas de homogeneidad para múltiples estaciones en una figura.
    
    Args:
        series_list: Lista de pd.Series (una por estación)
        stn_names: Lista de nombres de estaciones
        title: Título general de la figura
        freq_label: Etiqueta del eje X (Día, Mes, Año)
        var_name: Nombre de la variable
        save_path: Ruta para guardar
        figsize_per_stn: Tamaño por estación (ancho, alto)
        
    Returns:
        list: Lista de resultados por estación
    """
    if not HAS_PYHOMOGENEITY:
        print(f"  pyhomogeneity no disponible")
        return [None] * len(series_list)
    
    n_stns = len(series_list)
    fig, axs = plt.subplots(n_stns, 1, figsize=(figsize_per_stn[0], figsize_per_stn[1] * n_stns))
    
    if n_stns == 1:
        axs = [axs]
    
    results = []
    
    for i, (series, stn_name) in enumerate(zip(series_list, stn_names)):
        ax = axs[i]
        
        series_clean = series.dropna()
        
        if len(series_clean) < 10:
            ax.text(0.5, 0.5, f'{stn_name}: Datos insuficientes', 
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{stn_name}')
            results.append(None)
            continue
        
        # Pruebas de homogeneidad
        buishand = hg.buishand_u_test(series_clean, 0.05)
        snht = hg.snht_test(series_clean)
        
        # Valores para gráfico
        mn = series_clean.index[0]
        mx = series_clean.index[-1]
        loc = pd.to_datetime(buishand.cp) if buishand.cp is not None else mn
        mu1 = buishand.avg.mu1 if hasattr(buishand.avg, 'mu1') else np.nanmean(series_clean)
        mu2 = buishand.avg.mu2 if hasattr(buishand.avg, 'mu2') else np.nanmean(series_clean)
        mean_val = np.nanmean(series_clean)
        
        is_homogeneous = not (buishand.h or snht.h)
        
        if not is_homogeneous:
            status = 'No Homogéneo'
            status_color = 'red'
            ax.plot(series_clean.index, series_clean.values, 
                    label='Observación', marker='o', markersize=1, linewidth=0.5)
            ax.hlines(mu1, xmin=mn, xmax=loc, linestyles='--', colors='orange', lw=1.5, 
                       label=f'μ1: {mu1:.2f}')
            ax.hlines(mu2, xmin=loc, xmax=mx, linestyles='--', colors='g', lw=1.5, 
                       label=f'μ2: {mu2:.2f}')
            ax.axvline(x=loc, linestyle='-.', color='red', lw=1.5, 
                        label=f'Quiebre: {loc.strftime("%Y-%m-%d")}')
        else:
            status = 'Homogéneo'
            status_color = 'green'
            ax.plot(series_clean.index, series_clean.values, 
                    label='Observación', marker='o', markersize=1, linewidth=0.5)
            ax.hlines(mean_val, xmin=mn, xmax=mx, linestyles='--', colors='purple', lw=1.5, 
                       label=f'μ: {mean_val:.2f}')
        
        ax.set_title(f'{stn_name} - {status} (p={buishand.p:.4f})', 
                     fontweight='bold', color=status_color)
        ax.set_ylabel(get_ylabel(var_name))
        ax.legend(loc='upper right', frameon=False, fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        
        results.append({
            'is_homogeneous': is_homogeneous,
            'buishand_p': buishand.p,
            'breakpoint': loc if not is_homogeneous else None,
            'mu1': mu1,
            'mu2': mu2,
        })
    
    axs[-1].set_xlabel(freq_label)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return results


def analyze_variable_homogeneity(ds, var_name, sample_idx, method_name='',
                                   frequencies=['D', 'ME', 'YE'], save_dir=None):
    """
    Analiza homogeneidad de múltiples estaciones a cada frecuencia.
    Genera UNA figura por frecuencia con TODAS las estaciones.
    
    Args:
        ds: Dataset con datos imputados
        var_name: Nombre de la variable
        sample_idx: Índices de estaciones a analizar
        method_name: Nombre del método
        frequencies: Lista de frecuencias
        save_dir: Directorio para guardar
        
    Returns:
        dict: Resultados por frecuencia y estación
    """
    freq_labels = {'D': 'Día', 'ME': 'Mes', 'YE': 'Año'}
    freq_names = {'D': 'Diario', 'ME': 'Mensual', 'YE': 'Anual'}
    
    # Agregación según variable
    agg_func = 'sum' if var_name in ['prcp'] else 'mean'
    
    # Obtener nombres de estaciones
    stn_names = [str(ds.stn.values[idx]) for idx in sample_idx]
    
    results = {}
    
    for freq in frequencies:
        freq_label = freq_labels.get(freq, freq)
        freq_name = freq_names.get(freq, freq)
        
        print(f'\n  Análisis {freq_name} ({len(sample_idx)} estaciones)...')
        
        # Preparar series para todas las estaciones
        series_list = []
        for stn_idx in sample_idx:
            series = ds[var_name].isel(stn=stn_idx).to_pandas()
            
            if freq == 'D':
                series_agg = series
            else:
                series_agg = series.resample(freq).agg(agg_func)
            
            series_list.append(series_agg)
        
        # Título de la figura
        title = f'{var_name.upper()} - {freq_name}'
        if method_name:
            title += f' ({method_name})'
        
        save_path = None
        if save_dir:
            save_path = f'{save_dir}/{var_name}_{method_name}_{freq}.png'
        
        # Graficar todas las estaciones en una figura
        freq_results = plot_homogeneity_panel(
            series_list, stn_names, title, freq_label, var_name, save_path
        )
        
        # Organizar resultados por estación
        results[freq] = {stn: res for stn, res in zip(stn_names, freq_results)}
    
    return results


def analyze_all_variables(config, all_results, n_stations=3, frequencies=['D', 'ME', 'YE']):
    """
    Analiza homogeneidad para todas las variables.
    Genera una figura por frecuencia con todas las estaciones seleccionadas.
    
    Args:
        config: Configuración del proyecto
        all_results: Diccionario con todos los resultados
        n_stations: Número de estaciones a analizar por variable
        frequencies: Frecuencias de análisis
        
    Returns:
        dict: Resultados organizados por variable/método/frecuencia/estación
    """
    all_homogeneity = {}
    
    for i, var_name in enumerate(config['variables']):
        print(f"\n{'='*60}")
        print(f"  HOMOGENEIDAD: {var_name.upper()}")
        print(f"{'='*60}")
        
        var_results = {}
        
        for method in config['methods']:
            if method not in all_results[var_name]['results']:
                continue
            
            print(f"\n  Método: {method}")
            
            ds_imp = all_results[var_name]['results'][method]['ds']
            n_stns = ds_imp.sizes['stn']
            
            # Seleccionar estaciones distribuidas
            sample_idx = np.linspace(0, n_stns-1, min(n_stations, n_stns), dtype=int)
            
            print(f"  Estaciones: {[str(ds_imp.stn.values[idx]) for idx in sample_idx]}")
            
            # Analizar todas las estaciones para esta variable/método
            method_results = analyze_variable_homogeneity(
                ds_imp, var_name, sample_idx, method, frequencies
            )
            
            var_results[method] = method_results
        
        all_homogeneity[var_name] = var_results
    
    return all_homogeneity


def summarize_homogeneity(homogeneity_results):
    """
    Resume resultados de homogeneidad en una tabla.
    
    Args:
        homogeneity_results: Diccionario de resultados
        
    Returns:
        pd.DataFrame: Tabla resumen
    """
    rows = []
    
    for var_name, var_data in homogeneity_results.items():
        for method, method_data in var_data.items():
            for freq, freq_data in method_data.items():
                if freq_data is None:
                    continue
                for stn_name, stn_result in freq_data.items():
                    if stn_result is None:
                        continue
                    rows.append({
                        'Variable': var_name,
                        'Método': method,
                        'Frecuencia': freq,
                        'Estación': stn_name,
                        'Homogéneo': stn_result.get('is_homogeneous', None),
                        'p-value': stn_result.get('buishand_p', None),
                    })
    
    return pd.DataFrame(rows)
