"""
Módulo de Procesamiento de DEM para Extracción de Predictores Estáticos

Extrae elevación y pendientes del DEM para usar como predictores en la regresión.
"""

import numpy as np
import xarray as xr

try:
    import rasterio
    from rasterio.transform import rowcol
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Advertencia: rasterio no disponible. Usando elevación de estaciones.")


def extract_dem_at_stations(dem_path, lat, lon):
    """
    Extrae valores de elevación del DEM en las ubicaciones de las estaciones.
    
    Args:
        dem_path (str): Ruta al archivo DEM (GeoTIFF)
        lat (np.ndarray): Latitudes de las estaciones
        lon (np.ndarray): Longitudes de las estaciones
        
    Returns:
        np.ndarray: Elevaciones del DEM en cada estación
    """
    if not HAS_RASTERIO:
        return None
    
    try:
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)
            transform = src.transform
            nodata = src.nodata
            
            elev_dem = np.zeros(len(lat))
            
            for i in range(len(lat)):
                try:
                    # Convertir lat/lon a índices de fila/columna
                    row, col = rowcol(transform, lon[i], lat[i])
                    
                    # Verificar límites
                    if 0 <= row < dem_data.shape[0] and 0 <= col < dem_data.shape[1]:
                        val = dem_data[row, col]
                        if nodata is not None and val == nodata:
                            elev_dem[i] = np.nan
                        else:
                            elev_dem[i] = val
                    else:
                        elev_dem[i] = np.nan
                except Exception:
                    elev_dem[i] = np.nan
            
            return elev_dem
            
    except Exception as e:
        print(f"Error leyendo DEM: {e}")
        return None


def calculate_slopes(dem_path, lat, lon, cell_size_km=1.0):
    """
    Calcula pendientes Norte-Sur (slp_n) y Este-Oeste (slp_e) del DEM
    en las ubicaciones de las estaciones.
    
    slp_n: Pendiente en dirección norte (positivo = sube hacia el norte)
    slp_e: Pendiente en dirección este (positivo = sube hacia el este)
    
    Args:
        dem_path (str): Ruta al archivo DEM
        lat (np.ndarray): Latitudes de las estaciones
        lon (np.ndarray): Longitudes de las estaciones
        cell_size_km (float): Tamaño de celda para calcular gradiente (km)
        
    Returns:
        tuple: (slp_n, slp_e) arrays de pendientes
    """
    if not HAS_RASTERIO:
        return None, None
    
    try:
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1).astype(float)
            transform = src.transform
            nodata = src.nodata
            
            # Reemplazar nodata con NaN
            if nodata is not None:
                dem_data[dem_data == nodata] = np.nan
            
            # Calcular gradientes usando numpy
            # Tamaño de pixel en metros (aproximado)
            pixel_size_x = abs(transform[0])  # grados
            pixel_size_y = abs(transform[4])  # grados
            
            # Convertir grados a metros (aproximado)
            # 1 grado ≈ 111 km en latitud
            meters_per_degree_lat = 111000
            meters_per_degree_lon = 111000 * np.cos(np.radians(np.mean(lat)))
            
            dx = pixel_size_x * meters_per_degree_lon
            dy = pixel_size_y * meters_per_degree_lat
            
            # Calcular gradientes
            # gradient[0] = dz/dy (norte-sur)
            # gradient[1] = dz/dx (este-oeste)
            grad_y, grad_x = np.gradient(dem_data, dy, dx)
            
            slp_n = np.zeros(len(lat))
            slp_e = np.zeros(len(lat))
            
            for i in range(len(lat)):
                try:
                    row, col = rowcol(transform, lon[i], lat[i])
                    
                    if 0 <= row < grad_y.shape[0] and 0 <= col < grad_y.shape[1]:
                        slp_n[i] = -grad_y[row, col]  # Negativo porque row aumenta al sur
                        slp_e[i] = grad_x[row, col]
                    else:
                        slp_n[i] = np.nan
                        slp_e[i] = np.nan
                except Exception:
                    slp_n[i] = np.nan
                    slp_e[i] = np.nan
            
            return slp_n, slp_e
            
    except Exception as e:
        print(f"Error calculando pendientes: {e}")
        return None, None


def add_dem_predictors_to_dataset(ds, dem_path):
    """
    Añade predictores del DEM (elev_dem, slp_n, slp_e) al dataset de estaciones.
    
    Args:
        ds (xr.Dataset): Dataset con coordenadas lat, lon, elev
        dem_path (str): Ruta al archivo DEM
        
    Returns:
        xr.Dataset: Dataset con nuevas variables añadidas
    """
    lat = ds['lat'].values
    lon = ds['lon'].values
    
    print(f"Extrayendo predictores del DEM: {dem_path}")
    
    # Extraer elevación del DEM
    elev_dem = extract_dem_at_stations(dem_path, lat, lon)
    
    if elev_dem is not None:
        # Usar elevación del DEM, rellenar NaN con elevación de estación
        elev_stn = ds['elev'].values
        mask_nan = np.isnan(elev_dem)
        elev_dem[mask_nan] = elev_stn[mask_nan]
        ds['elev_dem'] = xr.DataArray(elev_dem, dims=['stn'])
        print(f"  elev_dem: rango [{np.nanmin(elev_dem):.1f}, {np.nanmax(elev_dem):.1f}] m")
    else:
        # Fallback: usar elevación de estación
        ds['elev_dem'] = ds['elev'].copy()
        print("  elev_dem: usando elevación de estaciones (DEM no disponible)")
    
    # Calcular pendientes
    slp_n, slp_e = calculate_slopes(dem_path, lat, lon)
    
    if slp_n is not None and slp_e is not None:
        # Reemplazar NaN con 0 (terreno plano)
        slp_n[np.isnan(slp_n)] = 0
        slp_e[np.isnan(slp_e)] = 0
        ds['slp_n'] = xr.DataArray(slp_n, dims=['stn'])
        ds['slp_e'] = xr.DataArray(slp_e, dims=['stn'])
        print(f"  slp_n: rango [{np.nanmin(slp_n):.4f}, {np.nanmax(slp_n):.4f}]")
        print(f"  slp_e: rango [{np.nanmin(slp_e):.4f}, {np.nanmax(slp_e):.4f}]")
    else:
        # Fallback: pendientes cero
        ds['slp_n'] = xr.DataArray(np.zeros(len(lat)), dims=['stn'])
        ds['slp_e'] = xr.DataArray(np.zeros(len(lat)), dims=['stn'])
        print("  slp_n, slp_e: usando 0 (pendientes no disponibles)")
    
    return ds


if __name__ == "__main__":
    print("Módulo de procesamiento de DEM")
    print(f"rasterio disponible: {HAS_RASTERIO}")
