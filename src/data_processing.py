# todas las funciones relacionadas con el procesamiento de datos

import os, time, sys
import pandas as pd
import numpy as np
import xarray as xr

########################################################################################################################
# transformación de datos

def boxcox_transform(data, exp=0.25):
    # transformar prcp para aproximar la distribución normal
    # modo: box-cox; ley de potencia
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    data[data<0] = 0
    datat = (data ** exp - 1) / exp
    # datat[data < -3] = -3
    return datat


def boxcox_retransform(data, exp=0.25):
    # transformar prcp para aproximar la distribución normal
    # modo: box-cox; ley de potencia
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    data[data<-1/exp] = -1/exp
    datat = (data * exp + 1) ** (1/exp)
    return datat

def data_transformation(data, method, settings, mode='transform'):
    if method == 'boxcox':
        if mode == 'transform':
            data = boxcox_transform(data, settings['exp'])
        elif mode == 'retransform':
            data = boxcox_retransform(data, settings['exp'])
        else:
            sys.exit('Modo de transformación desconocido')
    else:
        sys.exit('Método de transformación desconocido')
    return data


def load_precipitation_data(csv_path):
    """
    Carga datos de precipitación desde un archivo CSV con formato específico.
    
    El archivo tiene 5 filas de encabezado:
    1. longitude
    2. latitude
    3. elevation
    4. name_station
    5. id_station
    
    A partir de la fila 6 están los datos. La primera columna es la fecha.
    Los valores decimales usan coma (',') y deben convertirse a punto ('.').
    """
    print(f"Cargando datos desde: {csv_path}")
    
    # Leer metadatos (primeras 5 filas)
    # Usamos header=None para leer todo como strings inicialmente y evitar problemas de tipos
    df_meta = pd.read_csv(csv_path, nrows=5, header=None)
    
    # Extraer metadatos. Asumimos que la primera columna es la etiqueta y las siguientes son las estaciones
    # Transponemos para tener estaciones como filas
    meta_values = df_meta.iloc[:, 1:].T
    meta_values.columns = ['longitude', 'latitude', 'elevation', 'name_station', 'id_station']
    
    # Convertir tipos numéricos
    meta_values['longitude'] = pd.to_numeric(meta_values['longitude'], errors='coerce')
    meta_values['latitude'] = pd.to_numeric(meta_values['latitude'], errors='coerce')
    meta_values['elevation'] = pd.to_numeric(meta_values['elevation'], errors='coerce')
    
    # Leer datos de series temporales (saltando las primeras 5 filas)
    # La primera columna es la fecha
    df_data = pd.read_csv(csv_path, skiprows=5, header=None)
    
    # Renombrar la primera columna a 'time'
    df_data.rename(columns={0: 'time'}, inplace=True)
    
    # Convertir la columna de fecha a datetime
    # El formato parece ser D/M/Y según la vista previa (e.g., 1/1/1980)
    df_data['time'] = pd.to_datetime(df_data['time'], format='%d/%m/%Y', errors='coerce')
    
    # Establecer índice temporal
    df_data.set_index('time', inplace=True)
    
    # Reemplazar comas por puntos y convertir a float
    # Aplicamos esto a todas las columnas de datos
    for col in df_data.columns:
        if df_data[col].dtype == object:
            df_data[col] = df_data[col].astype(str).str.replace(',', '.').astype(float)
            
    # Crear xarray Dataset
    # Dimensiones: time, stn
    
    # IDs de estaciones
    stn_ids = meta_values['id_station'].values
    
    # Crear DataArrays para las coordenadas de las estaciones
    lat_da = xr.DataArray(meta_values['latitude'].values, dims=['stn'], coords={'stn': stn_ids})
    lon_da = xr.DataArray(meta_values['longitude'].values, dims=['stn'], coords={'stn': stn_ids})
    elev_da = xr.DataArray(meta_values['elevation'].values, dims=['stn'], coords={'stn': stn_ids})
    
    # Datos de precipitación
    # df_data tiene columnas 1, 2, 3... que corresponden a las filas de meta_values
    # Necesitamos asegurar que el orden coincida.
    # df_data.columns son enteros 1, 2, ... N
    # meta_values index es 1, 2, ... N (si no se reseteó el índice)
    
    prcp_values = df_data.values # shape (time, stn)
    
    prcp_da = xr.DataArray(
        prcp_values,
        dims=['time', 'stn'],
        coords={
            'time': df_data.index,
            'stn': stn_ids
        },
        name='prcp'
    )
    
    # Crear Dataset
    ds = xr.Dataset({
        'prcp': prcp_da,
        'lat': lat_da,
        'lon': lon_da,
        'elev': elev_da
    })
    
    print(f"Datos cargados. Dimensiones: {ds.sizes}")
    return ds

def check_and_clamp_data(ds, config, var_name='prcp'):
    """
    Verifica y ajusta los datos de entrada según los rangos definidos en config.
    
    Args:
        ds: xr.Dataset con la variable a verificar
        config: dict con 'minRange_vars' y 'maxRange_vars'
        var_name: Nombre de la variable a verificar
    """
    target_vars = [var_name]
    
    if 'minRange_vars' in config:
        minRange_vars = config['minRange_vars']
        if isinstance(minRange_vars, (int, float)):
            minRange_vars = [minRange_vars] * len(target_vars)
    else:
        minRange_vars = [-np.inf] * len(target_vars)

    if 'maxRange_vars' in config:
        maxRange_vars = config['maxRange_vars']
        if isinstance(maxRange_vars, (int, float)):
            maxRange_vars = [maxRange_vars] * len(target_vars)
    else:
        maxRange_vars = [np.inf] * len(target_vars)
        
    for i, vari in enumerate(target_vars):
        if vari in ds.data_vars:
            v = ds[vari].values
            if np.any(v < minRange_vars[i]):
                print(f'Los datos de {vari} tienen valores < {minRange_vars[i]}. Ajustándolos.')
                v[v < minRange_vars[i]] = minRange_vars[i]
            if np.any(v > maxRange_vars[i]):
                print(f'Los datos de {vari} tienen valores > {maxRange_vars[i]}. Ajustándolos.')
                v[v > maxRange_vars[i]] = maxRange_vars[i]
            ds[vari].values = v
            
    return ds


def load_meteorological_data(csv_path, var_name='prcp'):
    """
    Carga datos meteorológicos genéricos desde un archivo CSV.
    
    Soporta: prcp, tmax, tmin, flow
    
    El archivo tiene 5 filas de encabezado:
    1. longitude
    2. latitude
    3. elevation
    4. name_station
    5. id_station
    
    A partir de la fila 6 están los datos. La primera columna es la fecha.
    
    Args:
        csv_path: Ruta al archivo CSV
        var_name: Nombre de la variable ('prcp', 'tmax', 'tmin', 'flow')
        
    Returns:
        xr.Dataset con coordenadas (time, stn) y variable var_name
    """
    print(f"Cargando datos de {var_name} desde: {csv_path}")
    
    # Leer metadatos (primeras 5 filas)
    df_meta = pd.read_csv(csv_path, nrows=5, header=None)
    
    # Extraer metadatos
    meta_values = df_meta.iloc[:, 1:].T
    meta_values.columns = ['longitude', 'latitude', 'elevation', 'name_station', 'id_station']
    
    # Convertir tipos numéricos
    meta_values['longitude'] = pd.to_numeric(meta_values['longitude'], errors='coerce')
    meta_values['latitude'] = pd.to_numeric(meta_values['latitude'], errors='coerce')
    meta_values['elevation'] = pd.to_numeric(meta_values['elevation'], errors='coerce')
    
    # Leer datos de series temporales
    df_data = pd.read_csv(csv_path, skiprows=5, header=None)
    df_data.rename(columns={0: 'time'}, inplace=True)
    df_data['time'] = pd.to_datetime(df_data['time'], format='%d/%m/%Y', errors='coerce')
    df_data.set_index('time', inplace=True)
    
    # Convertir valores a float
    for col in df_data.columns:
        if df_data[col].dtype == object:
            df_data[col] = df_data[col].astype(str).str.replace(',', '.').astype(float)
    
    # IDs de estaciones
    stn_ids = meta_values['id_station'].values
    
    # Crear DataArrays
    lat_da = xr.DataArray(meta_values['latitude'].values, dims=['stn'], coords={'stn': stn_ids})
    lon_da = xr.DataArray(meta_values['longitude'].values, dims=['stn'], coords={'stn': stn_ids})
    elev_da = xr.DataArray(meta_values['elevation'].values, dims=['stn'], coords={'stn': stn_ids})
    
    # DataArray para la variable
    var_da = xr.DataArray(
        df_data.values,
        dims=['time', 'stn'],
        coords={
            'time': df_data.index,
            'stn': stn_ids
        },
        name=var_name
    )
    
    # Crear Dataset
    ds = xr.Dataset({
        var_name: var_da,
        'lat': lat_da,
        'lon': lon_da,
        'elev': elev_da
    })
    
    print(f"  Dimensiones: {ds.sizes}, Variable: {var_name}")
    return ds
