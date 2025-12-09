import os
import sys, time
import xarray as xr
import numpy as np
import multiprocessing

# def distance(lat1, lon1, lat2, lon2):
# # distancia de lat/lon a km
#     radius = 6371  # km
#     dlat = np.radians(lat2 - lat1)
#     dlon = np.radians(lon2 - lon1)
#     a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(np.radians(lat1)) \
#         * np.cos(np.radians(lat2)) * np.sin(dlon / 2) * np.sin(dlon / 2)
#     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
#     d = radius * c
#     return d

def distance(lat1, lon1, lat2, lon2):
    # distancia de lat/lon a km
    lat1r, lon1r, lat2r, lon2r = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    d = ((180 * 60) / np.pi) * (2 * np.arcsin(np.sqrt((np.sin((lat1r - lat2r) / 2)) ** 2 + np.cos(lat1r) * np.cos(lat2r) * (np.sin((lon1r - lon2r) / 2)) ** 2)))
    d = d * 1.852 # milla náutica a km
    return d

def find_nearstn_for_one_target(lat_tar, lon_tar, lat_stn, lon_stn, try_radius, initial_radius, nearstn_min, nearstn_max):
    # lat_tar/lon_tar: un valor
    # lat_stn/lon_stn: vector lat/lon de estaciones
    # try_radius: un radio grande que depende de la densidad de la estación. esto ayuda a reducir el tiempo de cálculo
    # initial_radius: radio inicial para encontrar estaciones
    # el número de estaciones cercanas será >= nearstn_min y <= nearstn_max

    # criterios: (1) encontrar todas las estaciones dentro del radio inicial, (2) si el número < nearstn_min, encontrar nearstn_min estaciones cercanas
    # sin considerar el radio inicial

    near_index = -99999 * np.ones(nearstn_max, dtype=int)
    near_dist = np.nan * np.ones(nearstn_max, dtype=np.float32)
    stnID = np.arange(len(lat_stn))

    # control básico para reducir el número de estaciones de entrada
    try_index = (np.abs(lat_stn - lat_tar) < try_radius) & (np.abs(lon_stn - lon_tar) < try_radius)
    lat_stn_try = lat_stn[try_index]
    lon_stn_try = lon_stn[try_index]
    stnID_try = stnID[try_index]

    # calcular distancia (km)
    dist_try = distance(lat_tar, lon_tar, lat_stn_try, lon_stn_try)
    index_use = (dist_try <= initial_radius)
    nstn = np.sum(index_use)
    if nstn >= nearstn_max:  # strategy-1
        dist_try = dist_try[index_use]  # delete redundant stations
        stnID_try = stnID_try[index_use]
        index_final = np.argsort(dist_try)[:nearstn_max]
        near_index[0:nearstn_max] = stnID_try[index_final]
        near_dist[0:nearstn_max] = dist_try[index_final]
    else:  # strategy-2
        dist = distance(lat_tar, lon_tar, lat_stn, lon_stn)
        index_use = dist <= initial_radius
        if np.sum(index_use) >= nearstn_min:
            stnID = stnID[index_use]
            dist = dist[index_use]
            nearstn_use = min(len(stnID), nearstn_max)
        else:
            nearstn_use = nearstn_min

        index_final = np.argsort(dist)[:nearstn_use]
        near_index[0:nearstn_use] = stnID[index_final]
        near_dist[0:nearstn_use] = dist[index_final]

    return near_index, near_dist

# versión paralela
def find_nearstn_for_Grids(lat_stn, lon_stn, lat_grid, lon_grid, mask_grid, try_radius, nearstn_min, nearstn_max, num_processes, initial_distance):
    if lat_grid.ndim != 2:
        sys.exit('¡Error! ¡Dimensión incorrecta de lat_grid!')

    # lon_stn/lat_stn puede contener nan

    # umbral de distancia simple
    try_radius = try_radius / 100  # intentar dentro de este grado (asumir 1 grado ~= 100 km). si falla, expandir a todas las estaciones.

    # inicialización
    nrows, ncols = np.shape(lat_grid)
    nearIndex = -99999 * np.ones([nrows, ncols, nearstn_max], dtype=int)
    nearDistance = np.nan * np.ones([nrows, ncols, nearstn_max], dtype=np.float32)

    # dividir la cuadrícula en trozos para el procesamiento en paralelo
    chunk_size = nrows // num_processes
    chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_processes)]
    chunks[-1] = (chunks[-1][0], nrows)  # el último trozo puede ser más grande si nrows no es un múltiplo de num_processes

    # procesar cada trozo en paralelo
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = []
        for chunk in chunks:
            result = pool.apply_async(process_chunk, (chunk, lat_stn, lon_stn, lat_grid, lon_grid, mask_grid, try_radius, nearstn_min, nearstn_max, initial_distance))
            results.append(result)

        for result, chunk in zip(results, chunks):
            chunk_ni, chunk_nd = result.get()
            nearIndex[chunk[0]:chunk[1], :, :] = chunk_ni
            nearDistance[chunk[0]:chunk[1], :, :] = chunk_nd

    return nearIndex, nearDistance

def process_chunk(chunk, lat_stn, lon_stn, lat_grid, lon_grid, mask_grid, try_radius, nearstn_min, nearstn_max, initial_distance):
    nearIndex = -99999 * np.ones([chunk[1]-chunk[0], lat_grid.shape[1], nearstn_max], dtype=int)
    nearDistance = np.nan * np.ones([chunk[1]-chunk[0], lat_grid.shape[1], nearstn_max], dtype=np.float32)

    for rr in range(chunk[0], chunk[1]):
        for cc in range(lat_grid.shape[1]):
            if mask_grid[rr, cc] == 1:
                ni, nd = find_nearstn_for_one_target(lat_grid[rr, cc], lon_grid[rr, cc], lat_stn, lon_stn, try_radius, initial_distance, nearstn_min, nearstn_max)
                nearIndex[rr - chunk[0], cc, :], nearDistance[rr - chunk[0], cc, :] = ni, nd

    return nearIndex, nearDistance

def find_nearstn_for_InStn(lat_stn, lon_stn, try_radius, nearstn_min, nearstn_max, initial_distance):
    # InStn: estaciones de entrada mismas
    # lon_stn/lat_stn puede contener nan
    # t1 = time.time()
    # print(f'Encontrar estación cercana para estaciones de entrada')

    # umbral de distancia simple
    try_radius = try_radius / 100  # intentar dentro de este grado (asumir 1 grado ~= 100 km). si falla, expandir a todas las estaciones.

    # inicialización
    nstn = len(lon_stn)
    nearIndex = -99999 * np.ones([nstn, nearstn_max], dtype=int)
    nearDistance = np.nan * np.ones([nstn, nearstn_max], dtype=np.float32)

    for i in range(nstn):
        lat_stni = lat_stn.copy()
        lon_stni = lon_stn.copy()
        lat_stni[i] = np.nan
        lon_stni[i] = np.nan
        if ~np.isnan(lat_stn[i]):
                nearIndex[i, :], nearDistance[i, :] = find_nearstn_for_one_target(lat_stn[i], lon_stn[i], lat_stni, lon_stni, try_radius, initial_distance, nearstn_min, nearstn_max)
    # t2 = time.time()
    # print('Costo de tiempo (segundos):', t2 - t1)

    return nearIndex, nearDistance


def get_near_station_info(config):

    t1 = time.time()

    # analizar y cambiar configuraciones
    path_stn_info = config['path_stn_info']
    file_stn_nearinfo = f'{path_stn_info}/all_stn_nearinfo.nc'
    config['file_stn_nearinfo'] = file_stn_nearinfo

    # información de entrada/salida para esta función
    file_allstn = config['file_allstn']
    infile_grid_domain = config['infile_grid_domain']
    file_stn_nearinfo = config['file_stn_nearinfo']

    try_radius = config['try_radius']
    initial_distance = config['initial_distance']
    nearstn_min = config['nearstn_min']
    nearstn_max = config['nearstn_max']
    # target_vars = ['prcp', 'tmean', 'trange']
    target_vars = config['target_vars']

    num_processes = config['num_processes']

    stn_lat_name = config['stn_lat_name']
    stn_lon_name = config['stn_lon_name']

    grid_lat_name = config['grid_lat_name']
    grid_lon_name = config['grid_lon_name']
    grid_mask_name = config['grid_mask_name']

    if 'overwrite_stninfo' in config:
        overwrite_stninfo = config['overwrite_stninfo']
    else:
        overwrite_stninfo = False

    print('#' * 50)
    print('Obtener información de la estación cercana')
    print('#' * 50)
    print('archivo de entrada file_allstn:', file_allstn)
    print('archivo de entrada infile_grid_domain:', infile_grid_domain)
    print('archivo de salida file_stn_nearinfo:', file_stn_nearinfo)
    print('nearstn_min:', nearstn_min)
    print('nearstn_max:', nearstn_max)
    print('try_radius:', try_radius)
    print('initial_distance:', initial_distance)
    print('Número de procesos:', num_processes)

    if os.path.isfile(file_stn_nearinfo):
        print('¡Nota! El archivo de información de la estación cercana ya existe')
        if overwrite_stninfo == True:
            print('overwrite_stninfo es True. Continuando.')
        else:
            print('overwrite_stninfo es False. Omitiendo la búsqueda de estaciones cercanas.')
            return config

    ########################################################################################################################
    # leer información de la estación

    ds_stn = xr.load_dataset(file_allstn)
    lat_stn_raw = ds_stn[stn_lat_name].values
    lon_stn_raw = ds_stn[stn_lon_name].values

    # para una variable, algunas estaciones no tienen registros
    var_mean = []
    lat_stn_valid = []
    lon_stn_valid = []
    for v in target_vars:
        vm = ds_stn[v].mean(dim='time').values
        lat_v = lat_stn_raw.copy()
        lon_v = lon_stn_raw.copy()
        lat_v[np.isnan(vm)] = np.nan
        lon_v[np.isnan(vm)] = np.nan
        var_mean.append(vm)
        lat_stn_valid.append(lat_v)
        lon_stn_valid.append(lon_v)

    ########################################################################################################################
    # leer información del dominio
    ds_domain = xr.load_dataset(infile_grid_domain)
    # ds_domain = ds_domain.rename({'x':'lon', 'y':'lat'})
    lat_grid = ds_domain[grid_lat_name].values
    lon_grid = ds_domain[grid_lon_name].values
    mask_grid = ds_domain[grid_mask_name].values
    ds_domain.coords['y'] = np.arange(lat_grid.shape[0])
    ds_domain.coords['x'] = np.arange(lat_grid.shape[1])

    # inicializar salida
    ds_nearinfo = ds_domain.copy()
    ds_nearinfo.coords['near'] = np.arange(nearstn_max)
    ds_nearinfo.coords['stn'] = ds_stn.stn.values
    for v in ds_stn.data_vars:
        if not 'time' in ds_stn[v].dims:
            ds_nearinfo['stn_'+v] = ds_stn[v]

    ########################################################################################################################
    # generar información cercana

    for i in range(len(target_vars)):

        lat_stn = lat_stn_valid[i]
        lon_stn = lon_stn_valid[i]
        vari = target_vars[i]

        print('Procesando:', vari)

        ########################################################################################################################
        # encontrar estaciones cercanas para estaciones/cuadrículas
        t11=time.time()
        nearIndex_Grid, nearDistance_Grid = find_nearstn_for_Grids(lat_stn, lon_stn, lat_grid, lon_grid, mask_grid, try_radius, nearstn_min, nearstn_max, num_processes, initial_distance)
        nearIndex_InStn, nearDistance_InStn = find_nearstn_for_InStn(lat_stn, lon_stn, try_radius, nearstn_min, nearstn_max, initial_distance)
        t22 = time.time()
        print('Costo de tiempo (segundos) para obtener el índice y la distancia de la estación cercana:', t22-t11)

        ########################################################################################################################
        # agregar información cercana al archivo de salida
        ds_nearinfo['nearIndex_Grid_' + vari] = xr.DataArray(nearIndex_Grid, dims=('y', 'x', 'near'))
        ds_nearinfo['nearDistance_Grid_' + vari] = xr.DataArray(nearDistance_Grid, dims=('y', 'x', 'near'))
        ds_nearinfo['nearIndex_InStn_' + vari] = xr.DataArray(nearIndex_InStn, dims=('stn', 'near'))
        ds_nearinfo['nearDistance_InStn_' + vari] = xr.DataArray(nearDistance_InStn, dims=('stn', 'near'))

    # guardar en archivos de salida
    encoding = {}
    for var in ds_nearinfo.data_vars:
        encoding[var] = {'zlib': True, 'complevel': 4}
    ds_nearinfo.to_netcdf(file_stn_nearinfo, encoding=encoding)

    t2 = time.time()
    print('Costo de tiempo (segundos):', t2 - t1)
    print('¡Búsqueda de estación cercana exitosa!\n\n')

    return config
