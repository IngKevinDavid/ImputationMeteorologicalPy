import numpy as np
import sys

def distanceweight(dist, maxdist = 100, exp = 3):
    """
    Calcula pesos basados en distancia usando fórmula cúbica inversa.
    
    Formula: w = (1 - (d/d_max)^3)^3
    
    Args:
        dist (np.ndarray): Distancias en km
        maxdist (float): Distancia máxima (default 100 km, PyGMET standard)
        exp (int): Exponente (default 3)
        
    Returns:
        np.ndarray: Pesos normalizados [0, 1]
    """
    weight = (1 - (dist / maxdist) ** exp) ** exp
    weight[weight < 0] = 0
    return weight

def distanceweight_userdefined(dist, maxdist, weight_formula):
    weight = eval(weight_formula)
    weight[weight < 0] = 0
    return weight


def calculate_weights_from_distance(nearDistance, initial_distance=100, exp=3, formula=''):
    """
    Calcula pesos basados en distancias usando maxdist dinámico.
    
    CAMBIO CRÍTICO: maxdist se calcula dinámicamente por estación como
    max(initial_distance, max(distancias_válidas) + 1). Esto asegura que
    los pesos se escalen apropiadamente según la configuración local de vecinos.
    
    Args:
        nearDistance (np.ndarray): Distancias a vecinos (2D o 3D)
        initial_distance (float): Distancia inicial mínima (default 100 km, PyGMET standard)
        exp (int): Exponente para la fórmula de peso (default 3)
        formula (str): Fórmula personalizada opcional
        
    Returns:
        np.ndarray: Pesos normalizados con mismo shape que nearDistance
    """
    if nearDistance.ndim == 2:
        nstn = nearDistance.shape[0]
        nearWeight = np.nan * np.ones([nstn, nearDistance.shape[1]], dtype=np.float32)
        for i in range(nstn):
            disti = nearDistance[i, :]
            if disti[0] >= 0:
                # Filtrar solo distancias válidas (no-NaN, >= 0)
                disti_valid = disti[disti >= 0]
                # CRÍTICO: Calcular maxdist dinámicamente
                max_dist = np.max([initial_distance, np.max(disti_valid) + 1])
                
                if len(formula) == 0:
                    nearWeight[i, 0:len(disti_valid)] = distanceweight(disti_valid, max_dist, exp)
                else:
                    nearWeight[i, 0:len(disti_valid)] = distanceweight_userdefined(disti_valid, max_dist, formula)

    elif nearDistance.ndim == 3:
        nrow = nearDistance.shape[0]
        ncol = nearDistance.shape[1]
        nearWeight = np.nan * np.ones([nrow, ncol, nearDistance.shape[2]], dtype=np.float32)
        for i in range(nrow):
            for j in range(ncol):
                distij = nearDistance[i, j, :]
                if distij[0] >= 0:
                    # Filtrar solo distancias válidas
                    distij_valid = distij[distij >= 0]
                    # CRÍTICO: Calcular maxdist dinámicamente
                    max_dist = np.max([initial_distance, np.max(distij_valid) + 1])
                    
                    if len(formula) == 0:
                        nearWeight[i, j, 0:len(distij_valid)] = distanceweight(distij_valid, max_dist, exp)
                    else:
                        nearWeight[i, j, 0:len(distij_valid)] = distanceweight_userdefined(distij_valid, max_dist, formula)

    else:
        sys.exit('¡Error! nearDistance debe tener ndim 2 o 3.')

    return nearWeight


