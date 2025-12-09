# Sistema de Imputación de Datos Meteorológicos Multi-Variable

Este proyecto es una herramienta avanzada para la imputación de datos meteorológicos faltantes (precipitación, temperatura máxima, temperatura mínima y caudales) utilizando técnicas de regresión y aprendizaje automático (Random Forest, LWR, etc.) con predictores espaciales y temporales.

**Desarrollado por:** Ing. Kevin David Condori Quispe  
**Contacto:** e93163@uajms.edu.bo, ingkevindavid98@gmail.com

---

## Funcionalidades

El sistema ofrece un flujo de trabajo completo para el procesamiento de datos hidrometeorológicos:

1.  **Imputación Multi-Variable**: Procesa secuencialmente múltiples variables (`prcp`, `tmax`, `tmin`, `flow`).
2.  **Métodos de Regresión**: Soporta múltiples algoritmos:
    *   `RF` (Random Forest) con predictores espaciales (elevación, pendientes).
    *   `LWR` (Locally Weighted Regression).
    *   `MLR` (Multiple Linear Regression).
    *   `SLR` (Simple Linear Regression).
3.  **Predictores Espaciales**: Integra datos de un Modelo Digital de Elevación (DEM) para incorporar elevación, pendiente norte y pendiente este como covariables.
4.  **Imputación de Caudales**: Lógica especializada que utiliza tanto precipitación (imputada previamente) como caudales de estaciones vecinas como predictores.
5.  **Transformación Box-Cox**: Normalización automática de datos para mejorar la linealidad y el rendimiento de los modelos.
6.  **Validación Cruzada Rigurosa**: Evaluación mediante técnicas de *hold-out* y validación cruzada para estimar métricas de desempeño (NSE, KGE, RMSE, CC, BIAS) de manera realista.
7.  **Pruebas de Homogeneidad**: Análisis automático de homogeneidad (Buishand U-test, SNHT) para detectar puntos de quiebre en las series temporales resultantes.
8.  **Visualización Avanzada**: Generación de paneles comparativos, mapas de interpolación y gráficos de series temporales.

---

## Instalación

El proyecto requiere Python 3.11 o superior. Se recomienda utilizar un entorno virtual.

### Dependencias Principales

*   `numpy`, `pandas`, `xarray` (Procesamiento de datos)
*   `scikit-learn` (Algoritmos de ML e imputación)
*   `matplotlib` (Visualización)
*   `joblib` (Procesamiento paralelo)
*   `rasterio` (Manejo de DEM/GeoTIFF)
*   `pyhomogeneity` (Pruebas de homogeneidad)

### Instalación vía `pip` (Python estándar)

```bash
# Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate  # En Linux/Mac
# venv\Scripts\activate   # En Windows

# Instalar dependencias
pip install numpy pandas xarray scikit-learn matplotlib joblib rasterio pyhomogeneity
```

### Instalación vía `conda`

```bash
# Crear ambiente conda
conda create -n meteo_imputation python=3.11
conda activate meteo_imputation

# Instalar librerías
conda install -c conda-forge numpy pandas xarray scikit-learn matplotlib joblib rasterio
pip install pyhomogeneity  # pyhomogeneity suele estar solo en PyPI
```

---

## Uso del Proyecto

El flujo de trabajo se controla principalmente desde el notebook `src/main.ipynb`, que orquesta las funciones definidas en los módulos del directorio `src/`.

### Estructura de Archivos

*   `src/main.ipynb`: Notebook principal (punto de entrada).
*   `src/imputation_pipeline.py`: Lógica de orquestación central.
*   `src/regression.py`: Implementación de algoritmos de imputación.
*   `src/homogeneity.py`: Módulo de pruebas de homogeneidad.
*   `src/visualization.py`: Generación de gráficos.
*   `src/data_processing.py`, `src/dem_processing.py`: Manejo de datos y DEM.

### Pasos para Ejecutar

1.  **Configurar Datos**: Asegúrese de que sus archivos CSV (`01_prcp.csv`, `02_tmax.csv`, etc.) y el DEM (`DEM.tif`) estén en las rutas correctas (por defecto `../`).
2.  **Abrir Notebook**: Inicie Jupyter Notebook o JupyterLab y abra `src/main.ipynb`.
3.  **Ajustar Configuración**: En la Celda 1 ("Configuración"), puede modificar:
    *   Métodos a utilizar (`methods=['RF', 'LWR']`).
    *   Rutas de archivos.
    *   Parámetros de los modelos.
4.  **Ejecutar**: Corra las celdas secuencialmente para realizar la imputación, evaluación, visualización y exportación.

### Salidas

El sistema generará en el directorio `output/`:
*   `DATOS_IMPUTADOS_*.csv`: Series temporales completas.
*   `COMPARACION_METODOS_MULTI.csv`: Métricas de validación cruzada.
*   `ESTADISTICAS_BASICAS.csv`: Resumen estadístico.
*   `imputation_results_*.nc`: Archivos NetCDF con resultados completos.

---

## Caso de Uso Actual

Este producto fue diseñado específicamente para solucionar problemas de datos faltantes en balances hídricos de cuenca. Actualmente se utiliza para completar series históricas diarias de precipitación, temperatura y caudal, permitiendo análisis hidrológicos más robustos y continuos, integrando información topográfica para mejorar la precisión en zonas de montaña o con topografía compleja.

---
© 2025 Kevin David Condori Quispe. Todos los derechos reservados.
