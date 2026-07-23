"""
v6/entrenar_utci.py

Entrena el modelo de temperatura radiante media (mrt) con ERA5, para reemplazar
la fórmula inventada de calcular_tr() (#18).

    python entrenar_utci.py

Lee de:    data_original_utci/   (16 zips ERA5 + 1 csv UTCI/mrt)
           o dataset_completo.csv si ya existe (se reutiliza, no se reconstruye)
Genera en: v6/                   (dataset_completo.csv, modelo_*.pkl)

DISEÑO
------
Se predice mrt, no UTCI: el UTCI final lo calcula el polinomio oficial de
pythermalcomfort. Así el modelo solo carga con la variable que no se puede medir.

Features: solo las que el Canil puede medir (sensores del Nodo B) o calcular
(timestamp + ubicación). No se usa radiación de ERA5 porque no hay piranómetro,
ni lags porque los snapshots son cada 5 min y ERA5 es horario (desalineamiento,
y la cadencia real es variable — "hace una hora" no significa nada estable en
producción). Se probaron y se descartaron (documento 05, parte 6).

Dos fases:
  A) split temporal 2022-24 / 2025 → mide el error honesto, se reporta.
     Se compara contra un set amplio de familias (lineal, MLP, GBM, RF,
     XGBoost, SVR) para no dejar la elección de arquitectura sin evidencia.
  B) reentrenar con el 100% de los datos → ese es el .pkl que usa core.py

CAMBIOS respecto de la versión anterior:
  - MLPRegressor ya no recibe hidden_layer_sizes posicional. sklearn >=1.8
    agregó el parámetro `loss` ANTES de hidden_layer_sizes en el __init__,
    así que un positional (64,64,32) se cuela como loss y explota. Ahora es
    keyword explícito: rompe menos con el tiempo.
  - Se agregaron XGBoost y SVR-RBF a la comparación de la fase A.
  - metadata.pkl usa una sola convención de nombres (ver ARTEFACTOS abajo)
    para que core.py no tenga que adivinar claves de corridas viejas.

ARTEFACTOS QUE GENERA (y solo estos — no dejar .pkl huérfanos de versiones
anteriores dando vueltas en la carpeta):
  - modelo_era5_mrt.pkl       el estimador entrenado con el 100% de los datos
  - modelo_era5_scaler.pkl    el StandardScaler ajustado sobre el 100%
  - modelo_era5_metadata.pkl  dict con 'features', 'familia', métricas de fase A
"""

import os, glob, zipfile, joblib, warnings
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
from pythermalcomfort.models import utci as utci_fn

try:
    from xgboost import XGBRegressor
    TIENE_XGBOOST = True
except ImportError:
    TIENE_XGBOOST = False

warnings.filterwarnings('ignore')

DIR_ORIGEN   = 'data_original_utci'
DIR_EXTRAIDO = 'era5_extraido'
RUTA_DATASET = 'dataset_completo.csv'

LAT, LON   = -33.51, -70.77
UTC_OFFSET = -4
ANIO_TEST  = 2025

# Fracción difusa de la radiación global bajo cielo despejado. Se usa en
# producción para simular sombra: bajo un quincho se bloquea el haz directo
# pero la difusa sigue llegando. Valor típico de literatura: 10-20%.
FRACCION_DIFUSA = 0.15

# Solo lo que el Canil tiene sin agregar hardware ni recalibrar nada.
# SIN lags: se probaron (t2m_lag1, rh_lag1, t2m_delta) y se descartaron.
FEATURES = ['t2m_c', 'rh', 'viento',                  # AHT10 + anemómetro
            'elev_solar', 'rad_cielo_despejado',      # física, desde timestamp+ubicación
            'hora_sin', 'hora_cos', 'dia_sin', 'dia_cos']

# Solo para medir cuánto se pierde por no tener piranómetro (argumento para Josué).
FEATURES_CON_RADIACION = FEATURES + ['ssrd_w', 'strd_w']


# ── datos ────────────────────────────────────────────────────────────────────
def extraer_zips():
    """Los .nc del CDS son ZIPs con dos NetCDF adentro (instant y accum).
    Cada trimestre va a su propia subcarpeta para que no se pisen."""
    os.makedirs(DIR_EXTRAIDO, exist_ok=True)
    zips = sorted(glob.glob(f'{DIR_ORIGEN}/era5_met_*.nc'))
    if not zips:
        raise FileNotFoundError(f'No hay era5_met_*.nc en {DIR_ORIGEN}/')
    for ruta in zips:
        destino = os.path.join(DIR_EXTRAIDO,
                               os.path.basename(ruta).replace('.nc', ''))
        if os.path.isdir(destino) and glob.glob(f'{destino}/*.nc'):
            continue
        os.makedirs(destino, exist_ok=True)
        with zipfile.ZipFile(ruta) as z:
            z.extractall(destino)
    return sorted(glob.glob(f'{DIR_EXTRAIDO}/era5_met_*'))


def construir_dataset(carpetas):
    """Une los 16 trimestres meteorológicos con el CSV de UTCI/mrt, por timestamp."""
    dfs = []
    for c in carpetas:
        inst = xr.open_dataset(f'{c}/data_stream-oper_stepType-instant.nc')
        acum = xr.open_dataset(f'{c}/data_stream-oper_stepType-accum.nc')
        dfs.append(inst[['t2m','d2m','u10','v10']].to_dataframe().reset_index()
                   .merge(acum[['ssrd','strd']].to_dataframe().reset_index(),
                          on='valid_time'))
        inst.close(); acum.close()

    met = (pd.concat(dfs, ignore_index=True)
             .rename(columns={'valid_time': 'timestamp'})
             .sort_values('timestamp').reset_index(drop=True))

    met['t2m_c']  = met['t2m'] - 273.15          # ERA5 entrega Kelvin
    met['d2m_c']  = met['d2m'] - 273.15
    met['viento'] = np.sqrt(met['u10']**2 + met['v10']**2)
    # humedad relativa desde punto de rocío (Magnus)
    met['rh'] = 100 * (np.exp((17.625*met['d2m_c'])/(243.04+met['d2m_c'])) /
                       np.exp((17.625*met['t2m_c'])/(243.04+met['t2m_c'])))
    met['ssrd_w'] = met['ssrd'] / 3600           # J/m² acumulados en 1 h → W/m²
    met['strd_w'] = met['strd'] / 3600

    utci = pd.read_csv(glob.glob(f'{DIR_ORIGEN}/*.csv')[0])
    utci['timestamp'] = pd.to_datetime(utci['valid_time'])
    utci['mrt_c']  = utci['mrt']  - 273.15
    utci['utci_c'] = utci['utci'] - 273.15

    cols = ['timestamp','t2m_c','rh','viento','ssrd_w','strd_w']
    return met[cols].merge(utci[['timestamp','mrt_c','utci_c']],
                           on='timestamp', how='inner')


# ── features ─────────────────────────────────────────────────────────────────
def elevacion_solar(ts_utc, lat=LAT, lon=LON):
    """Elevación del sol en grados, desde tiempo solar (no depende de zona horaria).
    mrt depende fuerte de esto: la misma radiación pega distinto con sol bajo."""
    n = ts_utc.dt.dayofyear
    decl = np.radians(23.45 * np.sin(np.radians(360/365 * (284 + n))))
    hora_solar = ts_utc.dt.hour + ts_utc.dt.minute/60 + lon/15
    ang_h = np.radians(15 * (hora_solar - 12))
    lat_r = np.radians(lat)
    return np.degrees(np.arcsin(np.clip(
        np.sin(lat_r)*np.sin(decl) + np.cos(lat_r)*np.cos(decl)*np.cos(ang_h), -1, 1)))


def radiacion_cielo_despejado(elev_grados):
    """Radiación global horizontal con cielo despejado, W/m². Modelo de Haurwitz
    (1945): depende solo de la elevación solar.

    Es la pieza que reemplaza al piranómetro: se calcula con fecha y ubicación,
    sin sensor. Le da al modelo el techo teórico de radiación; lo que el modelo
    infiere de las demás variables es cuánto de eso llegó realmente."""
    s = np.sin(np.radians(np.clip(elev_grados, 0, 90)))
    s = np.where(s < 1e-3, 0.0, s)
    with np.errstate(divide='ignore', invalid='ignore'):
        ghi = 1098 * s * np.exp(-0.059 / np.where(s > 0, s, 1))
    return np.where(s > 0, ghi, 0.0)


def agregar_features(df):
    """Geometría solar, radiación teórica y codificación cíclica de hora/día.
    Cíclica porque hora 23 y hora 0 son adyacentes, no opuestas."""
    df = df.copy()
    df['elev_solar']          = elevacion_solar(df['timestamp'])
    df['rad_cielo_despejado'] = radiacion_cielo_despejado(df['elev_solar'])
    hora = (df['timestamp'].dt.hour + UTC_OFFSET) % 24
    dia  = df['timestamp'].dt.dayofyear
    df['hora_local'] = hora
    df['hora_sin'] = np.sin(2*np.pi*hora/24)
    df['hora_cos'] = np.cos(2*np.pi*hora/24)
    df['dia_sin']  = np.sin(2*np.pi*dia/365)
    df['dia_cos']  = np.cos(2*np.pi*dia/365)
    return df


# ── evaluación ───────────────────────────────────────────────────────────────
def utci_desde_mrt(tdb, mrt, viento, rh):
    """UTCI oficial. El viento se clampea al rango válido de la librería (#20)."""
    v = np.clip(np.asarray(viento, float), 0.5, 17.0)
    out = []
    for t, m, vv, h in zip(np.asarray(tdb, float), np.asarray(mrt, float),
                           v, np.asarray(rh, float)):
        r = utci_fn(tdb=float(t), tr=float(m), v=float(vv), rh=float(h))
        out.append(r if np.isscalar(r) else getattr(r, 'utci', r))
    return np.array(out, float)


def evaluar(nombre, modelo, X_te_s, y_te, sub):
    """RMSE del mrt y del UTCI propagado (este último es el que importa en prod)."""
    pred = modelo.predict(X_te_s)
    utci_p = utci_desde_mrt(sub['t2m_c'], pred, sub['viento'], sub['rh'])
    r = {'nombre': nombre, 'modelo': modelo, 'pred': pred,
         'r2_mrt':  r2_score(y_te, pred),
         'rmse_mrt': float(np.sqrt(mean_squared_error(y_te, pred))),
         'r2_utci': r2_score(sub['utci_c'], utci_p),
         'rmse_utci': float(np.sqrt(mean_squared_error(sub['utci_c'], utci_p)))}
    print(f'  {nombre:<15} mrt: R²={r["r2_mrt"]:.4f} RMSE={r["rmse_mrt"]:5.2f}°C   '
          f'| UTCI: R²={r["r2_utci"]:.4f} RMSE={r["rmse_utci"]:5.2f}°C')
    return r


def candidatos_disponibles():
    """Set de familias a comparar en la fase A. Se amplió respecto de la
    versión original (solo lineal/MLP/GBM/RF) agregando XGBoost y SVR, para
    no elegir arquitectura con una comparación angosta. Ver documento 05
    parte 6: la búsqueda de hiperparámetros no mueve el resultado, así que
    esto es sobre todo para descartar con evidencia, no para optimizar más."""
    c = {
        'lineal':       LinearRegression(),
        'MLP 64-64-32': MLPRegressor(hidden_layer_sizes=(64,64,32), max_iter=600,
                                     random_state=42, early_stopping=True,
                                     n_iter_no_change=20),
        'MLP 128-64':   MLPRegressor(hidden_layer_sizes=(128,64), max_iter=600,
                                     random_state=42, early_stopping=True,
                                     n_iter_no_change=20),
        'GradBoost':    GradientBoostingRegressor(n_estimators=300, max_depth=5,
                                                  random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42,
                                              n_jobs=-1),
        'SVR-RBF':      SVR(kernel='rbf', C=10, epsilon=0.1),
    }
    if TIENE_XGBOOST:
        c['XGBoost'] = XGBRegressor(n_estimators=300, max_depth=5,
                                    learning_rate=0.05, random_state=42, n_jobs=-1)
    else:
        print('  (xgboost no instalado — se omite de la comparación; '
              'pip install xgboost para incluirlo)')
    return c


def comparar_modelos(df, features, etiqueta):
    """Fase A: split temporal para medir el error honesto y elegir la familia."""
    X, y = df[features], df['mrt_c']
    mask = df['timestamp'].dt.year < ANIO_TEST
    X_tr, X_te, y_tr, y_te, sub = X[mask], X[~mask], y[mask], y[~mask], df[~mask]
    sc = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)

    candidatos = candidatos_disponibles()
    print(f'\n  {etiqueta}: {len(features)} features, '
          f'train {len(X_tr)} (<{ANIO_TEST}) / test {len(X_te)} ({ANIO_TEST})')
    res = [evaluar(n, m.fit(X_tr_s, y_tr), X_te_s, y_te, sub)
           for n, m in candidatos.items()]
    res.sort(key=lambda r: r['rmse_utci'])
    mejor = res[0]
    mejor['sub'] = sub
    print(f'  → ranking: ' + ', '.join(f'{r["nombre"]}={r["rmse_utci"]:.2f}' for r in res))
    print(f'  → mejor: {mejor["nombre"]} (UTCI RMSE {mejor["rmse_utci"]:.2f}°C)')
    return mejor


def validacion_cruzada(plantilla, X, y, n_splits=4):
    """TimeSeriesSplit: entrena con el pasado, valida con el futuro, 4 veces.
    Más honesto que un único split, porque un año atípico puede sesgar."""
    rmses = []
    for i_tr, i_va in TimeSeriesSplit(n_splits=n_splits).split(X):
        sc = StandardScaler().fit(X.iloc[i_tr])
        m = type(plantilla)(**plantilla.get_params())
        m.fit(sc.transform(X.iloc[i_tr]), y.iloc[i_tr])
        rmses.append(np.sqrt(mean_squared_error(
            y.iloc[i_va], m.predict(sc.transform(X.iloc[i_va])))))
    return float(np.mean(rmses)), float(np.std(rmses))


def error_por_franja(mejor):
    """El RMSE global mezcla noche (fácil, sin radiación) con día (difícil),
    y el confort térmico importa de día."""
    sub = mejor['sub'].copy()
    sub['err'] = mejor['pred'] - sub['mrt_c']
    franjas = {'noche (21-6)':     (sub['hora_local'] >= 21) | (sub['hora_local'] <= 6),
               'mañana (7-11)':    sub['hora_local'].between(7, 11),
               'mediodía (12-16)': sub['hora_local'].between(12, 16),
               'tarde (17-20)':    sub['hora_local'].between(17, 20)}
    print('\n  error de mrt por franja:')
    for nombre, m in franjas.items():
        e = sub.loc[m, 'err']
        print(f'    {nombre:<18} RMSE={np.sqrt((e**2).mean()):5.2f}°C  '
              f'sesgo={e.mean():+5.2f}°C  (n={len(e)})')


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print('=== MODELO mrt DESDE ERA5 (#18) ===\n')

    print('1. Datos')
    if os.path.exists(RUTA_DATASET):
        df = pd.read_csv(RUTA_DATASET, parse_dates=['timestamp'])
        print(f'  {RUTA_DATASET} reutilizado ({len(df)} filas)')
    else:
        carpetas = extraer_zips()
        df = construir_dataset(carpetas)
        df.to_csv(RUTA_DATASET, index=False)
        print(f'  {RUTA_DATASET} creado ({len(df)} filas)')
    df = agregar_features(df)
    print(f'  {df["timestamp"].min()} → {df["timestamp"].max()}')

    print('\n2. FASE A — evaluación con split temporal, comparación amplia de arquitecturas')
    mejor = comparar_modelos(df, FEATURES, 'features del Canil')
    con_rad = comparar_modelos(df, FEATURES_CON_RADIACION,
                               'referencia: con piranómetro')
    brecha = mejor['rmse_utci'] - con_rad['rmse_utci']
    print(f'\n  costo de no tener piranómetro: +{brecha:.2f}°C en UTCI')
    print('  → bajo ±2°C: a lo más 1 categoría UTCI de error (Bröde et al.)'
          if mejor['rmse_utci'] < 2.0 else
          '  → sobre ±2°C: conviene evaluar un piranómetro con Josué')

    media, desv = validacion_cruzada(mejor['modelo'], df[FEATURES], df['mrt_c'])
    print(f'\n  validación cruzada temporal (4 splits): {media:.2f} ± {desv:.2f}°C')
    print(f'  (el split único de {ANIO_TEST} dio {mejor["rmse_mrt"]:.2f}°C)')
    error_por_franja(mejor)

    print('\n3. FASE B — reentrenar con el 100% de los datos')
    print(f'  familia elegida en fase A: {mejor["nombre"]}')
    X_all, y_all = df[FEATURES], df['mrt_c']
    scaler_final = StandardScaler().fit(X_all)
    modelo_final = type(mejor['modelo'])(**mejor['modelo'].get_params())
    modelo_final.fit(scaler_final.transform(X_all), y_all)
    print(f'  entrenado con {len(X_all)} filas '
          f'({df["timestamp"].dt.year.min()}-{df["timestamp"].dt.year.max()})')

    print('\n4. Guardando')
    joblib.dump(modelo_final, 'modelo_era5_mrt.pkl')
    joblib.dump(scaler_final, 'modelo_era5_scaler.pkl')
    joblib.dump({
        'features': FEATURES,
        'familia': mejor['nombre'],
        'fraccion_difusa': FRACCION_DIFUSA,
        'lat': LAT, 'lon': LON, 'utc_offset': UTC_OFFSET,
        # métricas de la fase A: son las que se reportan
        'rmse_mrt': mejor['rmse_mrt'],   'r2_mrt':  mejor['r2_mrt'],
        'rmse_utci': mejor['rmse_utci'], 'r2_utci': mejor['r2_utci'],
        'rmse_cv_media': media, 'rmse_cv_desv': desv,
        'brecha_piranometro': brecha,
        'anio_test': ANIO_TEST,
        'n_filas_entrenamiento_final': len(X_all),
    }, 'modelo_era5_metadata.pkl')
    print('  modelo_era5_mrt.pkl, modelo_era5_scaler.pkl, modelo_era5_metadata.pkl')
    print(f'\nPara reportar: UTCI RMSE {mejor["rmse_utci"]:.2f}°C, '
          f'R² {mejor["r2_utci"]:.4f}, familia {mejor["nombre"]}')


if __name__ == '__main__':
    main()