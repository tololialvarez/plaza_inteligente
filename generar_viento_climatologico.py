"""
v6/generar_viento_climatologico.py

Genera la tabla de viento climatológico (promedio histórico por mes y hora
local) desde dataset_completo.csv, para usar como respaldo del anemómetro que
no está comprado (ver documento 01/04).

Por qué no "viento de ERA5 en tiempo real": ERA5 tiene ~5 días de latencia —
no existe el dato de "ahora" cuando llega un snapshot real. La climatología
(promedio histórico por mes/hora) es la alternativa que no depende de ninguna
llamada en vivo, calculada una sola vez, offline, desde el mismo histórico que
ya se usó para entrenar el modelo de mrt.

Se corre UNA VEZ (o cuando se actualice el dataset con más años). Genera:
  modelo_era5_viento_climatologico.pkl

    python generar_viento_climatologico.py
"""

import joblib
import numpy as np
import pandas as pd

RUTA_DATASET = 'dataset_completo.csv'
UTC_OFFSET = -4


def main():
    df = pd.read_csv(RUTA_DATASET, parse_dates=['timestamp'])

    hora_local = (df['timestamp'].dt.hour + UTC_OFFSET) % 24
    mes = df['timestamp'].dt.month

    tabla = (pd.DataFrame({'mes': mes, 'hora': hora_local, 'viento': df['viento']})
             .groupby(['mes', 'hora'])['viento']
             .agg(['mean', 'std', 'count'])
             .reset_index())

    # dict (mes, hora) -> viento promedio, para lookup O(1) en producción
    climatologia = {(int(r.mes), int(r.hora)): float(r.mean)
                     for r in tabla.itertuples()}

    n_vacios = 12 * 24 - len(climatologia)
    if n_vacios:
        print(f'  ⚠️  {n_vacios} combinaciones mes/hora sin datos suficientes '
              f'— revisar antes de usar en producción')

    joblib.dump({
        'climatologia': climatologia,       # {(mes, hora_local): viento_m_s}
        'utc_offset': UTC_OFFSET,
        'n_filas_origen': len(df),
        'rango_fechas': f"{df['timestamp'].min()} a {df['timestamp'].max()}",
    }, 'modelo_era5_viento_climatologico.pkl')

    print(f'  Tabla generada desde {len(df)} filas '
          f'({df["timestamp"].min()} a {df["timestamp"].max()})')
    print(f'  {len(climatologia)}/288 combinaciones mes×hora cubiertas')
    print('  modelo_era5_viento_climatologico.pkl guardado')

    # muestra rápida para revisar que los números tengan sentido físico
    print('\n  Ejemplo — viento promedio a las 14:00 por mes:')
    for m in range(1, 13):
        v = climatologia.get((m, 14))
        print(f'    mes {m:2d}: {v:.2f} m/s' if v is not None else f'    mes {m:2d}: sin dato')


if __name__ == '__main__':
    main()