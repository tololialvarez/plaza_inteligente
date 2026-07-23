#!/usr/bin/env python3
# barrido_yolo_calibracion.py
# ─────────────────────────────────────────────────────────────────────────────
# Corre YOLO CRUDO (umbral bajo, SIN filtro por clase) sobre todas las imágenes de
# un dataset y guarda cada detección con su confianza real. Es el insumo para
# calibrar los umbrales de #4 (ver doc 05, metodología de etiquetado).
#
# CLAVE: usa conf bajo (0.10) a propósito. El pipeline normal filtra en 0.38 y
# descarta lo de abajo — pero para CALIBRAR el umbral hay que VER las detecciones
# cercanas al umbral (0.25-0.38), que son las informativas. Si filtráramos aquí,
# estaríamos calibrando con datos ya filtrados por el mismo umbral (círculo vicioso).
#
# SAFEGUARDS: barra de progreso con ETA, guardado incremental cada N imágenes
# (retoma donde quedó si se corta), rutas parametrizadas por CLI para reusar el
# script con otros datasets (Cinquez centro/oeste/este, Canil) sin editarlo.
#
# USO:
#   python barrido_yolo_calibracion.py --carpeta ../frames_cinquez_20260530/
#   (opcional: --salida detecciones_crudas.csv --imgsz 1920 --conf-min 0.10 --cada 100)
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import csv
import glob
import os
import time
from ultralytics import YOLO

# clases COCO que interesan (mismas que el notebook: persona, banca, gato, perro, silla)
CLASES_INTERES = {0: 'persona', 13: 'banca', 15: 'gato', 16: 'perro', 56: 'silla'}


def parse_args():
    p = argparse.ArgumentParser(description='Barrido YOLO crudo para calibración de umbrales.')
    p.add_argument('--carpeta', required=True,
                   help='Carpeta con las imágenes .jpg del dataset (parametrizada, reutilizable).')
    p.add_argument('--salida', default='detecciones_crudas.csv',
                   help='CSV de salida con una fila por detección.')
    p.add_argument('--pesos', default='yolov8s.pt', help='Pesos YOLO (mismos que el notebook).')
    p.add_argument('--imgsz', type=int, default=1920, help='Resolución de inferencia (notebook usa 1920).')
    p.add_argument('--conf-min', type=float, default=0.10,
                   help='Umbral MÍNIMO bajo a propósito — para ver detecciones cercanas al umbral.')
    p.add_argument('--iou', type=float, default=0.5, help='IoU del NMS (igual que el notebook).')
    p.add_argument('--cada', type=int, default=100, help='Guarda progreso cada N imágenes.')
    return p.parse_args()


def imagenes_ya_procesadas(ruta_csv):
    """Retoma: lee del CSV qué imágenes ya se barrieron (por nombre de archivo)."""
    if not os.path.exists(ruta_csv):
        return set()
    procesadas = set()
    with open(ruta_csv, newline='', encoding='utf-8') as f:
        for fila in csv.DictReader(f):
            procesadas.add(fila['archivo'])
    return procesadas


def formato_tiempo(seg):
    seg = int(seg)
    h, r = divmod(seg, 3600)
    m, s = divmod(r, 60)
    if h: return f'{h}h{m:02d}m'
    if m: return f'{m}m{s:02d}s'
    return f'{s}s'


def main():
    args = parse_args()

    todos = sorted(glob.glob(os.path.join(args.carpeta, '*.jpg')))
    if not todos:
        print(f'❌ No se encontraron .jpg en {args.carpeta}')
        return
    print(f'📁 {len(todos)} imágenes en {args.carpeta}')

    ya = imagenes_ya_procesadas(args.salida)
    pendientes = [r for r in todos if os.path.basename(r) not in ya]
    if ya:
        print(f'⏩ Retomando: {len(ya)} ya procesadas, {len(pendientes)} pendientes')

    if not pendientes:
        print('✅ Nada pendiente — el barrido ya está completo.')
        return

    print(f'🔧 YOLO {args.pesos} · imgsz={args.imgsz} · conf-min={args.conf_min} '
          f'(bajo A PROPÓSITO, sin filtro por clase — para calibración)')
    modelo = YOLO(args.pesos)

    # abrir CSV en modo append (crea headers si es nuevo)
    nuevo = not os.path.exists(args.salida)
    campos = ['archivo', 'clase_id', 'clase', 'confianza', 'x1', 'y1', 'x2', 'y2', 'pie_x', 'pie_y']
    f_out = open(args.salida, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(f_out, fieldnames=campos, quoting=csv.QUOTE_MINIMAL)
    if nuevo:
        writer.writeheader()

    t0 = time.time()
    n_det_total = 0
    try:
        for i, ruta in enumerate(pendientes, 1):
            nombre = os.path.basename(ruta)
            # inferencia CRUDA: conf bajo, sin filtro por clase, sin zonas de exclusión
            # (todo eso se aplica DESPUÉS en el análisis, probando distintos cortes)
            result = modelo(ruta, conf=args.conf_min, iou=args.iou,
                            imgsz=args.imgsz, agnostic_nms=False, verbose=False)[0]

            n_det = 0
            for caja in result.boxes:
                cid = int(caja.cls)
                if cid not in CLASES_INTERES:
                    continue
                x1, y1, x2, y2 = map(int, caja.xyxy[0].tolist())
                writer.writerow({
                    'archivo': nombre, 'clase_id': cid, 'clase': CLASES_INTERES[cid],
                    'confianza': round(float(caja.conf), 4),
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'pie_x': (x1 + x2) // 2, 'pie_y': y2,
                })
                n_det += 1
            n_det_total += n_det

            # guardado incremental + barra de progreso con ETA
            if i % args.cada == 0 or i == len(pendientes):
                f_out.flush()
                os.fsync(f_out.fileno())  # asegura que quedó en disco (retoma seguro)
                transcurrido = time.time() - t0
                por_img = transcurrido / i
                faltan = len(pendientes) - i
                eta = por_img * faltan
                pct = 100 * i / len(pendientes)
                barra = '█' * int(pct // 4) + '░' * (25 - int(pct // 4))
                print(f'\r[{barra}] {pct:5.1f}%  {i}/{len(pendientes)}  '
                      f'{por_img:.2f}s/img  transcurrido {formato_tiempo(transcurrido)}  '
                      f'ETA {formato_tiempo(eta)}  ·  {n_det_total} detecciones',
                      end='', flush=True)
    except KeyboardInterrupt:
        print('\n⏸️  Interrumpido — progreso guardado. Vuelve a correr el script para retomar.')
    finally:
        f_out.flush()
        f_out.close()

    print(f'\n✅ Listo. {n_det_total} detecciones guardadas en {args.salida}')
    print('   Siguiente paso: el script de muestreo estratificado lee este CSV.')


if __name__ == '__main__':
    main()