
import subprocess, sys, os, datetime, time

CARPETA_SALIDA  = f'frames_cinquez_west_{datetime.date.today().strftime("%Y%m%d")}'
ANCHO, ALTO     = 1280, 720
INTERVALO_SEG   = 5
DURACION_HORAS  = 12
URL_STREAM      = 'https://www.youtube.com/watch?v=i9LKXrE5QYA'  # ← cámara WEST (nuevo)
ARCHIVO_COOKIES = 'cookies_youtube.txt'
PYTHON = sys.executable

os.makedirs(CARPETA_SALIDA, exist_ok=True)

def obtener_url():
    r = subprocess.run(
        [PYTHON, '-m', 'yt_dlp', '--cookies', ARCHIVO_COOKIES,
         '-g', '-f', 'best[height<=720]', URL_STREAM],
        capture_output=True, text=True, timeout=60
    )
    if r.returncode != 0:
        print('⚠️  Error obteniendo URL:', r.stderr[:300])
        return None
    return r.stdout.strip().split('\n')[0]

url_directa = obtener_url()
if not url_directa:
    sys.exit('❌ No se pudo obtener la URL.')

ultima_renovacion = time.time()
t_fin  = time.time() + DURACION_HORAS * 3600   # corta por TIEMPO, no por conteo
guardados = 0
fallidos  = 0
i = 0

print(f'✅ Capturando cada {INTERVALO_SEG}s por {DURACION_HORAS}h en {CARPETA_SALIDA}/')

while time.time() < t_fin:
    ciclo_inicio = time.time()   # ← marca cuándo empieza este ciclo

    if time.time() - ultima_renovacion > 7200:
        nueva = obtener_url()
        if nueva:
            url_directa = nueva
            ultima_renovacion = time.time()
            print('✅ URL renovada')

    ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    ruta = os.path.join(CARPETA_SALIDA, f'{ts}.jpg')

    try:
        subprocess.run(
            ['ffmpeg', '-i', url_directa, '-frames:v', '1',
             '-vf', f'scale={ANCHO}:{ALTO}', '-y', ruta],
            capture_output=True, timeout=15
        )
    except Exception:
        fallidos += 1

    if os.path.exists(ruta) and os.path.getsize(ruta) > 1000:
        guardados += 1
        if guardados % 10 == 0:
            print(f'  {ts} → {guardados} guardados ({fallidos} fallidos)')
    else:
        fallidos += 1

    i += 1
    # esperar lo que FALTE para completar los 5 seg desde el inicio del ciclo
    transcurrido = time.time() - ciclo_inicio
    time.sleep(max(0, INTERVALO_SEG - transcurrido))

print(f'\n✅ Listo — {guardados} frames en {CARPETA_SALIDA}/  ({fallidos} fallidos)')