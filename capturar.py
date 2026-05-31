import subprocess, sys, os, datetime, time

CARPETA_SALIDA  = f'frames_cinquez_{datetime.date.today().strftime("%Y%m%d")}'
ANCHO, ALTO     = 1280, 720
INTERVALO_SEG   = 5
DURACION_HORAS  = 12
URL_STREAM      = 'https://www.youtube.com/watch?v=EsWV4O2ishg'
ARCHIVO_COOKIES = 'cookies_youtube.txt'
PYTHON = r'C:\Users\ACER\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe'

os.makedirs(CARPETA_SALIDA, exist_ok=True)

def obtener_url():
    r = subprocess.run(
        [PYTHON, '-m', 'yt_dlp', '--cookies', ARCHIVO_COOKIES,
         '-g', '-f', 'best[height<=720]', URL_STREAM],
        capture_output=True, text=True, timeout=60
    )
    return r.stdout.strip().split('\n')[0]

url_directa       = obtener_url()
ultima_renovacion = time.time()
n_capturas        = int(DURACION_HORAS * 3600 / INTERVALO_SEG)
print(f'✅ URL obtenida — guardando en {CARPETA_SALIDA}/')
print(f'Total frames a capturar: {n_capturas}')

guardados = 0
fallidos  = 0

for i in range(n_capturas):
    if time.time() - ultima_renovacion > 7200:
        print('Renovando URL...')
        url_directa       = obtener_url()
        ultima_renovacion = time.time()
        print('✅ URL renovada')

    ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    ruta = os.path.join(CARPETA_SALIDA, f'{ts}.jpg')

    r = subprocess.run(
        ['ffmpeg', '-i', url_directa,
         '-frames:v', '1', '-vf', f'scale={ANCHO}:{ALTO}',
         '-y', ruta],
        capture_output=True, timeout=15
    )

    if os.path.exists(ruta) and os.path.getsize(ruta) > 1000:
        guardados += 1
        print(f'  [{i+1}/{n_capturas}] {ts} → ok ({guardados} guardados)')
    else:
        fallidos += 1
        print(f'  [{i+1}/{n_capturas}] {ts} → fallido ({fallidos} fallidos)')

    time.sleep(max(0, INTERVALO_SEG - 5))

print(f'\n✅ Listo — {guardados} frames en {CARPETA_SALIDA}/')