# Plaza Inteligente — NODOSPLAZA (v6)

Sistema de visión por computadora para monitoreo de parques y caniles públicos.
Procesa imágenes de una cámara fija y produce métricas de uso del espacio,
confort térmico y estado del entorno, pensadas para decisiones municipales
(dónde plantar sombra, cómo ajustar mobiliario, cuándo regar).

Piloto en el Canil 3 Poniente, Maipú. Dataset de prueba: livestream del parque
Cinquez (Florida, EEUU), similar en escala al Canil.

## Qué hace el pipeline

Cada imagen ("snapshot") pasa por un pipeline de fases:

- **Detección (YOLOv8):** cuenta personas y mascotas, con umbral de confianza por
  clase y zonas de exclusión para falsos positivos de objetos fijos.
- **Homografía:** proyecta píxeles de la cámara a metros reales y a una vista
  satelital. Error mediano sub-métrico (~0.68 m). Es la base de todas las
  métricas espaciales.
- **Sombra (BDRAR) y condición de luz:** detecta sombra sobre el suelo y sobre la
  cabeza de cada persona; clasifica diurno/nublado/baja-luz.
- **Vegetación (ExG):** índice de verdor del pasto sobre zonas marcadas, con
  comparación entre zona de uso y zona de control.
- **Uso de mobiliario:** determina qué bancas están ocupadas, por distancia real.
- **UTCI (confort térmico):** índice térmico universal, con un modelo de
  temperatura radiante media entrenado sobre datos ERA5 (RMSE UTCI ~1.55°C).
- **Densidad y aglomeración:** usuarios por m² sobre el área realmente observada.
- **Mapa de islas de calor UTCI:** UTCI por celda de 2×2 m ajustado por sombra
  local — mapa de contraste térmico del parque.
- **Batch:** agrega métricas sobre muchos snapshots (ocupación, correlaciones,
  grilla de ocupación espacial, patrones temporales), con "guards" que desactivan
  cada métrica hasta tener datos suficientes.

## Archivos

| Archivo | Qué hace |
|---|---|
| `nodosplaza_v6.ipynb` | Notebook principal: todo el pipeline fase por fase, con visualizaciones y tests de invariantes. Corrido de punta a punta como evidencia. |
| `entrenar_utci.py` | Entrena el modelo de temperatura radiante media (mrt) desde datos ERA5. Produce los `.pkl` del modelo. |
| `generar_viento_climatologico.py` | Genera la tabla de viento climatológico (respaldo del anemómetro no comprado). |
| `barrido_yolo_calibracion.py` | Corre YOLO crudo sobre un dataset completo para calibrar umbrales de detección (labeling). |
| `capturar.py` | Captura frames de un livestream de YouTube (cámara del parque). |
| `modelo_era5_*.pkl` | Modelo UTCI-ERA5 entrenado (mrt, scaler, metadata, viento climatológico). Incluidos para correr sin reentrenar. |
| `mapa_conimg.png` | Imagen satelital del parque, usada para las visualizaciones cenitales. |

## Modelos y pesos

Los `.pkl` del modelo UTCI-ERA5 están en el repo para que el pipeline corra sin
reentrenar.

Los pesos de YOLO (`yolov8s.pt`) se descargan solos la primera vez con
`ultralytics`. Los pesos de BDRAR (detección de sombra) se obtienen del repo
original de BDRAR (Shadow-Detection-and-Removal) y no se versionan aquí.

## Cómo correr

Requiere Python 3.12. Dependencias principales:

```
pip install ultralytics torch torchvision opencv-python numpy pandas scikit-learn scipy pythermalcomfort matplotlib pillow joblib
```

Luego abrir `nodosplaza_v6.ipynb` y ejecutar de arriba a abajo. El notebook espera
una carpeta de frames de prueba (no incluida por tamaño) y el modelo ERA5 (incluido).

**Nota sobre `mapa_conimg.png`:** en el código la ruta está como `../mapa_conimg.png`
(un nivel arriba), porque el notebook se desarrolló dentro de una subcarpeta. Si
clonas el repo con todo junto, ajusta esa ruta a `mapa_conimg.png` en la celda de
configuración (variable `RUTA_SATELITAL`).

## Estado

Pipeline funcional y probado sobre el dataset de Cinquez. Pendientes principales:
calibración de umbrales de detección y sombra con etiquetado manual (ground truth);
integración al servidor de producción; recalibración completa el día de instalación
de la cámara en el Canil. Varias métricas están implementadas pero "dormidas" hasta
acumular datos reales suficientes (correlaciones, patrones temporales, mapa de calor
agregado, índice de vegetación estacional).

## Notas

- Todo valor que depende del sitio o de la cámara está marcado en el código con
  etiquetas (`[RECALIBRAR-CAM]`, `[RECALIBRAR-DATOS]`, `[PROD]`, `[AJUSTABLE]`) y
  centralizado en la celda de configuración, para facilitar la recalibración al
  cambiar de parque.
- Los umbrales de detección (confianza YOLO, binarización de sombra) son
  provisionales y se recalibran por dataset — no son valores definitivos.
