# Plaza Inteligente — NODOSPLAZA

Sistema de monitoreo IoT para parques caninos urbanos. Pipeline de visión computacional e IA para análisis de confort térmico, afluencia y uso del mobiliario.  
**Estado:** En desarrollo — fase de prototipo en notebook.

## Stack

- **Hardware:** Raspberry Pi 5 + Cámara RPi v3 + sensores AHT10, TEMT6000, GYML8511, anemómetro — *pendiente instalación física*
- **Transmisión:** HTTP POST 4G cada 5 minutos
- **Cloud:** AWS (Lambda + API Gateway + S3 + RDS PostgreSQL) — *cotizado, pendiente levantar*
- **Modelos IA:** YOLOv8s, BDRAR, KMeans — *implementados en notebook*
## Archivos

- `pipeline.ipynb`: notebook principal con visualizaciones y prints de debugging
- `pipeline_small.ipynb`: pipeline limpio sin outputs intermedios
- `capturar.py`: script para capturar frames desde livestreams de YouTube (datos de prueba)

---

## Pipeline implementado

| Fase | Descripción | Estado |
|------|-------------|--------|
| 0.5 | Insertar imágenes de cámara | funcional con frames de prueba |
| 1 | YOLO — detección personas, mascotas, mobiliario | ✅ |
| 2 | BDRAR — máscara de sombra + shade utilization | ✅ |
| 3 | Uso mobiliario por snapshot | ✅ bboxes manuales provisorias |
| 3.5 | Construcción snapshot_record + datos sensores | sensores simulados |
| 4 | UTCI con pythermalcomfort | ✅ |
| 5 | Aglomeración y densidad | ✅ |
| 5.5 | Guardar en BD (CSV local) | migrar a RDS |
| 6 | Métricas batch | ✅ |
| 7 | KMeans — mapa de islas de calor | ✅ |

---

## Métricas por snapshot

Crowd density · Conteo mascotas · Shade utilization · Uso mobiliario · UTCI

## Métricas por batch

Ocupación sostenida · Activity classification · Uso mobiliario histórico · Shade utilization histórico · Correlaciones · KMeans mapa de islas de calor

---

## Pendiente

- [ ] Instalación física cámara y sensores en Canil
- [ ] Definir bboxes manuales del mobiliario real con imagen de la cámara instalada
- [ ] Reemplazar datos de sensores simulados por datos reales del Nodo
- [ ] Levantar servidor AWS y conectar con RPi
- [ ] Migrar CSV local a PostgreSQL/RDS
- [ ] Convertir pipeline_small.ipynb en FastAPI para producción
- [ ] Fine-tuning YOLOv8s con imágenes reales del Canil
- [ ] Definir formato final del dashboard
- [ ] Calibrar parámetros ajustables con datos reales una vez instalada la cámara
