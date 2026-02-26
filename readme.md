# Drunk Detection - Face Extraction Pipeline

Pipeline para extraer rostros de videos de YouTube, diseñado para crear datasets de personas sobrias vs ebrias para entrenamiento de modelos de ML.

**Optimizado para bajo uso de disco**: Descarga, procesa y borra un video a la vez.

## Estructura del Proyecto

```
drunk_detection_project/
├── data/
│   ├── sober_videos.csv    # Links de videos (personas sobrias)
│   └── drunk_videos.csv    # Links de videos (personas ebrias)
├── output/
│   ├── sober/              # Rostros extraídos (sobrios)
│   └── drunk/              # Rostros extraídos (ebrios)
├── temp_videos/            # Videos temporales (se borran automáticamente)
├── src/
│   ├── main.py             # Script principal (configuración aquí)
│   ├── face_extractor.py   # Detección y extracción de rostros
│   └── video_downloader.py # Descarga de videos de YouTube
├── requirements.txt
└── README.md
```

## Instalación (Windows)

### 1. Crear entorno virtual

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Uso

### Paso 1: Crear archivos con URLs

Crear archivos CSV en la carpeta `data/`:

**sober_videos.csv:**
```csv
url
https://www.youtube.com/watch?v=VIDEO_ID_1
https://www.youtube.com/watch?v=VIDEO_ID_2
```

**drunk_videos.csv:**
```csv
url
https://www.youtube.com/watch?v=VIDEO_ID_3
https://www.youtube.com/watch?v=VIDEO_ID_4
```

También puedes usar archivos TXT con un link por línea.

### Paso 2: Configurar (opcional)

Abre `src/main.py` y modifica la sección de CONFIGURACION si es necesario:

```python
# Rutas a los archivos con links
SOBER_VIDEOS_FILE = "../data/sober_videos.csv"
DRUNK_VIDEOS_FILE = "../data/drunk_videos.csv"

# Carpetas de salida
OUTPUT_SOBER = "../output/sober"
OUTPUT_DRUNK = "../output/drunk"

# Configuración del detector
DETECTOR_TYPE = "mediapipe"  # "mediapipe" (rápido) o "mtcnn" (preciso)
FACE_OUTPUT_SIZE = 224       # Tamaño de imágenes de rostros
MIN_CONFIDENCE = 0.7         # Confianza mínima (0.0 a 1.0)
SAMPLE_INTERVAL = 0.5        # Segundos entre frames
MAX_FACES_PER_VIDEO = 100    # Máximo rostros por video
```

### Paso 3: Ejecutar

```bash
cd src
python main.py
```

El script procesará cada video secuencialmente:
1. Descarga el video
2. Extrae los rostros
3. Borra el video
4. Pasa al siguiente

## Flujo de Procesamiento

```
Video 1: Descargar → Extraer rostros → Borrar video
Video 2: Descargar → Extraer rostros → Borrar video
...
Video N: Descargar → Extraer rostros → Borrar video
```

**Uso de disco mínimo**: Solo un video a la vez en el disco.

## Características

- **Procesamiento secuencial**: No acumula videos, ahorra espacio
- **Dos detectores disponibles**:
  - MediaPipe: Rápido, recomendado para Windows
  - MTCNN: Más preciso con rostros pequeños
- **Sin argumentos de línea de comandos**: Configuración directa en el archivo
- **Soporte CSV y TXT**: Flexibilidad en formato de entrada

## Siguiente paso: Entrenamiento del modelo

Una vez extraídos los rostros, usa las imágenes en `output/sober` y `output/drunk` para entrenar tu clasificador.

## Notas

- Los videos de YouTube pueden no estar disponibles
- Para fines académicos, documenta las fuentes
- Si un video falla, el script continúa con el siguiente