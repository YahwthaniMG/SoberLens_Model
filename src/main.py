"""
Pipeline para extraer rostros de videos de personas sobrias y ebrias.
Procesa un video a la vez: descarga -> extrae rostros -> borra video.

Solo modifica las variables en la sección CONFIGURACION y ejecuta:
    python main.py
"""

import os
import time
from pathlib import Path
from typing import List

import pandas as pd

from video_downloader import download_video
from face_extractor import FaceExtractor


# =============================================================================
# CONFIGURACION - Modifica estas variables según tu proyecto
# =============================================================================

# Rutas a los archivos CSV o TXT con los links de videos
# Cada archivo debe tener una columna llamada "url" o ser un TXT con un link por línea
SOBER_VIDEOS_FILE = "../data/sober_videos.csv"
DRUNK_VIDEOS_FILE = "../data/drunk_videos.csv"

# Carpeta donde se guardarán los rostros extraídos
OUTPUT_SOBER = "../output/sober"
OUTPUT_DRUNK = "../output/drunk"

# Carpeta temporal para descargar videos (se borran después de procesar)
TEMP_VIDEO_DIR = "../temp_videos"

# Configuración del detector de rostros
DETECTOR_TYPE = "opencv"  # "opencv" (recomendado), "mediapipe" o "mtcnn"
FACE_OUTPUT_SIZE = 224  # Tamaño en pixeles de las imágenes de rostros
MIN_CONFIDENCE = 0.7  # Confianza mínima (0.0 a 1.0)
SAMPLE_INTERVAL = 0.5  # Segundos entre cada frame a analizar
MAX_FACES_PER_VIDEO = 100  # Máximo de rostros a extraer por video

# =============================================================================
# FIN DE CONFIGURACION
# =============================================================================


def read_urls_from_file(filepath: str) -> List[str]:
    """
    Lee URLs de un archivo CSV o TXT.

    Para CSV: debe tener columna 'url'
    Para TXT: un link por línea
    """
    filepath = Path(filepath)

    if not filepath.exists():
        print(f"Archivo no encontrado: {filepath}")
        return []

    urls = []

    if filepath.suffix.lower() == ".csv":
        df = pd.read_csv(filepath)
        if "url" in df.columns:
            urls = df["url"].dropna().tolist()
        elif "link" in df.columns:
            urls = df["link"].dropna().tolist()
        else:
            # Tomar la primera columna
            urls = df.iloc[:, 0].dropna().tolist()
    else:
        # Asumir TXT con un link por línea
        with open(filepath, "r", encoding="utf-8") as f:
            urls = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

    return urls


def delete_file(filepath: str) -> bool:
    """Elimina un archivo de forma segura."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
    except Exception as e:
        print(f"Error al borrar {filepath}: {e}")
    return False


def process_single_video(
    url: str,
    video_index: int,
    output_dir: str,
    temp_dir: str,
    extractor: FaceExtractor,
    category: str,
) -> dict:
    """
    Procesa un solo video: descarga, extrae rostros, borra.

    Returns:
        Diccionario con estadísticas del procesamiento
    """
    result = {"url": url, "success": False, "faces_extracted": 0, "error": None}

    video_id = f"{category}_{video_index:04d}"
    video_path = None

    try:
        # Paso 1: Descargar video
        print(f"\n{'='*60}")
        print(f"[{video_index + 1}] Procesando: {url[:60]}...")
        print(f"{'='*60}")

        print("Paso 1/3: Descargando video...")
        video_path = download_video(url, temp_dir, video_id)

        if not video_path or not os.path.exists(video_path):
            result["error"] = "Error en descarga"
            print("Error: No se pudo descargar el video")
            return result

        # Mostrar tamaño del video
        video_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"Video descargado: {video_size_mb:.1f} MB")

        # Paso 2: Extraer rostros
        print("Paso 2/3: Extrayendo rostros...")
        face_count = extractor.process_video(
            video_path=video_path,
            output_dir=output_dir,
            sample_interval=SAMPLE_INTERVAL,
            max_faces_per_video=MAX_FACES_PER_VIDEO,
        )

        result["faces_extracted"] = face_count
        result["success"] = True
        print(f"Rostros extraídos: {face_count}")

    except Exception as e:
        result["error"] = str(e)
        print(f"Error: {e}")

    finally:
        # Paso 3: Borrar video (siempre, incluso si hubo error)
        if video_path and os.path.exists(video_path):
            print("Paso 3/3: Eliminando video temporal...")
            if delete_file(video_path):
                print("Video eliminado correctamente")
            else:
                print("Advertencia: No se pudo eliminar el video")

    return result


def process_category(
    urls: List[str],
    output_dir: str,
    temp_dir: str,
    extractor: FaceExtractor,
    category: str,
) -> dict:
    """
    Procesa una categoría completa de videos (sobrio o ebrio).
    """
    stats = {
        "total_videos": len(urls),
        "successful": 0,
        "failed": 0,
        "total_faces": 0,
        "errors": [],
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    for idx, url in enumerate(urls):
        result = process_single_video(
            url=url,
            video_index=idx,
            output_dir=output_dir,
            temp_dir=temp_dir,
            extractor=extractor,
            category=category,
        )

        if result["success"]:
            stats["successful"] += 1
            stats["total_faces"] += result["faces_extracted"]
        else:
            stats["failed"] += 1
            stats["errors"].append({"url": url, "error": result["error"]})

        # Pequeña pausa para no saturar YouTube
        time.sleep(1)

    return stats


def print_summary(sober_stats: dict, drunk_stats: dict):
    """Imprime resumen final del procesamiento."""
    print("\n")
    print("=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)

    if sober_stats:
        print("\nVideos SOBRIOS:")
        print(
            f"  Total procesados: {sober_stats['successful']}/{sober_stats['total_videos']}"
        )
        print(f"  Rostros extraídos: {sober_stats['total_faces']}")
        print(f"  Fallidos: {sober_stats['failed']}")

    if drunk_stats:
        print("\nVideos EBRIOS:")
        print(
            f"  Total procesados: {drunk_stats['successful']}/{drunk_stats['total_videos']}"
        )
        print(f"  Rostros extraídos: {drunk_stats['total_faces']}")
        print(f"  Fallidos: {drunk_stats['failed']}")

    total_faces = (sober_stats.get("total_faces", 0) if sober_stats else 0) + (
        drunk_stats.get("total_faces", 0) if drunk_stats else 0
    )

    print(f"\nTOTAL DE ROSTROS EXTRAÍDOS: {total_faces}")
    print(f"\nRostros guardados en:")
    print(f"  Sobrios: {os.path.abspath(OUTPUT_SOBER)}")
    print(f"  Ebrios: {os.path.abspath(OUTPUT_DRUNK)}")


def main():
    print("=" * 60)
    print("PIPELINE DE EXTRACCIÓN DE ROSTROS")
    print("Procesamiento secuencial (descarga -> procesa -> borra)")
    print("=" * 60)

    # Crear directorios
    Path(TEMP_VIDEO_DIR).mkdir(parents=True, exist_ok=True)

    # Inicializar extractor
    print(f"\nInicializando detector: {DETECTOR_TYPE}")
    extractor = FaceExtractor(
        detector_type=DETECTOR_TYPE,
        output_size=(FACE_OUTPUT_SIZE, FACE_OUTPUT_SIZE),
        min_confidence=MIN_CONFIDENCE,
        min_face_size=FACE_OUTPUT_SIZE // 2,
    )

    sober_stats = None
    drunk_stats = None

    # Procesar videos de personas sobrias
    if os.path.exists(SOBER_VIDEOS_FILE):
        sober_urls = read_urls_from_file(SOBER_VIDEOS_FILE)
        if sober_urls:
            print(f"\n{'#' * 60}")
            print(f"PROCESANDO {len(sober_urls)} VIDEOS DE PERSONAS SOBRIAS")
            print(f"{'#' * 60}")

            sober_stats = process_category(
                urls=sober_urls,
                output_dir=OUTPUT_SOBER,
                temp_dir=TEMP_VIDEO_DIR,
                extractor=extractor,
                category="sober",
            )
    else:
        print(f"\nArchivo no encontrado: {SOBER_VIDEOS_FILE}")
        print("Saltando videos de personas sobrias...")

    # Procesar videos de personas ebrias
    if os.path.exists(DRUNK_VIDEOS_FILE):
        drunk_urls = read_urls_from_file(DRUNK_VIDEOS_FILE)
        if drunk_urls:
            print(f"\n{'#' * 60}")
            print(f"PROCESANDO {len(drunk_urls)} VIDEOS DE PERSONAS EBRIAS")
            print(f"{'#' * 60}")

            drunk_stats = process_category(
                urls=drunk_urls,
                output_dir=OUTPUT_DRUNK,
                temp_dir=TEMP_VIDEO_DIR,
                extractor=extractor,
                category="drunk",
            )
    else:
        print(f"\nArchivo no encontrado: {DRUNK_VIDEOS_FILE}")
        print("Saltando videos de personas ebrias...")

    # Limpiar carpeta temporal
    try:
        if os.path.exists(TEMP_VIDEO_DIR) and not os.listdir(TEMP_VIDEO_DIR):
            os.rmdir(TEMP_VIDEO_DIR)
    except:
        pass

    # Mostrar resumen
    print_summary(sober_stats, drunk_stats)


if __name__ == "__main__":
    main()
