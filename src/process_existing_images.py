"""
Script para procesar imágenes de un dataset existente.
Detecta rostros, los recorta y los guarda en tamaño estándar (224x224).

Modifica las variables en la sección CONFIGURACION y ejecuta:
    python process_existing_images.py
"""

import os
from pathlib import Path

import cv2

from face_extractor import FaceExtractor


# =============================================================================
# CONFIGURACION
# =============================================================================

# Carpetas de entrada con las imágenes originales del dataset
INPUT_SOBER = "../data/dataset_images/sober"
INPUT_DRUNK = "../data/dataset_images/drunk"

# Carpetas de salida (las mismas que usa main.py)
OUTPUT_SOBER = "../output/sober/example"
OUTPUT_DRUNK = "../output/drunk"

# Configuración del detector
DETECTOR_TYPE = "opencv"
FACE_OUTPUT_SIZE = 224
MIN_CONFIDENCE = 0.7

# Extensiones de imagen a procesar
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# =============================================================================
# FIN DE CONFIGURACION
# =============================================================================


def process_image(
    image_path: str,
    output_dir: str,
    extractor: FaceExtractor,
    image_index: int,
    prefix: str,
) -> int:
    """
    Procesa una imagen: detecta rostros y los guarda.

    Returns:
        Número de rostros extraídos de la imagen
    """
    image = cv2.imread(image_path)

    if image is None:
        print(f"  No se pudo leer: {image_path}")
        return 0

    # Detectar rostros
    faces = extractor.process_frame(image)

    if not faces:
        print(f"  Sin rostros: {Path(image_path).name}")
        return 0

    # Guardar cada rostro detectado
    for face_idx, face in enumerate(faces):
        filename = f"{prefix}_img{image_index:04d}_face{face_idx:02d}.jpg"
        output_path = Path(output_dir) / filename
        cv2.imwrite(str(output_path), face, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return len(faces)


def process_folder(
    input_dir: str, output_dir: str, extractor: FaceExtractor, prefix: str
) -> dict:
    """
    Procesa todas las imágenes de una carpeta.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {"images_processed": 0, "images_skipped": 0, "faces_extracted": 0}

    if not input_dir.exists():
        print(f"Carpeta no encontrada: {input_dir}")
        return stats

    # Obtener lista de imágenes
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))

    image_files = sorted(set(image_files))

    if not image_files:
        print(f"No se encontraron imágenes en: {input_dir}")
        return stats

    print(f"Procesando {len(image_files)} imágenes de: {input_dir}")

    for idx, image_path in enumerate(image_files):
        faces_count = process_image(
            str(image_path), str(output_dir), extractor, idx, prefix
        )

        if faces_count > 0:
            stats["images_processed"] += 1
            stats["faces_extracted"] += faces_count
        else:
            stats["images_skipped"] += 1

        # Mostrar progreso cada 50 imágenes
        if (idx + 1) % 50 == 0:
            print(f"  Progreso: {idx + 1}/{len(image_files)}")

    return stats


def main():
    print("=" * 60)
    print("PROCESAMIENTO DE DATASET DE IMAGENES")
    print("=" * 60)

    # Inicializar extractor
    print(f"\nInicializando detector: {DETECTOR_TYPE}")
    extractor = FaceExtractor(
        detector_type=DETECTOR_TYPE,
        output_size=(FACE_OUTPUT_SIZE, FACE_OUTPUT_SIZE),
        min_confidence=MIN_CONFIDENCE,
        min_face_size=FACE_OUTPUT_SIZE // 2,
    )

    # Procesar imágenes de personas sobrias
    print("\n" + "=" * 60)
    print("PROCESANDO IMAGENES - SOBRIOS")
    print("=" * 60)
    sober_stats = process_folder(
        INPUT_SOBER, OUTPUT_SOBER, extractor, prefix="dataset_sober"
    )

    # Procesar imágenes de personas ebrias
    print("\n" + "=" * 60)
    print("PROCESANDO IMAGENES - EBRIOS")
    print("=" * 60)
    drunk_stats = process_folder(
        INPUT_DRUNK, OUTPUT_DRUNK, extractor, prefix="dataset_drunk"
    )

    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)

    print("\nSOBRIOS:")
    print(f"  Imágenes procesadas: {sober_stats['images_processed']}")
    print(f"  Imágenes sin rostro: {sober_stats['images_skipped']}")
    print(f"  Rostros extraídos: {sober_stats['faces_extracted']}")

    print("\nEBRIOS:")
    print(f"  Imágenes procesadas: {drunk_stats['images_processed']}")
    print(f"  Imágenes sin rostro: {drunk_stats['images_skipped']}")
    print(f"  Rostros extraídos: {drunk_stats['faces_extracted']}")

    total = sober_stats["faces_extracted"] + drunk_stats["faces_extracted"]
    print(f"\nTOTAL ROSTROS EXTRAIDOS: {total}")
    print(f"\nGuardados en:")
    print(f"  Sobrios: {os.path.abspath(OUTPUT_SOBER)}")
    print(f"  Ebrios: {os.path.abspath(OUTPUT_DRUNK)}")


if __name__ == "__main__":
    main()
