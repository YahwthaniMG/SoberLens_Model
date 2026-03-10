"""
Data augmentation para el dataset de imagenes de rostros.

Simula condiciones reales de bar/fiesta siguiendo el paper DrunkSelfie:
- Rotacion leve (-8 a +8 grados)
- Cambios de brillo
- Blur gaussiano
- Cambio de perspectiva (flip horizontal aleatorio)
- Tinte de color (simula luces de bar)
- Cambio de contraste

Genera N versiones adicionales de cada imagen original.
Las imagenes aumentadas se guardan en la misma carpeta con sufijo _aug.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List
import random


# =============================================================================
# CONFIGURACION
# =============================================================================

AUGMENTATIONS_PER_IMAGE = 4  # Cuantas versiones adicionales generar por imagen
RANDOM_SEED = 42

# =============================================================================
# FIN CONFIGURACION
# =============================================================================


def rotate_image(image: np.ndarray, max_angle: float = 8.0) -> np.ndarray:
    """Rotacion aleatoria leve."""
    h, w = image.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def adjust_brightness(image: np.ndarray, max_delta: int = 45) -> np.ndarray:
    """Cambio de brillo aleatorio."""
    delta = random.randint(-max_delta, max_delta)
    result = image.astype(np.int16) + delta
    return np.clip(result, 0, 255).astype(np.uint8)


def gaussian_blur(image: np.ndarray, max_sigma: float = 1.0) -> np.ndarray:
    """Blur gaussiano leve."""
    sigma = random.uniform(0, max_sigma)
    if sigma < 0.1:
        return image
    ksize = 3
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def horizontal_flip(image: np.ndarray, prob: float = 0.5) -> np.ndarray:
    """Flip horizontal con probabilidad dada."""
    if random.random() < prob:
        return cv2.flip(image, 1)
    return image


def adjust_contrast(image: np.ndarray, max_factor: float = 0.3) -> np.ndarray:
    """Cambio de contraste aleatorio."""
    factor = 1.0 + random.uniform(-max_factor, max_factor)
    mean = np.mean(image.astype(np.float32))
    result = mean + factor * (image.astype(np.float32) - mean)
    return np.clip(result, 0, 255).astype(np.uint8)


def add_color_tint(image: np.ndarray, max_tint: float = 0.15) -> np.ndarray:
    """
    Agrega un tinte de color aleatorio a un canal (R, G o B).
    Simula iluminacion de bar (luces rojas, verdes o azules).
    """
    channel = random.randint(0, 2)
    tint = random.uniform(-max_tint, max_tint)
    result = image.astype(np.float32)
    result[:, :, channel] = np.clip(result[:, :, channel] * (1 + tint), 0, 255)
    return result.astype(np.uint8)


def perspective_transform(image: np.ndarray, max_scale: float = 0.05) -> np.ndarray:
    """Transformacion de perspectiva leve."""
    h, w = image.shape[:2]
    scale = random.uniform(0, max_scale)
    dx = int(w * scale)
    dy = int(h * scale)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32(
        [
            [random.randint(0, dx), random.randint(0, dy)],
            [w - random.randint(0, dx), random.randint(0, dy)],
            [w - random.randint(0, dx), h - random.randint(0, dy)],
            [random.randint(0, dx), h - random.randint(0, dy)],
        ]
    )

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


ALL_AUGMENTATIONS = [
    rotate_image,
    adjust_brightness,
    gaussian_blur,
    horizontal_flip,
    adjust_contrast,
    add_color_tint,
    perspective_transform,
]


def augment_image(image: np.ndarray) -> np.ndarray:
    """
    Aplica una combinacion aleatoria de augmentaciones a una imagen.
    Siempre aplica entre 2 y 4 augmentaciones.
    """
    n_augs = random.randint(2, 4)
    selected = random.sample(ALL_AUGMENTATIONS, n_augs)
    result = image.copy()
    for aug_fn in selected:
        result = aug_fn(result)
    return result


def augment_folder(
    folder: str,
    augmentations_per_image: int = AUGMENTATIONS_PER_IMAGE,
    seed: int = RANDOM_SEED,
) -> int:
    """
    Genera versiones aumentadas de todas las imagenes en una carpeta.
    Las guarda en la misma carpeta con sufijo _aug{N}.jpg

    Args:
        folder: carpeta con imagenes originales
        augmentations_per_image: cuantas versiones adicionales por imagen

    Returns:
        Total de imagenes generadas
    """
    random.seed(seed)
    np.random.seed(seed)

    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"Carpeta no encontrada: {folder_path}")
        return 0

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    # Solo procesar imagenes originales (sin _aug en el nombre)
    images = sorted(
        [
            p
            for p in folder_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS and "_aug" not in p.stem
        ]
    )

    if not images:
        print(f"No se encontraron imagenes originales en: {folder_path}")
        return 0

    print(f"Augmentando {len(images)} imagenes en: {folder_path}")
    total_generated = 0

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        for n in range(augmentations_per_image):
            augmented = augment_image(image)
            out_name = f"{img_path.stem}_aug{n:02d}{img_path.suffix}"
            out_path = folder_path / out_name
            cv2.imwrite(str(out_path), augmented, [cv2.IMWRITE_JPEG_QUALITY, 90])
            total_generated += 1

    print(f"  Imagenes generadas: {total_generated}")
    print(f"  Total en carpeta: {len(images) + total_generated}")
    return total_generated
