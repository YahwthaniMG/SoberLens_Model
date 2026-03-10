"""
Pipeline end-to-end de SoberLens.

Ejecuta en orden:
    1. Data augmentation (genera versiones adicionales de cada imagen)
    2. Extraccion de features (landmarks + vectores + lineas + color)
    3. Entrenamiento y evaluacion del clasificador

Uso:
    python pipeline.py

Solo modifica la seccion CONFIGURACION segun tu estructura de carpetas.
"""

from pathlib import Path

from augmentation import augment_folder
from feature_extractor import process_dataset
from train import run_training


# =============================================================================
# CONFIGURACION
# =============================================================================

SOBER_IMAGES_DIR = "../output/sober"
DRUNK_IMAGES_DIR = "../output/drunk"

FEATURES_CSV = "../output/features.csv"

# Cuantas imagenes aumentadas generar por cada original.
# Con 2147 sober + 1842 drunk y factor 4 -> ~15,000 imagenes en total.
# Poner en 0 para saltar la augmentation.
AUGMENTATIONS_PER_IMAGE = 4

# =============================================================================
# FIN CONFIGURACION
# =============================================================================


def main():
    print("=" * 60)
    print("PIPELINE END-TO-END - SoberLens")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Etapa 1: Data Augmentation
    # ------------------------------------------------------------------
    if AUGMENTATIONS_PER_IMAGE > 0:
        print("\n" + "#" * 60)
        print("ETAPA 1: DATA AUGMENTATION")
        print("#" * 60)

        sober_generated = augment_folder(SOBER_IMAGES_DIR, AUGMENTATIONS_PER_IMAGE)
        drunk_generated = augment_folder(DRUNK_IMAGES_DIR, AUGMENTATIONS_PER_IMAGE)

        print(f"\nAugmentation completada:")
        print(f"  Sober generadas: {sober_generated}")
        print(f"  Drunk generadas: {drunk_generated}")
    else:
        print("\nAugmentation desactivada (AUGMENTATIONS_PER_IMAGE = 0)")

    # ------------------------------------------------------------------
    # Etapa 2: Extraccion de Features
    # ------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("ETAPA 2: EXTRACCION DE FEATURES")
    print("#" * 60)

    df = process_dataset(
        sober_dir=SOBER_IMAGES_DIR,
        drunk_dir=DRUNK_IMAGES_DIR,
        output_csv=FEATURES_CSV,
    )

    print(f"\nFeatures extraidos: {len(df)} muestras")

    # ------------------------------------------------------------------
    # Etapa 3: Entrenamiento
    # ------------------------------------------------------------------
    print("\n" + "#" * 60)
    print("ETAPA 3: ENTRENAMIENTO DEL CLASIFICADOR")
    print("#" * 60)

    clf, scaler, feature_cols, accuracy = run_training()

    # ------------------------------------------------------------------
    # Resumen final
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO")
    print("=" * 60)
    print(f"Dataset final: {len(df)} imagenes")
    print(f"Features por muestra: {len(feature_cols)}")
    print(f"Accuracy del modelo: {accuracy * 100:.2f}%")
    print(f"\nArchivos generados:")
    print(f"  {FEATURES_CSV}")
    print(f"  ../output/models/model.pkl")
    print(f"  ../output/models/scaler.pkl")
    print(f"  ../output/models/features.txt")
    print(f"  ../output/models/metadata.txt")


if __name__ == "__main__":
    main()
