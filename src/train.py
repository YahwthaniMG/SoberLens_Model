"""
Entrenamiento del clasificador sobrio/ebrio.

Pipeline:
    1. Carga el CSV de features generado por feature_extractor.py
    2. Normaliza features con StandardScaler
    3. Entrena y compara multiples clasificadores con 10-fold cross-validation
    4. Evalua el mejor en el conjunto de prueba (80/20 split)
    5. Guarda el modelo y el scaler como archivos .pkl

Basado en DrunkSelfie (Willoughby et al., 2019):
- Mejores features: coordenadas X,Y + vectores + distancias de lineas
- Mejor clasificador: Random Forest / Gradient Boosting
- Split: 80/20 con 10-fold cross-validation
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =============================================================================
# CONFIGURACION
# =============================================================================

FEATURES_CSV = "../output/features.csv"
MODELS_OUTPUT_DIR = "../output/models"

TEST_SIZE = 0.20
RANDOM_SEED = 42
CV_FOLDS = 10

# =============================================================================
# FIN CONFIGURACION
# =============================================================================


def load_dataset(csv_path: str):
    """
    Carga el CSV de features y separa X (features) de y (labels).

    Columnas especiales que se descartan del vector de features:
        "image" -> nombre del archivo
        "label" -> etiqueta de clase (0=sober, 1=drunk)
    """
    df = pd.read_csv(csv_path)
    print(f"Dataset cargado: {len(df)} muestras, {len(df.columns)} columnas")
    print(f"Clase 0 (sober): {(df['label'] == 0).sum()}")
    print(f"Clase 1 (drunk): {(df['label'] == 1).sum()}")

    drop_cols = [c for c in ["image", "label"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values.astype(np.int32)

    # Reemplazar NaN o Inf que puedan haber quedado de features fallidos
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    print(f"Features por muestra: {X.shape[1]}")
    return X, y, feature_cols


def build_classifiers():
    """
    Define los clasificadores a comparar.
    Parametros basados en los resultados del paper DrunkSelfie.
    """
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=RANDOM_SEED,
        ),
        "SVM RBF": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=RANDOM_SEED,
        ),
        "SVM Lineal": SVC(
            kernel="linear",
            C=1.0,
            probability=True,
            random_state=RANDOM_SEED,
        ),
    }


def evaluate_classifiers(X_train, y_train, classifiers: dict) -> dict:
    """
    Evalua cada clasificador con k-fold cross-validation en el conjunto de entrenamiento.

    Returns:
        Diccionario {nombre: mean_cv_accuracy}
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results = {}

    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION ({CV_FOLDS} folds) en conjunto de entrenamiento")
    print(f"{'='*60}")

    for name, clf in classifiers.items():
        scores = cross_val_score(
            clf, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
        )
        mean_acc = float(scores.mean())
        std_acc = float(scores.std())
        results[name] = mean_acc
        print(f"  {name:<25} {mean_acc * 100:.2f}% (+/- {std_acc * 100:.2f}%)")

    return results


def train_best_model(X_train, y_train, classifiers: dict, cv_results: dict):
    """
    Re-entrena el mejor clasificador con todos los datos de entrenamiento.

    Returns:
        (nombre, clasificador entrenado)
    """
    best_name = max(cv_results, key=cv_results.get)
    best_clf = classifiers[best_name]

    print(f"\nMejor clasificador: {best_name} ({cv_results[best_name] * 100:.2f}% CV)")
    print("Entrenando con el 80% completo de los datos...")
    best_clf.fit(X_train, y_train)
    print("Entrenamiento completado.")

    return best_name, best_clf


def evaluate_on_test(clf, X_test, y_test, clf_name: str):
    """Evalua el modelo final en el conjunto de prueba."""
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"EVALUACION FINAL EN CONJUNTO DE PRUEBA (20%)")
    print(f"{'='*60}")
    print(f"Clasificador: {clf_name}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"\nReporte por clase:")
    print(classification_report(y_test, y_pred, target_names=["sober", "drunk"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Matriz de confusion:")
    print(f"              Predicho sober  Predicho drunk")
    print(f"  Real sober       {cm[0, 0]:5d}          {cm[0, 1]:5d}")
    print(f"  Real drunk       {cm[1, 0]:5d}          {cm[1, 1]:5d}")

    return acc, cm


def save_model(
    clf, scaler, feature_cols, output_dir: str, clf_name: str, accuracy: float
):
    """
    Guarda el modelo, el scaler y los nombres de features en disco.

    Archivos generados:
        model.pkl     -> clasificador entrenado
        scaler.pkl    -> StandardScaler ajustado con los datos de entrenamiento
        features.txt  -> lista de features en el orden correcto (para inferencia)
        metadata.txt  -> informacion del modelo
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, output_dir / "model.pkl")
    joblib.dump(scaler, output_dir / "scaler.pkl")

    with open(output_dir / "features.txt", "w") as f:
        for col in feature_cols:
            f.write(col + "\n")

    with open(output_dir / "metadata.txt", "w") as f:
        f.write(f"Clasificador: {clf_name}\n")
        f.write(f"Accuracy en test: {accuracy * 100:.2f}%\n")
        f.write(f"Features: {len(feature_cols)}\n")
        f.write(f"Clases: 0=sober, 1=drunk\n")

    print(f"\nModelo guardado en: {output_dir}")
    print(f"  model.pkl     -> clasificador ({clf_name})")
    print(f"  scaler.pkl    -> normalizador")
    print(f"  features.txt  -> {len(feature_cols)} features")
    print(f"  metadata.txt  -> informacion del modelo")


def run_training():
    """Pipeline completo de entrenamiento."""
    print("=" * 60)
    print("PIPELINE DE ENTRENAMIENTO - SoberLens")
    print("=" * 60)

    # 1. Cargar datos
    print(f"\nCargando features desde: {FEATURES_CSV}")
    X, y, feature_cols = load_dataset(FEATURES_CSV)

    # 2. Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    print(f"\nSplit -> Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")

    # 3. Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Cross-validation de todos los clasificadores
    classifiers = build_classifiers()
    cv_results = evaluate_classifiers(X_train_scaled, y_train, classifiers)

    # 5. Entrenar el mejor
    best_name, best_clf = train_best_model(
        X_train_scaled, y_train, classifiers, cv_results
    )

    # 6. Evaluacion final en test
    accuracy, cm = evaluate_on_test(best_clf, X_test_scaled, y_test, best_name)

    # 7. Guardar
    save_model(
        clf=best_clf,
        scaler=scaler,
        feature_cols=feature_cols,
        output_dir=MODELS_OUTPUT_DIR,
        clf_name=best_name,
        accuracy=accuracy,
    )

    return best_clf, scaler, feature_cols, accuracy


if __name__ == "__main__":
    run_training()
