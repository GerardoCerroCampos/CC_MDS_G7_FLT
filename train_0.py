"""
train.py — Entrenamiento y serialización del modelo de atraso de vuelos.

Entrena un Pipeline COMPLETO (preprocesamiento + estimador) para predecir
ARRIVAL_DELAY_15 (atraso de llegada >= 15 min) a partir de los 9 predictores
definidos en el EDA, y lo serializa en model/model.pkl + model/metadata.json.

Fuente de datos:
  1) Si existen data/train.csv y data/test.csv (salidas del EDA.ipynb), se usan
     esos archivos: ya traen los 9 predictores en su forma canónica y la etiqueta.
  2) Si no existen, se genera un dataset sintético reproducible con el mismo
     esquema, para que el proyecto corra end-to-end sin la descarga de Kaggle
     (~5.8M filas). Reentrena con los CSV reales cuando estén disponibles.

Uso:
    python train.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib

from app.features import (
    CATEGORICAS,
    NUMERICAS,
    FEATURES,
    TARGET,
    canonizar_aeropuerto,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

RANDOM_STATE = 42

# Valores de ejemplo para el dataset sintético (subconjunto realista)
AIRLINES = ["AA", "AS", "B6", "DL", "F9", "HA", "MQ", "NK", "OO", "UA", "US", "VX", "WN", "EV"]
AIRPORTS = [
    "ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MCO",
    "EWR", "CLT", "PHX", "IAH", "MIA", "BOS", "MSP", "DTW", "FLL", "PHL",
    "LGA", "BWI", "SLC", "SAN", "IAD", "DCA", "MDW", "TPA", "PDX", "HNL",
]


def generar_dataset_sintetico(n: int = 20000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Dataset sintético con el esquema del EDA (categóricas ya canonizadas)."""
    rng = np.random.default_rng(seed)
    airline = rng.choice(AIRLINES, size=n)
    origin = rng.choice(AIRPORTS, size=n)
    dest = rng.choice(AIRPORTS, size=n)

    month = rng.integers(1, 13, size=n)
    day = rng.integers(1, 29, size=n)
    dow = rng.integers(1, 8, size=n)
    dep_min = rng.integers(0, 1440, size=n)
    distance = np.round(rng.uniform(100, 2800, size=n), 0)
    # Duración correlacionada con distancia + ruido
    sched_time = np.round(distance / 7.5 + rng.uniform(20, 60, size=n), 0)

    # Probabilidad de atraso: mayor en la tarde/noche, ciertas aerolíneas y rutas largas
    logit = (
        -1.6
        + 0.0009 * dep_min                      # más tarde -> más atraso
        + 0.00015 * distance
        + 0.35 * np.isin(airline, ["NK", "F9", "B6"])
        + 0.25 * (dow >= 5)
        + rng.normal(0, 0.5, size=n)
    )
    prob = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < prob).astype(int)

    df = pd.DataFrame(
        {
            "AIRLINE": airline,
            "ORIGIN_AIRPORT": [canonizar_aeropuerto(a) for a in origin],
            "DESTINATION_AIRPORT": [canonizar_aeropuerto(a) for a in dest],
            "MONTH": month,
            "DAY": day,
            "DAY_OF_WEEK": dow,
            "SCHEDULED_DEPARTURE_MIN": dep_min,
            "SCHEDULED_TIME": sched_time.astype(float),
            "DISTANCE": distance.astype(float),
            TARGET: y,
        }
    )
    return df


def cargar_datos():
    """Devuelve (X_train, X_test, y_train, y_test, origen)."""
    if TRAIN_CSV.exists() and TEST_CSV.exists():
        print(f"[datos] Usando {TRAIN_CSV.name} y {TEST_CSV.name} del EDA.")
        train = pd.read_csv(TRAIN_CSV)
        test = pd.read_csv(TEST_CSV)
        Xtr, ytr = train[FEATURES].copy(), train[TARGET].astype(int)
        Xte, yte = test[FEATURES].copy(), test[TARGET].astype(int)
        return Xtr, Xte, ytr, yte, "eda_csv"

    print("[datos] data/train.csv no encontrado -> generando dataset sintético.")
    df = generar_dataset_sintetico()
    X, y = df[FEATURES].copy(), df[TARGET].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    return Xtr, Xte, ytr, yte, "sintetico"


def construir_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
            ("num", StandardScaler(), NUMERICAS),
        ]
    )
    # Clasificador lineal con manejo de desbalance (el EDA documenta clases
    # desbalanceadas). Se puede sustituir por otro estimador sin cambiar la API.
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test, origen = cargar_datos()

    pipe = construir_pipeline()

    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
    print(f"[cv] F1-macro (5-fold) = {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "precision_pos": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall_pos": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "cv_f1_macro_mean": round(float(cv_f1.mean()), 4),
        "cv_f1_macro_std": round(float(cv_f1.std()), 4),
    }
    print(f"[test] {metrics}")

    model_path = MODEL_DIR / "model.pkl"
    joblib.dump(pipe, model_path)
    print(f"[modelo] Pipeline serializado en {model_path} "
          f"({model_path.stat().st_size / 1024:.1f} KB)")

    metadata = {
        "model_type": type(pipe.named_steps["clf"]).__name__,
        "task": "binary_classification",
        "target": TARGET,
        "target_rule": "ARRIVAL_DELAY >= 15",
        "positive_class": "1 = atraso de llegada >= 15 min",
        "population": "Vuelos no cancelados ni desviados, con atraso de llegada observado",
        "prediction_moment": "Antes de la salida programada",
        "sklearn_version": sklearn.__version__,
        "features": FEATURES,
        "categorical_features": CATEGORICAS,
        "numeric_features": NUMERICAS,
        "metrics": metrics,
        "model_version": "1.0.0",
        "data_source": origen,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    meta_path = MODEL_DIR / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[modelo] Metadatos escritos en {meta_path}")


if __name__ == "__main__":
    main()
