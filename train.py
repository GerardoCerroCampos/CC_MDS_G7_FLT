"""
train.py — Entrenamiento y evaluación del modelo de atraso de vuelos (Paso 2).

Flujo completo e integrado con el EDA:

  1. Lee el CONTRATO de datos generado por EDA.ipynb (contrato_datos.json):
     lista autorizada de predictores, categóricas/numéricas, etiqueta y semilla.
     Si no está, cae a la definición de app/features.py (misma lista).
  2. Lee la partición del EDA: data/train.csv (ajuste) y data/test.csv (hold-out).
     Si no existen, genera un dataset sintético reproducible para correr end-to-end
     sin la descarga de Kaggle (~5.8M filas).
  3. Entrena un Pipeline COMPLETO (preprocesamiento + estimador) solo con train.
  4. Evalúa con metodología defendible:
       - Validación cruzada estratificada (5 folds) sobre train, varias métricas.
       - Evaluación final sobre el hold-out (test) que el EDA reservó.
       - Baseline de clase mayoritaria para dimensionar el desempeño.
       - Métricas apropiadas al desbalance: F1-macro, recall y precisión de la
         clase positiva, ROC-AUC y PR-AUC, además de exactitud y balanced acc.
       - Matriz de confusión e interpretación textual.
  5. Serializa el pipeline (model/model.pkl) y escribe:
       - model/metadata.json  (para /model-info)
       - reportes/evaluacion.json, reportes/matriz_confusion.csv,
         reportes/classification_report.csv  (evidencia reproducible)

Ausencia de fuga de información: los predictores provienen del contrato (nunca
las columnas de auditoría como ARRIVAL_DELAY), el preprocesamiento se ajusta solo
con train dentro del pipeline, y test no se toca hasta la evaluación final.

Uso:
    python train.py
    MODELO=random_forest python train.py     # estimador alternativo
"""

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import joblib

from app.features import (
    CATEGORICAS as CAT_DEFECTO,
    NUMERICAS as NUM_DEFECTO,
    FEATURES as FEAT_DEFECTO,
    TARGET as TARGET_DEFECTO,
    canonizar_aeropuerto,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"
REPORTES_DIR = BASE_DIR / "reportes"
CONTRATO_PATH = BASE_DIR / "contrato_datos.json"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"

RANDOM_STATE = 42
MODELO = os.environ.get("MODELO", "logistic")  # 'logistic' | 'random_forest'

# Columnas que el EDA reserva SOLO para auditoría: nunca deben ser predictores.
COLUMNAS_AUDITORIA = ["SOURCE_ROW", "FLIGHT_DATE", "ARRIVAL_DELAY"]

# Valores de ejemplo para el dataset sintético de respaldo
AIRLINES = ["AA", "AS", "B6", "DL", "F9", "HA", "MQ", "NK", "OO", "UA", "US", "VX", "WN", "EV"]
AIRPORTS = [
    "ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MCO",
    "EWR", "CLT", "PHX", "IAH", "MIA", "BOS", "MSP", "DTW", "FLL", "PHL",
    "LGA", "BWI", "SLC", "SAN", "IAD", "DCA", "MDW", "TPA", "PDX", "HNL",
]


# --------------------------------------------------------------------------- #
# Contrato de datos
# --------------------------------------------------------------------------- #
def leer_contrato() -> dict:
    """Carga el contrato del EDA; si falta, usa la definición de app/features.py."""
    if CONTRATO_PATH.exists():
        contrato = json.loads(CONTRATO_PATH.read_text(encoding="utf-8"))
        print(f"[contrato] Leído de {CONTRATO_PATH.name}")
        contrato.setdefault("categoricas", CAT_DEFECTO)
        contrato.setdefault("numericas", NUM_DEFECTO)
        contrato.setdefault("target", TARGET_DEFECTO)
        contrato.setdefault("semilla", RANDOM_STATE)
    else:
        print("[contrato] contrato_datos.json no encontrado -> uso app/features.py")
        contrato = {
            "features": FEAT_DEFECTO,
            "categoricas": CAT_DEFECTO,
            "numericas": NUM_DEFECTO,
            "target": TARGET_DEFECTO,
            "columnas_solo_auditoria": COLUMNAS_AUDITORIA,
            "semilla": RANDOM_STATE,
        }
    # La lista de features autorizada = categóricas + numéricas del contrato.
    contrato["features"] = list(contrato["categoricas"]) + list(contrato["numericas"])
    # Comprobación de no fuga: ningún predictor puede ser columna de auditoría.
    fuga = set(contrato["features"]) & set(contrato.get("columnas_solo_auditoria", COLUMNAS_AUDITORIA))
    assert not fuga, f"Fuga de información: predictores de auditoría {fuga}"
    # Coherencia con la API.
    if set(contrato["features"]) != set(FEAT_DEFECTO):
        print("[contrato] AVISO: las features del contrato difieren de app/features.py")
    return contrato


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
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
    sched_time = np.round(distance / 7.5 + rng.uniform(20, 60, size=n), 0)
    logit = (
        -1.6
        + 0.0009 * dep_min
        + 0.00015 * distance
        + 0.35 * np.isin(airline, ["NK", "F9", "B6"])
        + 0.25 * (dow >= 5)
        + rng.normal(0, 0.5, size=n)
    )
    prob = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < prob).astype(int)
    return pd.DataFrame(
        {
            "AIRLINE": airline,
            "ORIGIN_AIRPORT": [canonizar_aeropuerto(a) for a in origin],
            "DESTINATION_AIRPORT": [canonizar_aeropuerto(a) for a in dest],
            "MONTH": month, "DAY": day, "DAY_OF_WEEK": dow,
            "SCHEDULED_DEPARTURE_MIN": dep_min,
            "SCHEDULED_TIME": sched_time.astype(float),
            "DISTANCE": distance.astype(float),
            TARGET_DEFECTO: y,
        }
    )


def cargar_datos(contrato: dict):
    """Devuelve (X_train, X_test, y_train, y_test, origen)."""
    features, target = contrato["features"], contrato["target"]
    if TRAIN_CSV.exists() and TEST_CSV.exists():
        print(f"[datos] Partición del EDA: {TRAIN_CSV.name} + {TEST_CSV.name}")
        train = pd.read_csv(TRAIN_CSV)
        test = pd.read_csv(TEST_CSV)
        for nombre, d in [("train", train), ("test", test)]:
            faltan = [c for c in features + [target] if c not in d.columns]
            assert not faltan, f"Faltan columnas en {nombre}.csv: {faltan}"
        return (train[features].copy(), test[features].copy(),
                train[target].astype(int), test[target].astype(int), "eda_csv")

    print("[datos] data/train.csv no encontrado -> dataset sintético de respaldo.")
    df = generar_dataset_sintetico()
    X, y = df[features].copy(), df[target].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=contrato["semilla"]
    )
    return Xtr, Xte, ytr, yte, "sintetico"


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
def construir_estimador(nombre: str):
    if nombre == "random_forest":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1,
        )
    # Por defecto: modelo lineal con manejo de desbalance.
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)


def construir_pipeline(contrato: dict, nombre_modelo: str) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), contrato["categoricas"]),
            ("num", StandardScaler(), contrato["numericas"]),
        ]
    )
    return Pipeline([("pre", pre), ("clf", construir_estimador(nombre_modelo))])


# --------------------------------------------------------------------------- #
# Evaluación
# --------------------------------------------------------------------------- #
def metricas_test(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision_pos": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_pos": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_pos": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
    }


def interpretar(metricas: dict, base: dict, prevalencia: float) -> str:
    recall = metricas["recall_pos"]
    precision = metricas["precision_pos"]
    mejora_f1 = metricas["f1_macro"] - base["f1_macro"]
    return (
        f"El {prevalencia:.1%} de los vuelos de entrenamiento tienen atraso >= 15 min "
        f"(clase positiva), de modo que la exactitud por sí sola es engañosa. "
        f"En el hold-out, el modelo detecta el {recall:.1%} de los atrasos reales "
        f"(recall) y, de sus alertas de atraso, acierta el {precision:.1%} (precisión). "
        f"El ROC-AUC es {metricas['roc_auc']:.3f} y el PR-AUC {metricas['pr_auc']:.3f}. "
        f"Frente al baseline de clase mayoritaria (F1-macro {base['f1_macro']:.3f}), "
        f"el modelo mejora el F1-macro en {mejora_f1:+.3f} puntos, lo que confirma que "
        f"aprende señal por encima de predecir siempre la clase más frecuente."
    )


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTES_DIR.mkdir(parents=True, exist_ok=True)

    contrato = leer_contrato()
    X_train, X_test, y_train, y_test, origen = cargar_datos(contrato)
    prevalencia = float(y_train.mean())
    print(f"[datos] train={len(X_train):,} | test={len(X_test):,} | "
          f"prevalencia atraso={prevalencia:.3f} | origen={origen}")

    pipe = construir_pipeline(contrato, MODELO)

    # --- Validación cruzada estratificada sobre train (varias métricas) ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["f1_macro", "roc_auc", "recall", "precision", "balanced_accuracy"]
    cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    cv_resumen = {
        m: {"mean": round(float(cv_res[f"test_{m}"].mean()), 4),
            "std": round(float(cv_res[f"test_{m}"].std()), 4)}
        for m in scoring
    }
    print("[cv] " + " | ".join(
        f"{m}={cv_resumen[m]['mean']:.4f}±{cv_resumen[m]['std']:.4f}" for m in scoring))

    # --- Baseline de clase mayoritaria (referencia) ---
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    base_pred = dummy.predict(X_test)
    base_proba = dummy.predict_proba(X_test)[:, 1]
    base_metricas = metricas_test(y_test, base_pred, base_proba)

    # --- Ajuste final sobre train y evaluación en el hold-out (test) ---
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    metricas = metricas_test(y_test, y_pred, y_proba)
    print(f"[test] {metricas}")

    # Matriz de confusión y classification report
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    reporte = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    interpretacion = interpretar(metricas, base_metricas, prevalencia)
    print("[interpretacion] " + interpretacion)

    # --- Serialización del pipeline ---
    model_path = MODEL_DIR / "model.pkl"
    joblib.dump(pipe, model_path)
    print(f"[modelo] Pipeline serializado en {model_path} "
          f"({model_path.stat().st_size / 1024:.1f} KB)")

    # --- metadata.json (para /model-info) ---
    metadata = {
        "model_type": type(pipe.named_steps["clf"]).__name__,
        "task": "binary_classification",
        "target": contrato["target"],
        "target_rule": contrato.get("regla_objetivo", "ARRIVAL_DELAY >= 15"),
        "positive_class": "1 = atraso de llegada >= 15 min",
        "population": contrato.get(
            "poblacion", "Vuelos no cancelados ni desviados, con atraso observado"),
        "prediction_moment": contrato.get("momento_prediccion", "Antes de la salida programada"),
        "sklearn_version": sklearn.__version__,
        "features": contrato["features"],
        "categorical_features": contrato["categoricas"],
        "numeric_features": contrato["numericas"],
        "metrics": metricas,
        "cv_metrics": cv_resumen,
        "baseline_metrics": base_metricas,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "class_prevalence_train": round(prevalencia, 4),
        "interpretation": interpretacion,
        "model_version": "1.0.0",
        "data_source": origen,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "seed": RANDOM_STATE,
    }
    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[modelo] Metadatos escritos en {MODEL_DIR / 'metadata.json'}")

    # --- Reportes reproducibles ---
    (REPORTES_DIR / "evaluacion.json").write_text(
        json.dumps(
            {"modelo": metadata["model_type"], "origen_datos": origen,
             "cv": cv_resumen, "test": metricas, "baseline": base_metricas,
             "confusion_matrix": metadata["confusion_matrix"],
             "interpretacion": interpretacion},
            indent=2, ensure_ascii=False),
        encoding="utf-8")
    pd.DataFrame(cm, index=["real_0", "real_1"], columns=["pred_0", "pred_1"]).to_csv(
        REPORTES_DIR / "matriz_confusion.csv")
    pd.DataFrame(reporte).T.to_csv(REPORTES_DIR / "classification_report.csv")
    print(f"[reportes] evaluacion.json, matriz_confusion.csv y "
          f"classification_report.csv en {REPORTES_DIR.name}/")

    # --- Gráfico opcional de la matriz de confusión (si matplotlib está disponible) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], ["Real 0", "Real 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title("Matriz de confusion (hold-out)")
        fig.tight_layout()
        fig.savefig(REPORTES_DIR / "matriz_confusion.png", dpi=150)
        plt.close(fig)
        print(f"[reportes] matriz_confusion.png en {REPORTES_DIR.name}/")
    except Exception:
        pass  # matplotlib es opcional para el entrenamiento


if __name__ == "__main__":
    main()
