"""
app/main.py — API de inferencia con FastAPI (atraso de vuelos).

Expone el modelo que predice ARRIVAL_DELAY_15 (atraso de llegada >= 15 min):
  GET  /health         estado + confirmación de modelo cargado
  GET  /model-info     metadatos (tipo, target, features, métricas, versión)
  POST /predict        predicción de un vuelo (con probabilidad)
  POST /predict-batch  predicción de una lista de vuelos (mismo orden)
  GET  /docs           Swagger UI (automática)

Diseño exigido por la tarea:
  * El modelo se carga UNA vez al iniciar (lifespan), no por petición.
  * La validación de entrada la hace Pydantic -> entradas inválidas => 422.
  * Fallo interno del modelo => 500 con mensaje controlado (sin trazas al cliente).
  * Modelo no disponible => 503.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import logging

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from app.features import FEATURES, canonizar_registro
from app.schemas import (
    Vuelo,
    LoteVuelos,
    RespuestaPrediccion,
    RespuestaLote,
    RespuestaSalud,
    RespuestaModelInfo,
)

logger = logging.getLogger("api-vuelos")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
METADATA_PATH = BASE_DIR / "model" / "metadata.json"

ARTIFACTS: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo y los metadatos una sola vez, al arrancar el servicio."""
    try:
        ARTIFACTS["model"] = joblib.load(MODEL_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            ARTIFACTS["metadata"] = json.load(f)
        logger.info("Modelo y metadatos cargados correctamente.")
    except FileNotFoundError:
        logger.error("No se encontró el modelo. Ejecuta 'python train.py' primero.")
    yield
    ARTIFACTS.clear()
    logger.info("Artefactos liberados al cerrar la aplicación.")


app = FastAPI(
    title="API de inferencia — Atraso de vuelos",
    description=(
        "Predice si un vuelo llegará con 15 minutos o más de atraso "
        "(ARRIVAL_DELAY_15), usando solo información conocida antes de la salida."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _model_version() -> str:
    return ARTIFACTS.get("metadata", {}).get("model_version", "desconocida")


def _requiere_modelo():
    modelo = ARTIFACTS.get("model")
    if modelo is None:
        raise HTTPException(
            status_code=503,
            detail="El modelo no está cargado. Ejecuta 'python train.py' y reinicia.",
        )
    return modelo


def _predecir(modelo, vuelos: list[dict]) -> list[RespuestaPrediccion]:
    """Canoniza las categóricas, ejecuta el pipeline y arma las respuestas."""
    registros = [canonizar_registro(v) for v in vuelos]
    df = pd.DataFrame(registros)[FEATURES]  # orden de columnas exacto del contrato
    try:
        preds = modelo.predict(df)
        probas = modelo.predict_proba(df)
    except Exception:
        logger.exception("Fallo al generar la predicción")
        raise HTTPException(status_code=500, detail="Error al generar la predicción")

    ts = datetime.now(timezone.utc).isoformat()
    version = _model_version()
    salida: list[RespuestaPrediccion] = []
    for pred, proba in zip(preds, probas):
        pred = int(pred)
        salida.append(
            RespuestaPrediccion(
                prediccion=pred,
                etiqueta="atraso >= 15 min" if pred == 1 else "a tiempo",
                probabilidad=round(float(proba.max()), 4),
                probabilidad_atraso=round(float(proba[1]), 4),
                model_version=version,
                timestamp=ts,
            )
        )
    return salida


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=RespuestaSalud, tags=["infra"])
def health():
    """Estado del servicio y confirmación de que el modelo está en memoria."""
    return RespuestaSalud(status="ok", model_loaded="model" in ARTIFACTS)


@app.get("/model-info", response_model=RespuestaModelInfo, tags=["infra"])
def model_info():
    """Metadatos del modelo entrenado."""
    _requiere_modelo()
    meta = ARTIFACTS.get("metadata", {})
    return RespuestaModelInfo(
        model_type=meta.get("model_type", "desconocido"),
        task=meta.get("task", "desconocido"),
        target=meta.get("target", "desconocido"),
        features=meta.get("features", []),
        metrics=meta.get("metrics", {}),
        sklearn_version=meta.get("sklearn_version", "desconocida"),
        model_version=meta.get("model_version", "desconocida"),
    )


@app.post("/predict", response_model=RespuestaPrediccion, tags=["inferencia"])
def predict(vuelo: Vuelo):
    """Predice el atraso para un solo vuelo."""
    modelo = _requiere_modelo()
    return _predecir(modelo, [vuelo.model_dump()])[0]


@app.post("/predict-batch", response_model=RespuestaLote, tags=["inferencia"])
def predict_batch(lote: LoteVuelos):
    """Predice el atraso para una lista de vuelos (mismo orden de entrada)."""
    modelo = _requiere_modelo()
    salida = _predecir(modelo, [v.model_dump() for v in lote.items])
    return RespuestaLote(n=len(salida), predicciones=salida)
