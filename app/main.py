"""API de inferencia para el pipeline generado por el entrenamiento de vuelos."""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
import hashlib
import json
import logging

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from app.features import FEATURES, TARGET, canonizar_registro
from app.schemas import Vuelo, RespuestaPrediccion, RespuestaSalud, RespuestaModelInfo

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'model/model.pkl'
METADATA_PATH = BASE_DIR / 'model/metadata.json'
ARTIFACTS: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Publica ambos artefactos solo después de verificar su compatibilidad."""
    ARTIFACTS.clear()
    try:
        meta = json.loads(METADATA_PATH.read_text(encoding='utf-8'))
        if meta['sklearn'] != sklearn.__version__:
            raise ValueError('La versión de scikit-learn difiere de la de entrenamiento.')
        if meta['features'] != FEATURES or meta['target'] != TARGET:
            raise ValueError('Los metadatos no coinciden con las entradas de esta API.')
        if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != meta['model_sha256']:
            raise ValueError('El modelo y sus metadatos no corresponden al mismo artefacto.')
        modelo = joblib.load(MODEL_PATH)
        if not isinstance(modelo, Pipeline) or not isinstance(modelo.named_steps.get('pre'), ColumnTransformer):
            raise ValueError('El artefacto debe contener el pipeline completo.')
        if list(modelo.feature_names_in_) != FEATURES or list(modelo.classes_) != [0, 1]:
            raise ValueError('Variables o clases incompatibles.')
        if meta['estimator'] != type(modelo.named_steps['clf']).__name__:
            raise ValueError('El estimador no coincide con los metadatos.')
        RespuestaModelInfo(
            model_type=meta['estimator'], task='classification', target=meta['target'],
            features=meta['features'], metrics=meta['metrics'], sklearn_version=meta['sklearn'],
            model_version=meta['model_version'], run_id=meta.get('run_id'),
        )
        ARTIFACTS.update(model=modelo, metadata=meta)
        logger.info('Pipeline y metadatos cargados correctamente.')
    except Exception:
        logger.exception('No fue posible cargar un modelo compatible; inferencia deshabilitada.')
    try:
        yield
    finally:
        ARTIFACTS.clear()


app = FastAPI(
    title='API de atraso de vuelos', version='1.0.0', lifespan=lifespan,
    description='Predice atraso de llegada >= 15 minutos para vuelos que completan su ruta. '
                'Usa información disponible antes de la salida programada.',
)


def _requiere_modelo():
    if 'model' not in ARTIFACTS or 'metadata' not in ARTIFACTS:
        raise HTTPException(503, 'Modelo no disponible. Revisa los artefactos y reinicia el servicio.')
    return ARTIFACTS['model']


def _predecir(modelo, vuelos: list[dict]) -> list[RespuestaPrediccion]:
    try:
        datos = pd.DataFrame([canonizar_registro(v) for v in vuelos], columns=FEATURES)
        predicciones = modelo.predict(datos)
        probabilidades = modelo.predict_proba(datos)
        clases = list(modelo.classes_)
        if probabilidades.shape != (len(vuelos), len(clases)) or len(predicciones) != len(vuelos):
            raise ValueError('Dimensiones de salida inválidas.')
        if not np.isfinite(probabilidades).all():
            raise ValueError('Probabilidades no finitas.')
        fecha = datetime.now(timezone.utc)
        meta = ARTIFACTS['metadata']
        salida = []
        for pred, proba in zip(predicciones, probabilidades):
            pred = int(pred)
            salida.append(RespuestaPrediccion(
                prediccion=pred,
                etiqueta='atraso >= 15 min' if pred == 1 else 'atraso menor de 15 min',
                probabilidad=round(float(proba[clases.index(pred)]), 4),
                probabilidad_atraso=round(float(proba[clases.index(1)]), 4),
                model_version=meta['model_version'], run_id=meta.get('run_id'), timestamp=fecha,
            ))
        return salida
    except Exception:
        logger.exception('Fallo interno al generar la predicción.')
        raise HTTPException(500, 'Error al generar la predicción') from None


@app.get('/', include_in_schema=False)
def root():
    return RedirectResponse('/docs')


@app.get('/health', response_model=RespuestaSalud, tags=['infra'])
def health():
    cargado = 'model' in ARTIFACTS and 'metadata' in ARTIFACTS
    return RespuestaSalud(status='ok' if cargado else 'degraded', model_loaded=cargado)


@app.get('/model-info', response_model=RespuestaModelInfo, tags=['infra'])
def model_info():
    _requiere_modelo()
    meta = ARTIFACTS['metadata']
    return RespuestaModelInfo(
        model_type=meta['estimator'], task='classification', target=meta['target'],
        features=meta['features'], metrics=meta['metrics'], sklearn_version=meta['sklearn'],
        model_version=meta['model_version'], run_id=meta.get('run_id'),
    )


@app.post('/predict', response_model=RespuestaPrediccion, tags=['inferencia'])
def predict(vuelo: Vuelo):
    return _predecir(_requiere_modelo(), [vuelo.model_dump()])[0]


@app.post('/predict-batch', response_model=list[RespuestaPrediccion], tags=['inferencia'])
def predict_batch(vuelos: Annotated[list[Vuelo], Body(min_length=1, max_length=5000)]):
    """Recibe un arreglo JSON y devuelve las predicciones en el mismo orden."""
    return _predecir(_requiere_modelo(), [v.model_dump() for v in vuelos])
