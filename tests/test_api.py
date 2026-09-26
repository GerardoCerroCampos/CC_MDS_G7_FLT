"""Pruebas HTTP con el pipeline real y fallos controlados del modelo."""
import json
from unittest.mock import patch
import pandas as pd
import pytest
from fastapi.testclient import TestClient
import app.main as api
from app.features import FEATURES, canonizar_registro

VUELO = {
    'AIRLINE': 'AA', 'ORIGIN_AIRPORT': 'LAX', 'DESTINATION_AIRPORT': 'JFK',
    'MONTH': 7, 'DAY': 15, 'DAY_OF_WEEK': 3,
    'SCHEDULED_DEPARTURE_MIN': 1080, 'SCHEDULED_TIME': 320.0, 'DISTANCE': 2475.0,
}


@pytest.fixture
def client():
    with TestClient(api.app) as client:
        assert client.get('/health').json()['model_loaded'] is True
        yield client


def test_health_y_metadatos(client):
    assert client.get('/health').json() == {'status': 'ok', 'model_loaded': True}
    r = client.get('/model-info')
    assert r.status_code == 200
    m = api.ARTIFACTS['metadata']
    assert r.json()['model_type'] == m['estimator']
    assert r.json()['sklearn_version'] == m['sklearn']
    assert r.json()['features'] == FEATURES
    assert r.json()['metrics'] == m['metrics']
    assert r.json()['run_id'] == m['run_id']


def test_predict_igual_pipeline(client):
    r = client.post('/predict', json=VUELO)
    assert r.status_code == 200
    df = pd.DataFrame([canonizar_registro(VUELO)], columns=FEATURES)
    model = api.ARTIFACTS['model']
    pred = int(model.predict(df)[0])
    proba = model.predict_proba(df)[0]
    body = r.json()
    assert body['prediccion'] == pred
    assert body['probabilidad_atraso'] == round(float(proba[1]), 4)
    assert body['probabilidad'] == round(float(proba[pred]), 4)
    assert body['timestamp'] and body['model_version']
    if pred == 0:
        assert body['etiqueta'] == 'atraso menor de 15 min'


def test_lote_conserva_orden(client):
    vuelos = [VUELO, dict(VUELO, AIRLINE='DL', SCHEDULED_DEPARTURE_MIN=120),
              dict(VUELO, ORIGIN_AIRPORT='ZZZ')]
    r = client.post('/predict-batch', json=vuelos)
    assert r.status_code == 200
    assert isinstance(r.json(), list) and len(r.json()) == len(vuelos)
    for vuelo, resultado in zip(vuelos, r.json()):
        individual = client.post('/predict', json=vuelo).json()
        for campo in ['prediccion', 'probabilidad', 'probabilidad_atraso']:
            assert resultado[campo] == individual[campo]


@pytest.mark.parametrize('cambios', [
    {'MONTH': 13}, {'MONTH': '7'}, {'MONTH': True}, {'DISTANCE': -1},
    {'ARRIVAL_DELAY': 30}, {'MONTH': 4, 'DAY': 31}, {'ORIGIN_AIRPORT': 'INVALIDO'},
])
def test_entrada_invalida_422(client, cambios):
    assert client.post('/predict', json=dict(VUELO, **cambios)).status_code == 422


def test_campo_faltante_422(client):
    vuelo = dict(VUELO)
    del vuelo['DISTANCE']
    assert client.post('/predict', json=vuelo).status_code == 422


@pytest.mark.parametrize('lote', [[], {'items': [VUELO]}, [VUELO] * 5001])
def test_formato_lote_422(client, lote):
    assert client.post('/predict-batch', json=lote).status_code == 422


def test_codigos_normalizados(client):
    original = client.post('/predict', json=VUELO).json()
    normalizado = client.post('/predict', json=dict(VUELO, AIRLINE=' aa ',
        ORIGIN_AIRPORT=' iata:lax ', DESTINATION_AIRPORT='jfk')).json()
    assert original['probabilidad_atraso'] == normalizado['probabilidad_atraso']
    for aeropuerto in ['12345', 'BTS:12345', 'ZZZ']:
        assert client.post('/predict', json=dict(VUELO, ORIGIN_AIRPORT=aeropuerto)).status_code == 200


def test_fallo_modelo_500(client, monkeypatch):
    def falla(*args, **kwargs):
        raise RuntimeError('detalle interno que no debe exponerse')
    monkeypatch.setattr(api.ARTIFACTS['model'], 'predict', falla)
    r = client.post('/predict', json=VUELO)
    assert r.status_code == 500
    assert r.json() == {'detail': 'Error al generar la predicción'}


def test_modelo_carga_una_vez():
    with patch.object(api.joblib, 'load', wraps=api.joblib.load) as carga:
        with TestClient(api.app) as c:
            assert c.post('/predict', json=VUELO).status_code == 200
            assert c.post('/predict', json=VUELO).status_code == 200
            assert carga.call_count == 1


@pytest.mark.parametrize('problema', ['ausente', 'features', 'sklearn', 'hash'])
def test_arranque_incompatible_no_publica_modelo(tmp_path, monkeypatch, problema):
    destino = tmp_path / 'metadata.json'
    if problema != 'ausente':
        m = json.loads(api.METADATA_PATH.read_text(encoding='utf-8'))
        if problema == 'features': m['features'] = list(reversed(m['features']))
        if problema == 'sklearn': m['sklearn'] = '0.0.0'
        if problema == 'hash': m['model_sha256'] = 'incorrecto'
        destino.write_text(json.dumps(m), encoding='utf-8')
    monkeypatch.setattr(api, 'METADATA_PATH', destino)
    with TestClient(api.app) as c:
        assert c.get('/health').json() == {'status': 'degraded', 'model_loaded': False}
        assert c.get('/model-info').status_code == 503
        assert c.post('/predict', json=VUELO).status_code == 503
        assert not api.ARTIFACTS


def test_docs_y_contrato_lote(client):
    assert client.get('/docs').status_code == 200
    schema = client.get('/openapi.json').json()
    batch = schema['paths']['/predict-batch']['post']
    assert batch['requestBody']['content']['application/json']['schema']['type'] == 'array'
    assert batch['responses']['200']['content']['application/json']['schema']['type'] == 'array'
