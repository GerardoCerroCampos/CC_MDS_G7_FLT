"""
tests/test_api.py — Pruebas automatizadas de la API con TestClient.

Cubren salud, model-info, predicción válida, predicción por lote y dos casos
de entrada inválida que deben devolver 422.

Ejecutar desde la raíz del proyecto:
    pytest -v
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    # Como context manager, dispara el lifespan (donde se carga el modelo).
    with TestClient(app) as c:
        yield c


VUELO_VALIDO = {
    "AIRLINE": "AA",
    "ORIGIN_AIRPORT": "LAX",
    "DESTINATION_AIRPORT": "JFK",
    "MONTH": 7,
    "DAY": 15,
    "DAY_OF_WEEK": 3,
    "SCHEDULED_DEPARTURE_MIN": 1080,
    "SCHEDULED_TIME": 320.0,
    "DISTANCE": 2475.0,
}


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True  # requiere model/model.pkl (correr train.py antes)


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["target"] == "ARRIVAL_DELAY_15"
    assert len(body["features"]) == 9
    assert "metrics" in body


def test_predict_valido(client):
    r = client.post("/predict", json=VUELO_VALIDO)
    assert r.status_code == 200
    body = r.json()
    assert body["prediccion"] in (0, 1)
    assert 0.0 <= body["probabilidad"] <= 1.0
    assert 0.0 <= body["probabilidad_atraso"] <= 1.0
    assert "timestamp" in body and "model_version" in body


def test_predict_acepta_codigo_bts(client):
    # Aeropuerto en formato BTS (5 dígitos) también debe ser válido.
    vuelo = {**VUELO_VALIDO, "ORIGIN_AIRPORT": "12345"}
    r = client.post("/predict", json=vuelo)
    assert r.status_code == 200


def test_predict_batch(client):
    payload = {"items": [VUELO_VALIDO, {**VUELO_VALIDO, "AIRLINE": "WN", "DISTANCE": 500.0}]}
    r = client.post("/predict-batch", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 2
    assert len(body["predicciones"]) == 2


def test_predict_entrada_invalida_422(client):
    # MONTH fuera de rango y AIRLINE con formato inválido => 422
    invalido = {**VUELO_VALIDO, "MONTH": 13, "AIRLINE": "AMERICAN"}
    r = client.post("/predict", json=invalido)
    assert r.status_code == 422


def test_predict_campo_faltante_422(client):
    incompleto = dict(VUELO_VALIDO)
    del incompleto["DISTANCE"]
    r = client.post("/predict", json=incompleto)
    assert r.status_code == 422
