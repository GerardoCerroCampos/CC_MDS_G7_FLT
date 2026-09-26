"""
app/features.py — Contrato de datos y canonicalización de entradas.

Centraliza la definición de los 9 predictores del EDA y la función que
transforma un código de aeropuerto/aerolínea "plano" a la forma canónica con
la que se entrena el modelo (misma lógica que la limpieza del EDA). Al usar la
misma función en train.py y en la API, no hay training–serving skew.

Contrato (según EDA.ipynb, celda de "contrato_datos"):
  - Población: vuelos no cancelados ni desviados, con atraso de llegada observado.
  - Momento de predicción: antes de la salida programada.
  - Etiqueta: ARRIVAL_DELAY_15 = 1 si el atraso de llegada es >= 15 min, si no 0.
"""

import re

CATEGORICAS = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
NUMERICAS = [
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "SCHEDULED_DEPARTURE_MIN",
    "SCHEDULED_TIME",
    "DISTANCE",
]
FEATURES = CATEGORICAS + NUMERICAS
TARGET = "ARRIVAL_DELAY_15"

_RE_IATA = re.compile(r"^[A-Z]{3}$")
_RE_BTS = re.compile(r"^\d{5}$")


def canonizar_aeropuerto(codigo: str) -> str:
    """Normaliza un código de aeropuerto a la forma usada en el entrenamiento.

    - '  lax ' -> 'IATA:LAX'   (código IATA de 3 letras)
    - '12345'  -> 'BTS:12345'  (código numérico de 5 dígitos, BTS)
    - Si ya viene con prefijo 'IATA:'/'BTS:', se respeta.
    Idéntico a la regla de limpieza del EDA para no introducir skew.
    """
    c = str(codigo).strip().upper()
    if c.startswith("IATA:") or c.startswith("BTS:"):
        return c
    if _RE_BTS.match(c):
        return "BTS:" + c
    if _RE_IATA.match(c):
        return "IATA:" + c
    # Valor no reconocido: se devuelve tal cual; OneHotEncoder(handle_unknown=
    # 'ignore') lo tratará como categoría desconocida sin fallar.
    return c


def canonizar_aerolinea(codigo: str) -> str:
    """Normaliza el código de aerolínea (mayúsculas, sin espacios)."""
    return str(codigo).strip().upper()


def canonizar_registro(d: dict) -> dict:
    """Aplica la canonicalización de categóricas sobre un dict de features."""
    out = dict(d)
    out["AIRLINE"] = canonizar_aerolinea(out["AIRLINE"])
    out["ORIGIN_AIRPORT"] = canonizar_aeropuerto(out["ORIGIN_AIRPORT"])
    out["DESTINATION_AIRPORT"] = canonizar_aeropuerto(out["DESTINATION_AIRPORT"])
    return out
