"""
app/schemas.py — Contrato de entrada y salida de la API (Pydantic v2).

Define el esquema de un vuelo (los 9 predictores conocidos ANTES de la salida
programada) y la forma de las respuestas. Las restricciones producen un 422
automático ante entradas mal formadas, sin validación manual en el endpoint.

Los códigos se aceptan en forma "plana" (AIRLINE='AA', ORIGIN_AIRPORT='LAX'):
la API los canoniza internamente a la forma del entrenamiento ('IATA:LAX').
"""

from typing import List

from pydantic import BaseModel, Field


class Vuelo(BaseModel):
    """Un vuelo a evaluar. Solo variables conocidas antes de la salida."""

    # --- Categóricas ---
    AIRLINE: str = Field(
        ..., pattern=r"^[A-Za-z0-9]{2}$",
        description="Código IATA de la aerolínea (2 caracteres). Ej: 'AA'.",
    )
    ORIGIN_AIRPORT: str = Field(
        ..., pattern=r"^(?:IATA:)?[A-Za-z]{3}$|^(?:BTS:)?\d{5}$",
        description="Aeropuerto de origen: código IATA de 3 letras o código BTS de 5 dígitos.",
    )
    DESTINATION_AIRPORT: str = Field(
        ..., pattern=r"^(?:IATA:)?[A-Za-z]{3}$|^(?:BTS:)?\d{5}$",
        description="Aeropuerto de destino: código IATA de 3 letras o código BTS de 5 dígitos.",
    )

    # --- Numéricas (calendario y plan de vuelo) ---
    MONTH: int = Field(..., ge=1, le=12, description="Mes (1-12).")
    DAY: int = Field(..., ge=1, le=31, description="Día del mes (1-31).")
    DAY_OF_WEEK: int = Field(..., ge=1, le=7, description="Día de la semana (1=lunes … 7=domingo).")
    SCHEDULED_DEPARTURE_MIN: int = Field(
        ..., ge=0, le=1440,
        description="Salida programada en minutos desde medianoche (0-1440).",
    )
    SCHEDULED_TIME: float = Field(..., gt=0, le=1000, description="Duración programada en minutos.")
    DISTANCE: float = Field(..., gt=0, le=6000, description="Distancia de la ruta en millas.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class LoteVuelos(BaseModel):
    """Lista de vuelos para predicción por lote."""

    items: List[Vuelo] = Field(
        ..., min_length=1, max_length=5000,
        description="Lista de vuelos a evaluar (1 a 5000).",
    )


class RespuestaPrediccion(BaseModel):
    """Respuesta de una predicción individual."""

    prediccion: int = Field(..., description="1 = atraso >= 15 min, 0 = a tiempo.")
    etiqueta: str = Field(..., description="Etiqueta legible de la predicción.")
    probabilidad: float = Field(..., description="Probabilidad de la clase predicha.")
    probabilidad_atraso: float = Field(..., description="Probabilidad de atraso >= 15 min (clase 1).")
    model_version: str = Field(..., description="Versión del modelo usado.")
    timestamp: str = Field(..., description="Marca de tiempo UTC de la predicción.")

    model_config = {"protected_namespaces": ()}


class RespuestaLote(BaseModel):
    n: int = Field(..., description="Número de vuelos procesados.")
    predicciones: List[RespuestaPrediccion]


class RespuestaSalud(BaseModel):
    status: str
    model_loaded: bool

    model_config = {"protected_namespaces": ()}


class RespuestaModelInfo(BaseModel):
    model_type: str
    task: str
    target: str
    features: List[str]
    metrics: dict
    sklearn_version: str
    model_version: str

    model_config = {"protected_namespaces": ()}
