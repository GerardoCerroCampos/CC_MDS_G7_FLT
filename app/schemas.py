"""Esquemas Pydantic: entradas inválidas producen 422 antes de inferir."""
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Vuelo(BaseModel):
    model_config = ConfigDict(strict=True, extra='forbid', allow_inf_nan=False,
        json_schema_extra={'examples': [{
            'AIRLINE': 'AA', 'ORIGIN_AIRPORT': 'LAX', 'DESTINATION_AIRPORT': 'JFK',
            'MONTH': 7, 'DAY': 15, 'DAY_OF_WEEK': 3,
            'SCHEDULED_DEPARTURE_MIN': 1080, 'SCHEDULED_TIME': 320.0, 'DISTANCE': 2475.0,
        }]})
    AIRLINE: str = Field(pattern=r'^[A-Z0-9]{2}$', description='Código de aerolínea de dos caracteres.')
    ORIGIN_AIRPORT: str = Field(pattern=r'^(?:IATA:)?[A-Z]{3}$|^(?:BTS:)?[0-9]{5}$')
    DESTINATION_AIRPORT: str = Field(pattern=r'^(?:IATA:)?[A-Z]{3}$|^(?:BTS:)?[0-9]{5}$')
    MONTH: int = Field(ge=1, le=12)
    DAY: int = Field(ge=1, le=31)
    DAY_OF_WEEK: int = Field(ge=1, le=7, description='1=lunes, 7=domingo.')
    SCHEDULED_DEPARTURE_MIN: int = Field(ge=0, le=1440, description='Minutos desde medianoche, no HHMM.')
    SCHEDULED_TIME: float = Field(gt=0, description='Duración programada en minutos.')
    DISTANCE: float = Field(gt=0, description='Distancia en millas.')

    @field_validator('AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', mode='before')
    @classmethod
    def normalizar_texto(cls, valor):
        return valor.strip().upper() if isinstance(valor, str) else valor

    @model_validator(mode='after')
    def validar_calendario(self):
        # Sin YEAR no se comprueba el día de semana ni si febrero es bisiesto.
        maximos = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if self.DAY > maximos[self.MONTH - 1]:
            raise ValueError('El día no existe en el mes indicado.')
        return self


class RespuestaPrediccion(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prediccion: Literal[0, 1]
    etiqueta: str
    probabilidad: float = Field(ge=0, le=1, description='Probabilidad de la clase predicha.')
    probabilidad_atraso: float = Field(ge=0, le=1)
    model_version: str
    run_id: str | None = None
    timestamp: datetime


class RespuestaSalud(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: Literal['ok', 'degraded']
    model_loaded: bool


class RespuestaModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_type: str
    task: Literal['classification']
    target: str
    features: list[str]
    metrics: dict[str, float]
    sklearn_version: str
    model_version: str
    run_id: str | None = None
