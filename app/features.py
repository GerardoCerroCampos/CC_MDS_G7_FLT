"""Entradas del EDA y adaptación de códigos al formato del entrenamiento.

Train recibe aeropuertos ya normalizados por el EDA. La API acepta esos códigos
o códigos sin prefijo y los adapta antes de llamar al pipeline serializado.
OneHotEncoder y StandardScaler permanecen dentro del modelo; no se replican aquí.
"""
import re

CATEGORICAS = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
NUMERICAS = ['MONTH', 'DAY', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE_MIN',
             'SCHEDULED_TIME', 'DISTANCE']
FEATURES = CATEGORICAS + NUMERICAS
TARGET = 'ARRIVAL_DELAY_15'


def canonizar_aeropuerto(codigo: str) -> str:
    codigo = codigo.strip().upper()
    if re.fullmatch(r'(IATA:[A-Z]{3}|BTS:[0-9]{5})', codigo):
        return codigo
    if re.fullmatch(r'[A-Z]{3}', codigo):
        return 'IATA:' + codigo
    if re.fullmatch(r'[0-9]{5}', codigo):
        return 'BTS:' + codigo
    raise ValueError('Código de aeropuerto inválido.')


def canonizar_registro(registro: dict) -> dict:
    salida = dict(registro)
    salida['AIRLINE'] = salida['AIRLINE'].strip().upper()
    for columna in ['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']:
        salida[columna] = canonizar_aeropuerto(salida[columna])
    return salida
