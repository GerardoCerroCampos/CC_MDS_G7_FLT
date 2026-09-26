# Evidencia automatizada de API

Ejecución UTC: 20260926T152212073522Z

Resultado: **CORRECTO**

## Pytest

Python: 3.13.13. Código de salida: 0.

```text
......................                                                   [100%]
============================== warnings summary ===============================
.venv\Lib\site-packages\starlette\testclient.py:40
  C:\Users\sanbo\Desktop\Cloud Computing\Tarea CC\CC_MDS_G7_FLT\.venv\Lib\site-packages\starlette\testclient.py:40: DeprecationWarning: The anyio.abc.BlockingPortal alias is deprecated, use anyio.from_thread.BlockingPortal instead.
    _PortalFactoryType = typing.Callable[[], typing.ContextManager[anyio.abc.BlockingPortal]]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
- Registro guardado en C:\Users\sanbo\Desktop\Cloud Computing\Tarea CC\CC_MDS_G7_FLT\docs\api\pruebas\20260926T152219262560Z -
22 passed, 1 warning in 6.48s
```

## Llamadas HTTP reales

### Individual

`POST http://localhost:8000/predict`

HTTP esperado: 200. Obtenido: 200.

Solicitud:

```json
{
  "AIRLINE": "AA",
  "ORIGIN_AIRPORT": "LAX",
  "DESTINATION_AIRPORT": "JFK",
  "MONTH": 7,
  "DAY": 15,
  "DAY_OF_WEEK": 3,
  "SCHEDULED_DEPARTURE_MIN": 1080,
  "SCHEDULED_TIME": 320.0,
  "DISTANCE": 2475.0
}
```

Respuesta:

```json
{
  "prediccion": 1,
  "etiqueta": "atraso >= 15 min",
  "probabilidad": 0.5421,
  "probabilidad_atraso": 0.5421,
  "model_version": "1.0.0",
  "run_id": "20260926T145348857218Z",
  "timestamp": "2026-09-26T15:22:22.091670Z"
}
```

### Lote

`POST http://localhost:8000/predict-batch`

HTTP esperado: 200. Obtenido: 200.

Solicitud:

```json
[
  {
    "AIRLINE": "AA",
    "ORIGIN_AIRPORT": "LAX",
    "DESTINATION_AIRPORT": "JFK",
    "MONTH": 7,
    "DAY": 15,
    "DAY_OF_WEEK": 3,
    "SCHEDULED_DEPARTURE_MIN": 1080,
    "SCHEDULED_TIME": 320.0,
    "DISTANCE": 2475.0
  },
  {
    "AIRLINE": "DL",
    "ORIGIN_AIRPORT": "LAX",
    "DESTINATION_AIRPORT": "JFK",
    "MONTH": 7,
    "DAY": 15,
    "DAY_OF_WEEK": 3,
    "SCHEDULED_DEPARTURE_MIN": 1080,
    "SCHEDULED_TIME": 320.0,
    "DISTANCE": 2475.0
  }
]
```

Respuesta:

```json
[
  {
    "prediccion": 1,
    "etiqueta": "atraso >= 15 min",
    "probabilidad": 0.5421,
    "probabilidad_atraso": 0.5421,
    "model_version": "1.0.0",
    "run_id": "20260926T145348857218Z",
    "timestamp": "2026-09-26T15:22:24.262853Z"
  },
  {
    "prediccion": 1,
    "etiqueta": "atraso >= 15 min",
    "probabilidad": 0.5095,
    "probabilidad_atraso": 0.5095,
    "model_version": "1.0.0",
    "run_id": "20260926T145348857218Z",
    "timestamp": "2026-09-26T15:22:24.262853Z"
  }
]
```

### Entrada inválida

`POST http://localhost:8000/predict`

HTTP esperado: 422. Obtenido: 422.

Solicitud:

```json
{
  "AIRLINE": "AA",
  "ORIGIN_AIRPORT": "LAX",
  "DESTINATION_AIRPORT": "JFK",
  "MONTH": 13,
  "DAY": 15,
  "DAY_OF_WEEK": 3,
  "SCHEDULED_DEPARTURE_MIN": 1080,
  "SCHEDULED_TIME": 320.0,
  "DISTANCE": 2475.0
}
```

Respuesta:

```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": [
        "body",
        "MONTH"
      ],
      "msg": "Input should be less than or equal to 12",
      "input": 13,
      "ctx": {
        "le": 12
      }
    }
  ]
}
```

## Alcance

Pytest importa la API del repositorio; las llamadas HTTP consultan el servicio indicado por --url. Inicia ese servicio desde este mismo proyecto para que ambas comprobaciones correspondan a tu entrega. Este registro no toma una captura de pantalla de Swagger; esa evidencia se guarda por separado.