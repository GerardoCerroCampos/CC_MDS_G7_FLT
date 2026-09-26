# Evidencia de ejecución local

Servicio levantado con:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Log de arranque (el modelo se carga UNA sola vez, en el lifespan):

```
INFO:     Waiting for application startup.
INFO:api-vuelos:Modelo y metadatos cargados correctamente.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

## 1) GET /health

```json
{"status":"ok","model_loaded":true}
```

## 2) GET /model-info

```json
{"model_type":"LogisticRegression","task":"binary_classification",
 "target":"ARRIVAL_DELAY_15",
 "features":["AIRLINE","ORIGIN_AIRPORT","DESTINATION_AIRPORT","MONTH","DAY",
             "DAY_OF_WEEK","SCHEDULED_DEPARTURE_MIN","SCHEDULED_TIME","DISTANCE"],
 "metrics":{"accuracy":0.584,"f1_macro":0.576,"precision_pos":0.4618,
            "recall_pos":0.5897,"roc_auc":0.6091},
 "sklearn_version":"1.8.0","model_version":"1.0.0"}
```

> Las métricas mostradas corresponden al **dataset sintético de respaldo**. Al
> reentrenar con `data/train.csv` y `data/test.csv` (salidas del EDA sobre los
> datos reales), estas métricas se recalculan y son las que deben reportarse.

## 3) POST /predict — predicción exitosa

Petición (se aceptan códigos IATA "planos"; la API los canoniza a IATA:LAX):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"AIRLINE":"AA","ORIGIN_AIRPORT":"LAX","DESTINATION_AIRPORT":"JFK",
       "MONTH":7,"DAY":15,"DAY_OF_WEEK":3,"SCHEDULED_DEPARTURE_MIN":1080,
       "SCHEDULED_TIME":320.0,"DISTANCE":2475.0}'
```

Respuesta:

```json
{"prediccion":1,"etiqueta":"atraso >= 15 min","probabilidad":0.5841,
 "probabilidad_atraso":0.5841,"model_version":"1.0.0",
 "timestamp":"2026-09-26T00:58:05Z"}
```

## 4) POST /predict-batch — predicción por lote (mismo orden de entrada)

```json
{"n":2,"predicciones":[
  {"prediccion":1,"etiqueta":"atraso >= 15 min","probabilidad_atraso":0.5841, "...":"..."},
  {"prediccion":0,"etiqueta":"a tiempo","probabilidad_atraso":0.4823, "...":"..."}
]}
```

## 5) POST /predict — entrada inválida => 422

Petición (aerolínea con formato inválido y MONTH fuera de rango):

```json
{"detail":[
  {"type":"string_pattern_mismatch","loc":["body","AIRLINE"],
   "msg":"String should match pattern '^[A-Za-z0-9]{2}$'"},
  {"type":"less_than_equal","loc":["body","MONTH"],
   "msg":"Input should be less than or equal to 12"}
]}
```

## 6) GET /docs

`HTTP 200` — Swagger UI navegable.

> Pendiente por el equipo: agregar la captura de pantalla de /docs abierta en el
> navegador (guardarla como `docs/docs_swagger.png`).

## Carga del artefacto en un proceso independiente

```
Carga OK en proceso independiente.
Registro canonizado -> IATA:LAX IATA:JFK
Predicción: 1 | P(atraso)= 0.5841
```

---

## Flujo de entrenamiento y evaluación (Paso 2)

`train.py` está integrado con el EDA. Verificación de la ruta que lee la
partición del EDA (`contrato_datos.json` + `data/train.csv` + `data/test.csv`):

```
[contrato] Leído de contrato_datos.json
[datos] Partición del EDA: train.csv + test.csv
[datos] train=80,000 | test=20,000 | prevalencia atraso=0.444 | origen=eda_csv
[cv] f1_macro=0.6121±0.0049 | roc_auc=0.6547±0.0024 | recall=0.6184±0.0096 | precision=0.5591±0.0046 | balanced_accuracy=0.6141±0.0051
[test] accuracy=0.6145 balanced_accuracy=0.6151 precision_pos=0.5597 recall_pos=0.6209 f1_macro=0.6129 roc_auc=0.6547 pr_auc=0.5868
```

Salidas escritas por `train.py`:
- `model/metadata.json` — features, métricas de CV y test, baseline, matriz de
  confusión, prevalencia e interpretación (lo consume `/model-info`).
- `reportes/evaluacion.json`, `reportes/matriz_confusion.csv`,
  `reportes/classification_report.csv` — evidencia reproducible.

> El `model.pkl` y los `reportes/` incluidos en la entrega se generaron con el
> dataset sintético de respaldo (`data_source: "sintetico"`). Con los CSV reales
> del EDA, el log muestra `origen=eda_csv` y las métricas se recalculan.

---

## Verificación de versiones (coherencia .pkl ↔ requirements.txt ↔ runtime)

El `model.pkl` se generó con **Python 3.13.13** y las versiones **exactas** de
`requirements.txt`. Verificaciones en entorno limpio:

```
Python        3.13.13
numpy         2.3.5   OK
pandas        3.0.1   OK
scipy         1.16.3  OK
scikit-learn  1.8.0   OK   <- versión que serializa/deserializa el .pkl
joblib        1.5.3   OK
threadpoolctl 3.6.0   OK
matplotlib    3.10.8  OK
fastapi       0.115.0 OK
pydantic      2.9.2   OK   (pydantic-core 2.23.4)
uvicorn       0.30.6  OK
pytest        8.3.3   OK
httpx         0.27.2  OK

pip install -r requirements.txt   -> sin errores
Carga del .pkl en proceso limpio  -> OK | pred=1 | P(atraso)=0.5841
pytest                            -> 7 passed
GET /health                       -> {"status":"ok","model_loaded":true}
POST /predict                     -> 200
GET /docs                         -> 200
```

> **Sobre el intérprete.** `runtime.txt` fija `python-3.13.13` en lugar de 3.14.2:
> `pydantic==2.9.2` (pydantic-core 2.23.4) y `fastapi==0.115.0` **no tienen wheels
> para Python 3.14**, por lo que en 3.14.x `pip install -r requirements.txt`
> fallaría en un entorno nuevo. Con 3.13.13 todo instala desde wheels **sin
> cambiar ninguna versión de librería**. Para usar Python 3.14 habría que subir
> `pydantic`/`fastapi` a versiones compatibles.
