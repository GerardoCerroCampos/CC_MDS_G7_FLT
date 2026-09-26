# API de inferencia — Atraso de vuelos (FastAPI)

Servicio HTTP que expone un modelo de clasificación que predice si un vuelo
llegará con **15 minutos o más de atraso** (`ARRIVAL_DELAY_15`), usando solo
información conocida **antes de la salida programada**. El modelo se entrena con
scikit-learn y se sirve con FastAPI.

Este repositorio corresponde al **Paso 5 — API con FastAPI** de la tarea.

> **Nota sobre el objetivo.** El `data/README.md` describe la fuente (2015 Flight
> Delays and Cancellations, USDOT). El problema modelado, siguiendo el
> `EDA.ipynb`, es la predicción de **atraso de llegada ≥ 15 min**
> (`ARRIVAL_DELAY_15`) sobre vuelos no cancelados ni desviados — no la predicción
> de cancelación. La API respeta el contrato de datos generado por el EDA.

## Coherencia de versiones (.pkl ↔ requirements.txt ↔ runtime)

Todas las versiones de librerías son **exactamente** las de `requirements.txt`
(scikit-learn 1.8.0, numpy 2.3.5, scipy 1.16.3, pandas 3.0.1, joblib 1.5.3,
threadpoolctl 3.6.0, matplotlib 3.10.8, fastapi 0.115.0, pydantic 2.9.2,
uvicorn 0.30.6, pytest 8.3.3, httpx 0.27.2).

`runtime.txt` fija **Python 3.13.13**. Motivo: `pydantic==2.9.2`
(vía `pydantic-core 2.23.4`) y `fastapi==0.115.0` **no publican wheels para
Python 3.14**, de modo que con Python 3.14.x `pip install -r requirements.txt`
falla en un entorno nuevo (intentaría compilar `pydantic-core` desde código
fuente). Con Python 3.13.13 todo el stack instala desde wheels **sin cambiar
ninguna versión de librería**. Si el equipo requiere Python 3.14, habría que
subir `pydantic`/`fastapi` a versiones con wheels para 3.14.

El `model.pkl` se generó con Python 3.13.13 y estas versiones exactas, y se
verificó que:
- `pip install -r requirements.txt` instala sin errores,
- el `.pkl` carga en un proceso limpio,
- `pytest` pasa completo y el servicio responde (ver `docs/evidencia_local.md`).

## Estructura

```
proyecto/
├── app/
│   ├── __init__.py
│   ├── main.py         # aplicación FastAPI (endpoints, lifespan, errores)
│   ├── schemas.py      # modelos Pydantic (contrato de entrada/salida)
│   └── features.py     # contrato de features + canonicalización de códigos
├── model/
│   ├── model.pkl       # pipeline serializado (lo genera train.py)
│   └── metadata.json   # versiones, features y métricas
├── notebooks/
│   └── EDA.ipynb       # análisis exploratorio y limpieza (genera el contrato)
├── tests/
│   └── test_api.py     # pruebas con TestClient
├── reportes/
│   ├── evaluacion.json          # CV, test, baseline, matriz, interpretación
│   ├── matriz_confusion.csv
│   └── classification_report.csv
├── docs/
│   ├── evidencia_local.md
│   └── salida_pytest.txt
├── data/
│   └── README.md       # origen del dataset (no se versionan los CSV)
├── train.py            # entrenamiento + evaluación + serialización (Paso 2)
├── requirements.txt    # dependencias con versiones fijas
├── runtime.txt         # versión de Python
├── Procfile            # comando de arranque (usa $PORT)
└── .gitignore
```

## Contrato de entrada (9 predictores del EDA)

| Campo                     | Tipo   | Restricción / formato                              |
|---------------------------|--------|----------------------------------------------------|
| `AIRLINE`                 | str    | Código IATA de 2 caracteres (ej. `AA`).            |
| `ORIGIN_AIRPORT`          | str    | IATA de 3 letras (`LAX`) o BTS de 5 dígitos.       |
| `DESTINATION_AIRPORT`     | str    | IATA de 3 letras (`JFK`) o BTS de 5 dígitos.       |
| `MONTH`                   | int    | 1–12                                               |
| `DAY`                     | int    | 1–31                                               |
| `DAY_OF_WEEK`             | int    | 1–7 (1 = lunes)                                    |
| `SCHEDULED_DEPARTURE_MIN` | int    | 0–1440 (minutos desde medianoche)                  |
| `SCHEDULED_TIME`          | float  | > 0 (duración programada en minutos)               |
| `DISTANCE`                | float  | > 0 (millas)                                       |

Los códigos se aceptan en forma "plana" (`LAX`, `AA`) y la API los normaliza a la
forma del entrenamiento (`IATA:LAX`), replicando exactamente la limpieza del EDA
para evitar *training–serving skew* (ver `app/features.py`).

## Puesta en marcha (local)

```bash
# 1) Entorno virtual e instalación
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) (Opcional) Colocar data/train.csv y data/test.csv generados por EDA.ipynb.
#    Si no están, train.py genera un dataset sintético de respaldo.

# 3) Entrenar el modelo (genera model/model.pkl y model/metadata.json)
python train.py

# 4) Levantar el servicio
uvicorn app.main:app --reload --port 8000
```

Documentación interactiva en <http://localhost:8000/docs>.

## Endpoints

| Método | Ruta             | Descripción                                                   |
|--------|------------------|--------------------------------------------------------------|
| GET    | `/health`        | Estado del servicio y confirmación de modelo cargado.        |
| GET    | `/model-info`    | Metadatos: estimador, target, features, métricas y versión.  |
| POST   | `/predict`       | Predicción para un vuelo (incluye probabilidad).             |
| POST   | `/predict-batch` | Predicción para una lista de vuelos (mismo orden).           |
| GET    | `/docs`          | Swagger UI (automática).                                     |

### Ejemplo: predicción individual

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

## Decisiones de diseño (lo que evalúa la rúbrica del Paso 5)

- **Carga única del modelo.** El `.pkl` se carga una sola vez al iniciar la app,
  en el evento `lifespan`, y queda en memoria (`ARTIFACTS`); no se recarga por
  petición.
- **Validación con Pydantic.** El esquema `Vuelo` fija tipos, rangos y patrones
  (`AIRLINE` de 2 caracteres, aeropuertos IATA/BTS, `MONTH` 1–12, etc.). Una
  entrada mal formada produce **422** automáticamente.
- **Manejo de errores.**
  - Campos faltantes o valores fuera de contrato → **422** (Pydantic).
  - Fallo interno del modelo → **500** con mensaje controlado (traza al log, no
    al cliente).
  - Modelo no disponible → **503** con instrucción para entrenarlo.
- **Sin skew.** La canonicalización de categóricas se comparte entre `train.py`
  y la API (`app/features.py`). El resto del preprocesamiento (OneHot + escalado)
  vive dentro del pipeline serializado.
- **Respuesta enriquecida.** Predicción, etiqueta legible, probabilidad,
  probabilidad de atraso, versión del modelo y marca de tiempo (UTC).
- **Rutas relativas.** El modelo se referencia con rutas relativas al proyecto.

## Pruebas

```bash
pytest -v
```

Cubren `/health`, `/model-info`, una predicción válida, aceptación de código BTS,
predicción por lote y dos casos de entrada inválida (422). Salida en
`docs/salida_pytest.txt` (**7 passed**).

## Entrenamiento y evaluación (Paso 2)

`train.py` implementa el flujo completo, integrado con el EDA:

1. **Lee el contrato del EDA** (`contrato_datos.json`): toma de ahí la lista
   autorizada de predictores, categóricas/numéricas, etiqueta y semilla. Si el
   archivo no está, usa la definición equivalente de `app/features.py`. Comprueba
   que ningún predictor sea una columna de auditoría (previene fuga de información).
2. **Usa la partición del EDA** (`data/train.csv` para ajustar, `data/test.csv`
   como hold-out). Si no existen, genera un dataset sintético reproducible.
3. **Entrena el pipeline** solo con train (el preprocesamiento se ajusta dentro
   del pipeline, nunca con test).
4. **Evalúa con metodología defendible:**
   - Validación cruzada estratificada de 5 folds sobre train, con varias métricas.
   - Evaluación final sobre el hold-out reservado por el EDA.
   - Baseline de clase mayoritaria (`DummyClassifier`) como referencia.
   - Métricas apropiadas al desbalance: F1-macro, recall y precisión de la clase
     positiva, ROC-AUC, PR-AUC, exactitud y balanced accuracy.
   - Matriz de confusión, `classification_report` e interpretación textual.
5. **Serializa** el pipeline y escribe `model/metadata.json` más los reportes
   reproducibles en `reportes/`.

```bash
python train.py                      # modelo por defecto (LogisticRegression)
MODELO=random_forest python train.py # estimador alternativo
```

Todas las métricas y la interpretación quedan también en `/model-info` y en
`reportes/evaluacion.json`.

### Reentrenar con los datos reales

1. Ejecutar `EDA.ipynb` con `flights.csv`: genera `contrato_datos.json`,
   `data/train.csv` y `data/test.csv`.
2. Ejecutar `python train.py`: detecta el contrato y la partición y entrena con
   ellos (el log mostrará `origen=eda_csv`).
3. Reiniciar el servicio. `/model-info` mostrará las métricas reales.

> El `model.pkl` y los reportes incluidos se generaron con el **dataset sintético
> de respaldo** (`data_source: "sintetico"`), para que el repo funcione sin la
> descarga de Kaggle. Se recalculan al reentrenar con los CSV del EDA.
