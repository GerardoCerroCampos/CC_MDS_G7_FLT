# Predicción de atraso de vuelos — API con FastAPI

Proyecto de Cloud Computing, Diploma en Data Science, Universidad Adolfo Ibáñez.
El servicio predice si un vuelo llegará con **15 minutos o más de atraso** (`ARRIVAL_DELAY_15`). El modelo utiliza información disponible antes de la salida programada y se aplica a vuelos no cancelados ni desviados, con atraso de llegada observado. No estima cancelaciones ni el riesgo total de interrupción de un vuelo.

La entrega incluye EDA, entrenamiento reproducible, pipeline serializado, API, pruebas automatizadas y evidencia de llamadas a localhost.

## 1. Datos y preparación

Fuente pública: [2015 Flight Delays and Cancellations, USDOT en Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays). Las instrucciones de descarga y la identificación del archivo están en [data/README.md](data/README.md).

El EDA procesa 5.819.079 filas y conserva 5.714.008 vuelos. Excluye 89.884 cancelados y 15.187 desviados. Revisa nulos, duplicados, fechas, formatos y rangos; conserva atrasos negativos y extremos válidos. Los códigos de aeropuerto de tres letras se representan como `IATA:XXX` y los numéricos como `BTS:xxxxx`, sin inventar equivalencias entre ambos sistemas.

Se selecciona una muestra aleatoria de **10.000 vuelos**, con semilla 42. El EDA genera una partición estratificada de **8.000 para entrenamiento y 2.000 para prueba**. El análisis detallado utiliza entrenamiento; no se ajustan transformaciones aprendidas sobre test.

Entradas, en el orden utilizado por el modelo:

| Variable | Tipo y significado |
|---|---|
| AIRLINE | Código de aerolínea de dos caracteres |
| ORIGIN_AIRPORT | Aeropuerto de origen, IATA o BTS |
| DESTINATION_AIRPORT | Aeropuerto de destino, IATA o BTS |
| MONTH | Entero de 1 a 12 |
| DAY | Día del mes |
| DAY_OF_WEEK | Entero de 1 a 7; 1 es lunes |
| SCHEDULED_DEPARTURE_MIN | Minutos desde medianoche, de 0 a 1440; no HHMM |
| SCHEDULED_TIME | Duración programada positiva, en minutos |
| DISTANCE | Distancia positiva, en millas |

`ARRIVAL_DELAY` permite construir la etiqueta, pero no es predictor. `SOURCE_ROW` y `FLIGHT_DATE` también se reservan para auditoría.

## 2. Estructura del repositorio

```text
app/                         API y esquemas de entrada/salida
data/README.md               Origen y preparación de los datos
docs/eda/                    Reportes, gráficos y resumen del EDA
docs/entrenamiento/reportes/ Balance, métricas y matriz de confusión
docs/api/ejecuciones/        Pytest y llamadas HTTP reales
docs/api/pruebas/            Registro automático de pytest
model/model.pkl             Pipeline completo
model/metadata.json         Variables, versiones, métricas y trazabilidad
notebooks/EDA.ipynb          Preparación y exploración
notebooks/train.ipynb        Entrenamiento y serialización
scripts/registrar_pruebas.py Registro automático de evidencia
tests/test_api.py           Pruebas de la API
tests/conftest.py           Configuración y registro de pytest
requirements.txt            Dependencias fijadas
runtime.txt                 Python utilizado
Procfile                    Comando de arranque portable
```

Los CSV grandes, los entornos virtuales, las carpetas temporales y el historial local de modelos están excluidos de Git. El entrenamiento se realiza en notebook; no requiere `train.py` ni `configuracion_datos.json`.

## 3. Instalación en Windows / PowerShell

Requisitos: Git y **Python 3.13.13**, la versión declarada por el modelo entregado y runtime.txt. Comprobarla antes de crear el entorno:

```powershell
py -3.13 --version
git clone https://github.com/GerardoCerroCampos/CC_MDS_G7_FLT.git
cd CC_MDS_G7_FLT
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

El primer comando debe mostrar Python 3.13.13. No cambiar únicamente runtime.txt para declarar otra versión: si se cambia el entorno, se debe volver a entrenar y probar. Los comandos siguientes usan el intérprete del entorno explícitamente; no requieren activar PowerShell con Activate.ps1.

## 4. Reproducir EDA y entrenamiento

Para utilizar el modelo incluido en Git se puede pasar directamente a la sección 6. Para reproducir su preparación y entrenamiento:

1. Descargar y descomprimir `flights.csv` según [data/README.md](data/README.md). Guardarlo en `data/raw/flights.csv`.
2. Iniciar JupyterLab desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

3. Abrir `notebooks/EDA.ipynb`, seleccionar el kernel del entorno y usar **Restart Kernel and Run All Cells**. Esperar a que finalice sin errores y guardar el notebook.
4. Abrir `notebooks/train.ipynb` y repetir el procedimiento. Entrena con los CSV que produjo el EDA.

Para confirmar el kernel, ejecutar `import sys; print(sys.executable); print(sys.version)` en una celda temporal: debe corresponder al `.venv` del proyecto. El EDA lee el archivo completo por bloques de 200.000 filas. Requiere espacio para el original (aproximadamente 592 MB), el limpio (aproximadamente 408 MB) y los demás resultados.

## 5. Entrenamiento y resultados

El modelo es **RandomForestClassifier**, con 200 árboles, profundidad máxima 12, un máximo de 256 hojas, mínimo de ocho observaciones por hoja, ponderación balanceada de clases y semilla 42. Los parámetros se fijan antes de evaluar test; esta versión no hace búsqueda de hiperparámetros ni validación cruzada.

Un único `Pipeline` contiene `ColumnTransformer` —OneHotEncoder con categorías desconocidas ignoradas y StandardScaler— y Random Forest. Todo se ajusta únicamente con train. StandardScaler explicita el tratamiento numérico, aunque el bosque no lo necesita para funcionar. La API adapta los códigos al formato del EDA; la codificación y el escalado siguen dentro del artefacto.

Resultados de la ejecución registrada en [model/metadata.json](model/metadata.json):

| Métrica de test | Valor |
|---|---:|
| F1 macro | 0,5184 |
| Recall de atrasos | 0,4718 |
| Precisión de alertas | 0,2310 |
| Average precision | 0,2403 |
| ROC AUC | 0,5876 |
| Accuracy | 0,6085 |

F1 macro considera ambas clases con igual peso. Recall mide la detección de atrasos y precisión mide cuántas alertas son correctas. Se reportan ambas para mostrar las omisiones y falsas alarmas. Average precision considera el ordenamiento de probabilidades. Accuracy se interpreta con cautela: predecir siempre sin atraso alcanza 0,8135 de accuracy, pero no detecta ningún atraso. El modelo no se presenta como una solución de alta precisión.

El pipeline se guarda con joblib en `model/model.pkl`: **362.216 bytes**, menos de 100 MB. El notebook lo carga en un proceso Python nuevo y compara predicciones y probabilidades, incluyendo una categoría desconocida. Los metadatos registran sklearn 1.8.0, Python 3.13.13, variables ordenadas, parámetros, métricas, versión e identificación de ejecución y hashes de datos y modelo.

Los reportes se guardan en `docs/entrenamiento/reportes/`. Cada entrenamiento conserva una copia de los artefactos y del código en `model/versions/<version>/<ejecucion>/`, excluido de Git. Las copias en `model/` corresponden al entrenamiento actual.

Limitaciones: datos de 2015; separación aleatoria dentro del mismo año; posible dependencia entre rutas y fechas; aeropuertos IATA/BTS sin unificar; probabilidades no calibradas. El conjunto de prueba se consultó en ejercicios anteriores y no es una evaluación externa nueva.

## 6. Iniciar la API

Desde la raíz, con el modelo y sus metadatos presentes:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Abrir [Swagger](http://localhost:8000/docs). Mantener esta terminal abierta durante las llamadas HTTP.

| Método y ruta | Función |
|---|---|
| GET /health | Estado y confirmación de modelo cargado |
| GET /model-info | Estimador, variables, métricas, versión sklearn y versión/ejecución del modelo |
| POST /predict | Predicción de un vuelo y probabilidades |
| POST /predict-batch | Lista de vuelos y lista de resultados, en el mismo orden |
| GET /docs | Documentación interactiva |

El modelo se carga una vez en lifespan. Se verifican versión sklearn, variables, clases, estimador y hash antes de publicar modelo y metadatos juntos. Datos inválidos devuelven 422; un fallo interno de inferencia devuelve 500 con mensaje controlado. Si los artefactos no están disponibles o son incompatibles, la inferencia devuelve 503 y health informa `model_loaded=false`.

Los códigos admiten espacios externos, minúsculas y prefijos IATA/BTS. Los números deben ser números JSON; los campos extra se rechazan. La clase 0 significa **atraso menor de 15 minutos**, no necesariamente llegada anticipada o puntual. El lote recibe un arreglo directo, no un objeto con `items`.

Ejemplos en otra terminal PowerShell:

```powershell
$vuelo = @{
    AIRLINE = 'AA'; ORIGIN_AIRPORT = 'LAX'; DESTINATION_AIRPORT = 'JFK'
    MONTH = 7; DAY = 15; DAY_OF_WEEK = 3
    SCHEDULED_DEPARTURE_MIN = 1080; SCHEDULED_TIME = 320.0; DISTANCE = 2475.0
}
Invoke-RestMethod 'http://localhost:8000/health'
Invoke-RestMethod 'http://localhost:8000/model-info'
Invoke-RestMethod 'http://localhost:8000/predict' -Method Post -ContentType 'application/json' -Body ($vuelo | ConvertTo-Json)
$lote = @($vuelo, $vuelo)
Invoke-RestMethod 'http://localhost:8000/predict-batch' -Method Post -ContentType 'application/json' -Body (ConvertTo-Json -InputObject $lote -Depth 5)
```

## 7. Pruebas y evidencia

Las pruebas automatizadas no requieren un servidor Uvicorn activo; TestClient inicia la aplicación durante las pruebas:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_api.py
```

`tests/conftest.py` guarda el resumen, detalles, errores, avisos y duración en `docs/api/pruebas/FECHA/`. También elige una carpeta temporal nueva para evitar conflictos de permisos. Las 22 pruebas cubren metadatos, predicciones iguales al pipeline, orden de lotes, normalización, entradas inválidas, error 500, disponibilidad, carga única y documentación OpenAPI.

Salida registrada con Python 3.13.13 en la ejecución del 26 de septiembre de 2026:

```text
22 passed, 1 warning in 6.48s
```

El aviso es una deprecación de BlockingPortal en Starlette/AnyIO. No ocasionó fallos. Véase la [salida completa de pytest](docs/api/ejecuciones/20260926T152212073522Z/pytest.txt).

Con Uvicorn activo, ejecutar en otra terminal:

```powershell
.\.venv\Scripts\python.exe scripts/registrar_pruebas.py
```

Se guardan pytest, solicitudes, respuestas y códigos HTTP en `docs/api/ejecuciones/FECHA/`. La ejecución publicada acredita **individual 200, lote 200 e inválida 422**:

- [Reporte de llamadas y pruebas](docs/api/ejecuciones/20260926T152212073522Z/reporte.md).
- [Solicitudes y respuestas JSON](docs/api/ejecuciones/20260926T152212073522Z/llamadas.json).
- [Resumen de ejecución](docs/api/ejecuciones/20260926T152212073522Z/resumen.json).

**Evidencia pendiente:** guardar y versionar una captura real de Swagger funcionando, por ejemplo `docs/api/swagger_localhost.png`, y enlazarla aquí después de incorporarla. Los registros anteriores no sustituyen esa captura. Si se actualizan código o modelo, repetir las pruebas y actualizar los enlaces y el resultado de este README.

## 8. Entorno y alcance

requirements.txt fija las dependencias. runtime.txt declara `python-3.13.13`. Procfile contiene:

```text
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

La evidencia corresponde a ejecución local. Esta entrega no acredita un despliegue en nube. Antes de entregar, realizar una instalación en una copia nueva con entorno virtual nuevo, comprobar carga del modelo, ejecutar pruebas e iniciar el servicio. El historial de Git debe incluir las contribuciones de todos los integrantes; si el repositorio es privado, se debe otorgar acceso al equipo docente.
