# Datos: 2015 Flight Delays and Cancellations

Fuente pública: [USDOT, 2015 Flight Delays and Cancellations, en Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays).

## Archivo necesario

Descargar el dataset, descomprimirlo y guardar **flights.csv** en:

```text
data/raw/flights.csv
```

Puede ser necesario iniciar sesión en Kaggle para descargar. El EDA actual utiliza únicamente flights.csv; no requiere airlines.csv ni airports.csv. También permite indicar una ruta externa en RUTA_CSV o mediante la variable de entorno FLIGHTS_CSV.

El original procesado contiene 5.819.079 registros y 31 columnas y ocupa 592.406.591 bytes. Su SHA-256 registrado es:

```text
afa4e42188b113dd69427a3fbd00a4b6979c6235c866f963c2899d18a3403a83
```

Para comparar la descarga en PowerShell:

```powershell
Get-FileHash -LiteralPath .\data\raw\flights.csv -Algorithm SHA256
```

Una huella diferente significa que los bytes del archivo no son idénticos a los usados en la ejecución documentada. Revisar la versión antes de comparar resultados.

## Problema supervisado

El objetivo es **ARRIVAL_DELAY_15**, con valor 1 cuando ARRIVAL_DELAY >= 15 minutos y 0 en caso contrario. La población comprende vuelos no cancelados ni desviados, con atraso de llegada observado. CANCELLED se utiliza como filtro, no como variable objetivo ni predictor.

Se utilizan nueve predictores: AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT, MONTH, DAY, DAY_OF_WEEK, SCHEDULED_DEPARTURE_MIN, SCHEDULED_TIME y DISTANCE. SOURCE_ROW, FLIGHT_DATE y ARRIVAL_DELAY se conservan para auditoría y no entran al modelo.

## Archivos derivados

Ejecutar notebooks/EDA.ipynb completo desde Jupyter con el entorno del proyecto. Genera:

| Archivo | Contenido |
|---|---|
| data/flights_limpio.csv | Base completa limpia: 5.714.008 vuelos en la ejecución documentada |
| data/flights_muestra.csv | Muestra aleatoria de 10.000 vuelos, semilla 42 |
| data/train.csv | 8.000 vuelos para entrenamiento |
| data/test.csv | 2.000 vuelos para prueba |

La separación es estratificada y reproducible. El entrenamiento lee train.csv y test.csv; no genera datos sintéticos de respaldo. Si faltan, primero debe ejecutarse el EDA.

El original se conserva. Reejecutar el EDA reemplaza los CSV derivados. Las tablas y los gráficos se guardan en docs/eda/. El CSV limpio completo ocupa aproximadamente 408 MB. Los CSV se mantienen fuera de Git mediante .gitignore; no deben subirse al repositorio.
