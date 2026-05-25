# Tarea 2 - Pipeline ML E2E

Este repositorio contiene un pipeline de machine learning para el caso CU Venta.

Repositorio:

```text
https://github.com/xfibben/tarea2_UCSP
```

El flujo hace:

- lectura de datos
- preprocesamiento
- entrenamiento con XGBoost y Optuna
- registro del modelo con MLflow
- evaluacion con mes OOT
- monitoreo con PSI
- postprocesamiento TLV
- generacion de replicas para `s3`, `athena` y `onpremise`

## Dataset

La data no se sube al repositorio porque pesa demasiado.

Descargar los archivos CSV desde:

```text
https://drive.google.com/drive/folders/1BbaYLS_Cy5pbvfE6JH3P7KlLdlRf_Cds?usp=drive_link
```

Luego colocar los CSV en:

```text
data/raw/
```

La columna usada para separar el mes OOT es `p_fecinformacion`.

## Instalacion

Crear el entorno e instalar librerias:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En macOS, si XGBoost muestra error con `libomp.dylib`, ejecutar:

```bash
brew install libomp
```

## Como ejecutar

Primero se puede revisar que los CSV esten bien ubicados:

```bash
python main.py inspect-data
```

Para correr todo el pipeline:

```bash
python main.py run-all --validation-codmes 20221201 --n-trials 30
```

Para una prueba rapida se puede usar:

```bash
python main.py run-all --validation-codmes 20221201 --n-trials 1 --max-retrain-attempts 0
```

Para ejecutar solo inferencia despues de entrenar:

```bash
python main.py inference
```

Para abrir el dashboard:

```bash
streamlit run dashboard.py
```

## Archivos generados

El pipeline genera archivos en estas carpetas:

```text
data/processed/
data/postprocessed/output_tlv.csv
data/replica/
artifacts/
mlruns/
```

Estos archivos no se suben a GitHub porque se pueden volver a generar.

## Estructura

```text
.
├── main.py
├── config.py
├── preprocessing.py
├── training.py
├── monitoring.py
├── postprocessing.py
├── dashboard.py
├── requirements.txt
├── data/
└── artifacts/
```

## Validacion local

El pipeline fue probado con los CSV reales del Drive usando el mes OOT `20221201`.

Resultado de la prueba:

```text
PSI score: 0.006616 (OK)
Matriz de grupos rota: False
Reentrenamiento requerido: False
```
