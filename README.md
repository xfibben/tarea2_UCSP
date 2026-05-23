# Tarea 2 - Pipeline ML E2E CU Venta

Proyecto en reimplementacion para la tarea de pipeline ML end-to-end del curso.

Esta version reemplaza el trabajo anterior y sigue la consigna especifica del caso CU Venta:

- Dataset del Drive del curso.
- Split temporal con validacion OOT.
- Entrenamiento con Optuna.
- Monitoreo con PSI, AUC y recall por decil.
- Postprocesamiento TLV con `get_groups()` copiado tal cual desde clase.
- Reentrenamiento automatico cuando falle el monitoreo.

## Dataset

El dataset no se incluye en el repositorio por tamano. Descargar los archivos CSV desde:

```text
https://drive.google.com/drive/folders/1BbaYLS_Cy5pbvfE6JH3P7KlLdlRf_Cds?usp=drive_link
```

Opciones de uso:

1. Colocar los CSV en `data/raw/`.
2. Usar una carpeta externa y pasarla por parametro:

```bash
python main.py inspect-data --data-dir /ruta/local/dataset_cu_venta
```

Tambien se puede definir la variable de entorno:

```bash
export CU_DATA_DIR=/ruta/local/dataset_cu_venta
```

## Instalacion

En macOS, XGBoost necesita OpenMP. Si aparece un error sobre `libomp.dylib`, instalar:

```bash
brew install libomp
```

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Estado del proyecto

Fase actual: **Etapa 2 - Entrenamiento y monitoreo**.

Implementado:

- Estructura inicial.
- Configuracion central.
- Exclusiones para no subir datasets pesados.
- Comando de inspeccion de CSV.
- Preprocesamiento con split temporal train/test/OOT.
- Entrenamiento XGBoost con Optuna.
- Registro de experimento y modelo en MLflow.
- Monitoreo de score con PSI.
- AUC de validacion y recall por decil.
- Reentrenamiento automatico si el PSI entra en estado `ALERT`.
- Validacion tecnica con datos sinteticos usando `--n-trials 1`.

Pendiente:

- Postprocesamiento TLV.
- Replica final.

## Comandos

Inspeccionar la carpeta de datos:

```bash
python main.py inspect-data --data-dir /ruta/local/dataset_cu_venta
```

Cuando los CSV se coloquen en `data/raw/`, basta con:

```bash
python main.py inspect-data
```

Entrenar con Optuna, registrar en MLflow y monitorear OOT:

```bash
python main.py train --data-dir /ruta/local/dataset_cu_venta --n-trials 30
```

Si se desea indicar manualmente el mes OOT:

```bash
python main.py train --data-dir /ruta/local/dataset_cu_venta --validation-codmes 201912 --n-trials 30
```

El flujo disponible hasta fase 2 tambien puede ejecutarse con:

```bash
python main.py run-all --data-dir /ruta/local/dataset_cu_venta
```

El comando de inferencia queda reservado para la fase 3:

```bash
python main.py inference
```

## Estructura

```text
.
├── main.py
├── config.py
├── preprocessing.py
├── training.py
├── monitoring.py
├── postprocessing.py
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   ├── postprocessed/
│   └── replica/
└── artifacts/
    ├── models/
    ├── metrics/
    └── monitoring/
```

## Regla de entrega

No subir los CSV ni artefactos generados a GitHub. El repositorio debe contener codigo, documentacion y estructura; los datos deben descargarse desde el Drive de la tarea.

## Nota de validacion

La fase 2 ya fue probada con datos sinteticos que respetan las columnas esperadas de la consigna. Falta ejecutar el mismo flujo con los CSV reales del Drive para confirmar nombres de columnas, meses disponibles y comportamiento OOT.
