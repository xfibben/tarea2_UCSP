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

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Estado del proyecto

Fase actual: **Etapa 1 - Base del proyecto y entendimiento de datos**.

Implementado:

- Estructura inicial.
- Configuracion central.
- Exclusiones para no subir datasets pesados.
- Comando de inspeccion de CSV.
- Notas del dataset.

Pendiente:

- Preprocesamiento completo.
- Entrenamiento con Optuna.
- Registro MLflow.
- Monitoreo PSI/AUC/recall.
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

Los siguientes comandos quedan reservados para las siguientes etapas:

```bash
python main.py train
python main.py inference
python main.py run-all
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
├── DATA_NOTES.md
├── PLAN_REIMPLEMENTACION.md
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

