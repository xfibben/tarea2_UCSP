from __future__ import annotations

from pathlib import Path

from config import config


def _pd():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Falta pandas. Instala dependencias con: pip install -r requirements.txt"
        ) from exc
    return pd


def list_csv_files(data_dir: Path | None = None) -> list[Path]:
    base_dir = data_dir or config.paths.raw_dir
    return sorted(base_dir.glob("*.csv"))


def profile_raw_files(data_dir: Path | None = None, sample_rows: int = 5000):
    """Inspecciona archivos grandes leyendo solo una muestra inicial."""

    pd = _pd()
    rows: list[dict[str, object]] = []
    for csv_path in list_csv_files(data_dir):
        sample = pd.read_csv(csv_path, nrows=sample_rows)
        rows.append(
            {
                "file": csv_path.name,
                "size_mb": round(csv_path.stat().st_size / 1024 / 1024, 2),
                "sample_rows": len(sample),
                "columns": len(sample.columns),
                "has_month": config.columns.month in sample.columns,
                "has_target": config.columns.target in sample.columns,
                "has_customer_id": config.columns.customer_id in sample.columns,
                "months_in_sample": _safe_unique_values(sample, config.columns.month),
            }
        )
    return pd.DataFrame(rows)


def read_schema_sample(csv_path: Path, sample_rows: int = 5000):
    pd = _pd()
    sample = pd.read_csv(csv_path, nrows=sample_rows)
    return pd.DataFrame(
        {
            "column": sample.columns,
            "dtype_sample": [str(dtype) for dtype in sample.dtypes],
            "null_rate_sample": [round(sample[column].isna().mean(), 4) for column in sample.columns],
        }
    )


def _safe_unique_values(df: pd.DataFrame, column: str, limit: int = 12) -> list[object]:
    if column not in df.columns:
        return []
    values = sorted(df[column].dropna().unique().tolist())
    return values[:limit]


def run_preprocessing(*_: object, **__: object) -> None:
    raise NotImplementedError("Se implementara en la etapa 2 despues de validar el esquema real.")
