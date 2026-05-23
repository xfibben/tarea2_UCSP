from __future__ import annotations

import json
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


def read_raw_dataset(data_dir: Path | None = None) -> object:
    pd = _pd()
    files = list_csv_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No se encontraron CSV en {data_dir or config.paths.raw_dir}")

    frames = []
    for csv_path in files:
        frame = pd.read_csv(csv_path)
        frame["_source_file"] = csv_path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


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


def run_preprocessing(
    data_dir: Path | None = None,
    validation_codmes: float | None = None,
) -> dict[str, str]:
    pd = _pd()
    df = read_raw_dataset(data_dir)
    _validate_required_columns(df)

    month_col = config.columns.month
    target_col = config.columns.target
    validation_month = validation_codmes or config.split.validation_codmes

    df = _drop_sparse_columns(df)
    df[target_col] = df[target_col].astype(int)

    if validation_month not in set(df[month_col].dropna().unique()):
        # Preferimos no fallar antes de conocer el dataset real; se usa el ultimo mes como OOT.
        validation_month = float(sorted(df[month_col].dropna().unique())[-1])

    df_val_raw = df[df[month_col] == validation_month].copy()
    df_model = df[df[month_col] != validation_month].copy()

    df_train_raw, df_test_raw = _split_train_test(df_model)
    encoded = _encode_frames(df_train_raw, df_test_raw, df_val_raw)

    config.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": config.paths.processed_dir / "df_train.csv",
        "test": config.paths.processed_dir / "df_test.csv",
        "val": config.paths.processed_dir / "df_val.csv",
        "metadata": config.paths.processed_dir / "preprocessing_metadata.json",
    }

    encoded["train"].to_csv(paths["train"], index=False)
    encoded["test"].to_csv(paths["test"], index=False)
    encoded["val"].to_csv(paths["val"], index=False)

    metadata = {
        "source_files": [path.name for path in list_csv_files(data_dir)],
        "rows": int(len(df)),
        "train_rows": int(len(encoded["train"])),
        "test_rows": int(len(encoded["test"])),
        "val_rows": int(len(encoded["val"])),
        "validation_codmes": validation_month,
        "feature_columns": encoded["feature_columns"],
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {key: str(value) for key, value in paths.items()}


def _validate_required_columns(df: object) -> None:
    required = [config.columns.month, config.columns.target]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")


def _drop_sparse_columns(df: object):
    threshold = config.split.nan_threshold_pct / 100
    keep = [column for column in df.columns if df[column].isna().mean() <= threshold]
    return df[keep].copy()


def _split_train_test(df: object):
    from sklearn.model_selection import train_test_split

    target = config.columns.target
    stratify = df[target] if df[target].nunique(dropna=True) == 2 else None
    return train_test_split(
        df,
        test_size=config.split.test_size,
        random_state=config.split.random_state,
        stratify=stratify,
    )


def _encode_frames(df_train, df_test, df_val) -> dict[str, object]:
    pd = _pd()
    protected = set(config.columns.protected) | {"_source_file"}
    frames = {"train": df_train.copy(), "test": df_test.copy(), "val": df_val.copy()}

    combined = pd.concat(frames.values(), keys=frames.keys(), names=["split"])
    object_columns = [
        column
        for column in combined.columns
        if column not in protected and combined[column].dtype == "object"
    ]
    combined = pd.get_dummies(combined, columns=object_columns, dummy_na=True)

    encoded = {
        split: combined.xs(split, level="split").reset_index(drop=True)
        for split in frames
    }
    feature_columns = _feature_columns(encoded["train"])
    encoded["feature_columns"] = feature_columns
    return encoded


def _feature_columns(df: object) -> list[str]:
    pd = _pd()
    protected = set(config.columns.protected) | {"_source_file"}
    return [
        column
        for column in df.columns
        if column not in protected and pd.api.types.is_numeric_dtype(df[column])
    ]
