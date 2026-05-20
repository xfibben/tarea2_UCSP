from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Paths:
    """Rutas del proyecto. DATA_DIR permite evaluar con datos fuera del repo."""

    raw_dir: Path = Path(os.getenv("CU_DATA_DIR", ROOT_DIR / "data" / "raw"))
    processed_dir: Path = ROOT_DIR / "data" / "processed"
    postprocessed_dir: Path = ROOT_DIR / "data" / "postprocessed"
    replica_dir: Path = ROOT_DIR / "data" / "replica"
    artifacts_dir: Path = ROOT_DIR / "artifacts"
    model_dir: Path = ROOT_DIR / "artifacts" / "models"
    metrics_dir: Path = ROOT_DIR / "artifacts" / "metrics"
    monitoring_dir: Path = ROOT_DIR / "artifacts" / "monitoring"


@dataclass(frozen=True)
class Columns:
    month: str = "p_codmes"
    target: str = "target"
    customer_id: str = "key_value"
    campaign_group: str = "grp_campecs06m"
    contact_value_probability: str = "prob_value_contact"
    amount: str = "monto"


@dataclass(frozen=True)
class SplitConfig:
    validation_codmes: float = 201912.0
    test_size: float = 0.30
    random_state: int = 123


@dataclass(frozen=True)
class MonitoringConfig:
    psi_ok_threshold: float = 0.10
    psi_alert_threshold: float = 0.25


@dataclass(frozen=True)
class ProjectConfig:
    paths: Paths = Paths()
    columns: Columns = Columns()
    split: SplitConfig = SplitConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    drive_url: str = (
        "https://drive.google.com/drive/folders/"
        "1BbaYLS_Cy5pbvfE6JH3P7KlLdlRf_Cds?usp=drive_link"
    )


config = ProjectConfig()


def ensure_project_dirs() -> None:
    for directory in (
        config.paths.raw_dir,
        config.paths.processed_dir,
        config.paths.postprocessed_dir,
        config.paths.replica_dir / "s3",
        config.paths.replica_dir / "athena",
        config.paths.replica_dir / "onpremise",
        config.paths.model_dir,
        config.paths.metrics_dir,
        config.paths.monitoring_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

