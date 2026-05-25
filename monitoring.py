from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score

from config import config


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    quantiles = np.linspace(0, 1, buckets + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    if len(cuts) < 3:
        cuts = np.linspace(0, 1, buckets + 1)

    expected_counts = np.histogram(expected, bins=cuts)[0] / max(len(expected), 1)
    actual_counts = np.histogram(actual, bins=cuts)[0] / max(len(actual), 1)

    expected_counts = np.clip(expected_counts, 1e-6, None)
    actual_counts = np.clip(actual_counts, 1e-6, None)
    return float(np.sum((actual_counts - expected_counts) * np.log(actual_counts / expected_counts)))


def psi_flag(value: float) -> str:
    if value < config.monitoring.psi_ok_threshold:
        return "OK"
    if value < config.monitoring.psi_alert_threshold:
        return "WARN"
    return "ALERT"


def recall_by_decile(y_true: np.ndarray, scores: np.ndarray, n_deciles: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"target": y_true, "score": scores})
    df["decile"] = pd.qcut(df["score"], q=n_deciles, labels=range(n_deciles, 0, -1), duplicates="drop")
    rows = []
    for decile, group in df.groupby("decile", observed=True):
        prediction = np.ones(len(group), dtype=int)
        rows.append(
            {
                "decile": int(decile),
                "rows": int(len(group)),
                "target_rate": float(group["target"].mean()),
                "recall_if_selected": float(recall_score(group["target"], prediction, zero_division=0)),
            }
        )
    return pd.DataFrame(rows).sort_values("decile")


def compute_recall_by_decile(y_true: np.ndarray, scores: np.ndarray, n_deciles: int = 10) -> pd.DataFrame:
    return recall_by_decile(y_true, scores, n_deciles)


def run_monitoring(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    y_val: np.ndarray,
    output_path: Path | None = None,
) -> dict[str, object]:
    """calcula PSI, AUC y recall por decil sobre el mes OOT."""

    output_path = output_path or config.paths.monitoring_dir / "monitoring_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    psi_score = psi(train_scores, val_scores)
    auc_val = roc_auc_score(y_val, val_scores) if len(np.unique(y_val)) == 2 else None
    recall_deciles = recall_by_decile(y_val, val_scores)

    report = {
        "score_psi": psi_score,
        "score_psi_status": psi_flag(psi_score),
        "val_auc": auc_val,
        "requires_retraining": psi_flag(psi_score) == "ALERT",
        "recall_by_decile": recall_deciles.to_dict(orient="records"),
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    recall_deciles.to_csv(config.paths.monitoring_dir / "recall_by_decile.csv", index=False)
    return report


def groups_matrix_is_broken(
    current: pd.DataFrame,
    group_column: str = "grupo_ejec_tlv",
    min_expected_groups: int = 10,
) -> bool:
    if group_column not in current.columns:
        return True
    return current[group_column].nunique(dropna=True) < min_expected_groups
