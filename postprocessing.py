from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config import config


REPLICA_COLUMNS = [
    "codmes",
    "tipdoc",
    "coddoc",
    "puntuacion",
    "modelo",
    "fec_replica",
    "grupo_ejec",
    "score",
    "orden",
    "variable1",
    "variable2",
    "variable3",
]


# esta funcion se deja aislada porque la consigna pide respetar la logica TLV de clase.
def get_groups(df):
    DIST_GE = [0, 0.035, 0.087, 0.237, 0.393, 0.529, 0.664, 0.787, 0.862, 0.95, 1.0]
    df["prob_frescura"] = np.where(
        df["grp_campecs06m"] == "G1",
        0.066,
        np.where(
            df["grp_campecs06m"] == "G2",
            0.028,
            np.where(
                df["grp_campecs06m"] == "G3",
                0.022,
                np.where(df["grp_campecs06m"] == "G4", 0.008, 0.004),
            ),
        ),
    )
    df["prob_value_contact"] = df["prob_value_contact"].fillna(0.000001)
    df["puntuacion_tlv"] = (
        df["prob"] * df["prob_value_contact"] * np.log(df["monto"] + 1) * df["prob_frescura"]
    )
    df["grupo_ejec_tlv"] = pd.qcut(
        df["puntuacion_tlv"], q=DIST_GE, labels=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    )
    return df


def run_postprocessing(
    scored: pd.DataFrame,
    output_dir: Path | None = None,
    replica_dir: Path | None = None,
) -> dict[str, str]:
    _validate_postprocessing_columns(scored)

    output_dir = output_dir or config.paths.postprocessed_dir
    replica_dir = replica_dir or config.paths.replica_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for target in ("s3", "athena", "onpremise"):
        (replica_dir / target).mkdir(parents=True, exist_ok=True)

    result = scored.copy()
    if "prob" not in result.columns:
        result["prob"] = result["score"]
    result = get_groups(result)
    result["grupo_ejec_tlv"] = result["grupo_ejec_tlv"].astype(int)
    result = result.sort_values(["p_fecinformacion", "grupo_ejec_tlv", "puntuacion_tlv"], ascending=[True, True, False])
    result["orden"] = result.groupby("p_fecinformacion").cumcount() + 1

    postprocessed_path = output_dir / "df_scored_tlv.csv"
    result.to_csv(postprocessed_path, index=False)

    replica = _build_replica(result)
    paths = {"postprocessed": str(postprocessed_path)}
    for target in ("s3", "athena", "onpremise"):
        path = replica_dir / target / "cu_venta_replica.txt"
        replica.to_csv(path, sep="|", index=False)
        paths[target] = str(path)
    return paths


def _build_replica(df: pd.DataFrame) -> pd.DataFrame:
    # la replica final usa el formato pipe-delimited pedido para ambientes destino.
    replica = pd.DataFrame(
        {
            "codmes": df[config.columns.month].astype(int),
            "tipdoc": df.get("tip_doc", ""),
            "coddoc": df[config.columns.customer_id],
            "puntuacion": df["puntuacion_tlv"],
            "modelo": config.model_name,
            "fec_replica": date.today().isoformat(),
            "grupo_ejec": df["grupo_ejec_tlv"].astype(int),
            "score": df["score"],
            "orden": df["orden"].astype(int),
            "variable1": df[config.columns.campaign_group],
            "variable2": df[config.columns.amount],
            "variable3": df[config.columns.contact_value_probability],
        }
    )
    return replica[REPLICA_COLUMNS]


def _validate_postprocessing_columns(df: pd.DataFrame) -> None:
    required = [
        config.columns.month,
        config.columns.customer_id,
        config.columns.campaign_group,
        config.columns.contact_value_probability,
        config.columns.amount,
        "score",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para postprocesamiento TLV: {missing}")
