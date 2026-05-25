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


DIST_GE = [0, 0.035, 0.087, 0.237, 0.393, 0.529, 0.664, 0.787, 0.862, 0.95, 1.0]


def get_groups(scores, df_post):
    """calcula puntuacion_tlv y grupo_ejec_tlv con la logica de clase."""

    df_post["prob"] = scores
    df_post["prob_frescura"] = np.where(
        df_post["grp_campecs06m"] == "G1",
        0.066,
        np.where(
            df_post["grp_campecs06m"] == "G2",
            0.028,
            np.where(
                df_post["grp_campecs06m"] == "G3",
                0.022,
                np.where(df_post["grp_campecs06m"] == "G4", 0.008, 0.004),
            ),
        ),
    )
    df_post["prob_value_contact"] = df_post["prob_value_contact"].fillna(0.000001)
    df_post["puntuacion_tlv"] = (
        df_post["prob"]
        * df_post["prob_value_contact"]
        * np.log(df_post["monto"] + 1)
        * df_post["prob_frescura"]
    )
    df_post["grupo_ejec_tlv"] = pd.qcut(
        df_post["puntuacion_tlv"], q=DIST_GE, labels=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    )
    return df_post


def run_postprocessing(
    scores,
    df_post: pd.DataFrame,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """ejecuta get_groups y guarda el resultado TLV si se indica una ruta."""

    _validate_postprocessing_columns(df_post)

    result = get_groups(scores, df_post.copy())
    result["score"] = result["prob"]
    result["grupo_ejec_tlv"] = result["grupo_ejec_tlv"].astype(int)
    result = result.sort_values(["p_fecinformacion", "grupo_ejec_tlv", "puntuacion_tlv"], ascending=[True, True, False])
    result["orden"] = result.groupby("p_fecinformacion").cumcount() + 1

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
    return result


def save_replica(
    df_post: pd.DataFrame,
    table: str,
    partition: str,
    dir_s3: str | Path = "data/replica/s3",
    dir_athena: str | Path = "data/replica/athena",
    dir_onpremise: str | Path = "data/replica/onpremise",
) -> dict[str, str]:
    """genera archivos pipe-delimitados para s3, athena y onpremise."""

    # la replica final usa el formato pipe-delimited pedido para ambientes destino.
    replica = pd.DataFrame(
        {
            "codmes": df_post[config.columns.month].astype(int),
            "tipdoc": df_post.get("tip_doc", ""),
            "coddoc": df_post[config.columns.customer_id],
            "puntuacion": df_post["puntuacion_tlv"],
            "modelo": table,
            "fec_replica": date.today().isoformat(),
            "grupo_ejec": df_post["grupo_ejec_tlv"].astype(int),
            "score": df_post["score"],
            "orden": df_post["orden"].astype(int),
            "variable1": df_post[config.columns.campaign_group],
            "variable2": df_post[config.columns.amount],
            "variable3": df_post[config.columns.contact_value_probability],
        }
    )[REPLICA_COLUMNS]

    paths = {}
    for name, directory in {
        "s3": Path(dir_s3),
        "athena": Path(dir_athena),
        "onpremise": Path(dir_onpremise),
    }.items():
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{table}_{partition}.txt"
        replica.to_csv(path, sep="|", index=False)
        paths[name] = str(path)
    return paths


def _validate_postprocessing_columns(df: pd.DataFrame) -> None:
    required = [
        config.columns.month,
        config.columns.customer_id,
        config.columns.campaign_group,
        config.columns.contact_value_probability,
        config.columns.amount,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para postprocesamiento TLV: {missing}")
