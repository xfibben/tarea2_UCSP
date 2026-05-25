from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


POST_PATH = Path("data/postprocessed/output_tlv.csv")
MONITORING_PATH = Path("artifacts/monitoring/monitoring_report.json")


@st.cache_data
def load_tlv(path: Path) -> pd.DataFrame:
    columns = [
        "key_value",
        "p_fecinformacion",
        "monto",
        "prob",
        "score",
        "puntuacion_tlv",
        "grupo_ejec_tlv",
    ]
    return pd.read_csv(path, usecols=columns)


@st.cache_data
def load_monitoring(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


st.set_page_config(page_title="Dashboard CU Venta", layout="wide")
st.title("Dashboard Pipeline CU Venta")
st.caption("resultados del scoring TLV y monitoreo OOT")

if not POST_PATH.exists():
    st.error("no existe data/postprocessed/output_tlv.csv. Ejecuta primero: python main.py run-all")
    st.stop()

df = load_tlv(POST_PATH)
monitoring = load_monitoring(MONITORING_PATH) if MONITORING_PATH.exists() else {}

col1, col2, col3, col4 = st.columns(4)
col1.metric("clientes evaluados", f"{len(df):,}")
col2.metric("PSI", round(float(monitoring.get("score_psi", 0)), 6))
col3.metric("estado PSI", str(monitoring.get("score_psi_status", "sin dato")))
col4.metric("AUC OOT", round(float(monitoring.get("val_auc", 0)), 6))

st.subheader("Distribucion por grupo de ejecucion")
group_counts = df["grupo_ejec_tlv"].value_counts().sort_index()
st.bar_chart(group_counts)

st.subheader("Resumen por grupo")
summary = (
    df.groupby("grupo_ejec_tlv", observed=True)
    .agg(
        clientes=("key_value", "count"),
        score_promedio=("score", "mean"),
        tlv_promedio=("puntuacion_tlv", "mean"),
        monto_promedio=("monto", "mean"),
    )
    .reset_index()
    .sort_values("grupo_ejec_tlv")
)
st.dataframe(summary, width="stretch")

st.subheader("Top 20 clientes por puntuacion TLV")
top_clients = df.sort_values("puntuacion_tlv", ascending=False).head(20)
st.dataframe(
    top_clients[["key_value", "prob", "score", "puntuacion_tlv", "grupo_ejec_tlv", "monto"]],
    width="stretch",
)

if monitoring.get("recall_by_decile"):
    st.subheader("Recall por decil")
    recall = pd.DataFrame(monitoring["recall_by_decile"])
    st.dataframe(recall, width="stretch")
