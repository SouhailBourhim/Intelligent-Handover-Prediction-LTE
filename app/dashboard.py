"""
Phase 5 — Streamlit Dashboard

Features:
  • Network KPI monitor (RSRP, SINR, RSRQ, CQI per UE over time)
  • Handover event timeline
  • Real-time handover risk gauge (XGBoost prediction probability)
  • Model comparison table
  • Interactive filters: UE selector, time range
"""

import json
import sys
import numpy as np
import pandas as pd
import joblib
import torch
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

# Allow imports from src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import LSTMClassifier, SequenceDataset, SEQ_LEN

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LTE Handover Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data & models ─────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    raw  = pd.read_csv(ROOT / "data" / "raw" / "dataset.csv")
    test = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
    with open(ROOT / "data" / "processed" / "meta.json") as f:
        meta = json.load(f)
    return raw, test, meta


@st.cache_resource
def load_models(feat_cols):
    models = {}
    try:
        models["Random Forest"] = joblib.load(ROOT / "models" / "random_forest.pkl")
    except Exception:
        pass
    try:
        models["Logistic Regression"] = joblib.load(ROOT / "models" / "logistic_regression.pkl")
    except Exception:
        pass
    try:
        lstm = LSTMClassifier(input_size=len(feat_cols))
        lstm.load_state_dict(torch.load(ROOT / "models" / "lstm.pt", map_location="cpu"))
        lstm.eval()
        models["LSTM"] = lstm
    except Exception:
        pass
    return models


def predict_proba_sklearn(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def predict_proba_lstm(model, df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    ds = SequenceDataset(df, feat_cols, seq_len=SEQ_LEN)
    if len(ds) == 0:
        return np.array([])
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            probs.extend(torch.sigmoid(model(xb)).numpy())
    return np.array(probs)


# ── Sidebar ────────────────────────────────────────────────────────────────────

raw, test, meta = load_data()
feat_cols = [c for c in meta["feature_cols"] if c in test.columns]
models = load_models(feat_cols)

st.sidebar.title("📡 LTE Handover Prediction")
st.sidebar.markdown("---")

selected_model = st.sidebar.selectbox(
    "Active prediction model",
    options=list(models.keys()) if models else ["No models found"],
)

all_ues = sorted(raw["ue_id"].unique())
selected_ues = st.sidebar.multiselect(
    "UE filter", options=all_ues,
    default=all_ues[:5],
)

t_min, t_max = int(raw["timestamp"].min()), int(raw["timestamp"].max())
time_range = st.sidebar.slider(
    "Time range (s)",
    min_value=t_min, max_value=t_max,
    value=(t_min, t_min + 500),
    step=10,
)

prob_threshold = st.sidebar.slider("Risk threshold", 0.1, 0.9, 0.5, 0.05)

# ── Filter data ────────────────────────────────────────────────────────────────

mask = (
    raw["ue_id"].isin(selected_ues) &
    raw["timestamp"].between(*time_range)
)
df_view = raw[mask].copy()

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("📡 Intelligent Handover Prediction — LTE Network Dashboard")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total UEs", len(all_ues))
col2.metric("Timesteps shown", f"{df_view['timestamp'].nunique()}")
col3.metric("Handover events", int(df_view["handover_event"].sum()))
col4.metric("Dataset rows", f"{len(raw):,}")

st.markdown("---")

# ── KPI Charts ─────────────────────────────────────────────────────────────────

st.subheader("📊 Radio KPIs Over Time")

tab1, tab2, tab3, tab4 = st.tabs(["RSRP", "SINR", "RSRQ", "CQI"])

kpi_map = {
    "RSRP": ("rsrp_serving", "dBm"),
    "SINR": ("sinr",          "dB"),
    "RSRQ": ("rsrq_serving", "dB"),
    "CQI":  ("cqi",           ""),
}

for tab, (kpi_name, (col_name, unit)) in zip([tab1, tab2, tab3, tab4], kpi_map.items()):
    with tab:
        fig = go.Figure()
        for ue in selected_ues:
            ue_data = df_view[df_view["ue_id"] == ue]
            fig.add_trace(go.Scatter(
                x=ue_data["timestamp"], y=ue_data[col_name],
                mode="lines", name=f"UE {ue}",
                line=dict(width=1.5),
            ))
        # Mark handover events
        ho_data = df_view[df_view["handover_event"] == 1]
        fig.add_trace(go.Scatter(
            x=ho_data["timestamp"],
            y=ho_data[col_name],
            mode="markers", name="Handover",
            marker=dict(symbol="x", size=10, color="red"),
        ))
        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title=f"{kpi_name} ({unit})" if unit else kpi_name,
            height=350,
            legend=dict(orientation="h", y=-0.2),
            margin=dict(t=20),
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Handover timeline ──────────────────────────────────────────────────────────

st.subheader("🔁 Handover Event Timeline")

ho_events = df_view[df_view["handover_event"] == 1][
    ["timestamp", "ue_id", "serving_cell_id", "target_cell_id",
     "rsrp_serving", "rsrp_neighbor"]
].copy()

if ho_events.empty:
    st.info("No handover events in selected time window.")
else:
    fig_ho = px.scatter(
        ho_events, x="timestamp", y="ue_id",
        color="target_cell_id",
        symbol="target_cell_id",
        size_max=12,
        hover_data=["serving_cell_id", "target_cell_id",
                    "rsrp_serving", "rsrp_neighbor"],
        labels={"timestamp": "Time (s)", "ue_id": "UE ID",
                "target_cell_id": "Target Cell"},
        title="Handover Events (color = target cell)",
        height=300,
    )
    fig_ho.update_traces(marker=dict(size=10))
    st.plotly_chart(fig_ho, use_container_width=True)

# ── Handover Risk ──────────────────────────────────────────────────────────────

st.subheader("⚠️  Handover Risk Prediction")

if models and selected_model in models and selected_model != "No models found":
    model = models[selected_model]

    if selected_model == "LSTM":
        # Use test split for LSTM (needs sequences)
        test_ues = test[test["ue_id"].isin(selected_ues)]
        probs = predict_proba_lstm(model, test_ues, feat_cols)
        if len(probs) > 0:
            risk_val = float(probs.mean())
        else:
            risk_val = 0.0
        risk_df = None  # sequence misalignment makes per-row mapping non-trivial
        st.info("LSTM risk uses test-set sequences for selected UEs.")
    else:
        test_view = test[test["ue_id"].isin(selected_ues)].copy()
        X = test_view[feat_cols].fillna(0).values
        if len(X) > 0:
            probs = predict_proba_sklearn(model, X)
            test_view["risk"] = probs
            risk_val = float(probs.mean())

            # Heatmap: risk per (timestamp, UE)
            pivot = test_view.pivot_table(
                index="ue_id", columns="timestamp",
                values="risk", aggfunc="mean"
            )
            # Show only every 5th timestamp for readability
            cols_to_show = pivot.columns[::5]
            pivot = pivot[cols_to_show]

            fig_heat = px.imshow(
                pivot,
                color_continuous_scale="RdYlGn_r",
                zmin=0, zmax=1,
                labels=dict(x="Timestamp", y="UE ID", color="Risk"),
                title=f"Handover Risk Heatmap — {selected_model}",
                aspect="auto",
                height=300,
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            risk_val = 0.0

    # Risk gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_val * 100,
        number={"suffix": "%"},
        title={"text": f"Average Handover Risk ({selected_model})"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "darkred"},
            "steps": [
                {"range": [0, 30],  "color": "#2dc653"},
                {"range": [30, 60], "color": "#f0c040"},
                {"range": [60, 100],"color": "#e34234"},
            ],
            "threshold": {
                "line":  {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": prob_threshold * 100,
            },
        },
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=30, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

else:
    st.warning("Train models first: `python run_pipeline.py`")

# ── Model comparison table ─────────────────────────────────────────────────────

st.subheader("📋 Model Comparison")
report_path = ROOT / "reports" / "evaluation.txt"
if report_path.exists():
    report_lines = report_path.read_text().splitlines()
    # Extract summary rows
    rows = []
    for name in ["Logistic Regression", "Random Forest", "LSTM"]:
        prec = rec = f1 = auc = None
        for i, l in enumerate(report_lines):
            if l.strip().startswith(f"── {name}"):
                for j in range(i + 1, min(i + 6, len(report_lines))):
                    if "Precision" in report_lines[j]:
                        prec = report_lines[j].split(":")[-1].strip()
                    if "Recall" in report_lines[j]:
                        rec  = report_lines[j].split(":")[-1].strip()
                    if "F1-score" in report_lines[j]:
                        f1   = report_lines[j].split(":")[-1].strip()
                    if "ROC-AUC" in report_lines[j]:
                        auc  = report_lines[j].split(":")[-1].strip()
        if prec:
            rows.append({"Model": name, "Precision": prec,
                         "Recall": rec, "F1": f1, "ROC-AUC": auc})

    if rows:
        cmp_df = pd.DataFrame(rows).set_index("Model")
        st.dataframe(cmp_df, use_container_width=True)
    else:
        st.code(report_path.read_text())
else:
    st.info("Run the evaluation pipeline to see model comparison results.")

# ── UE Mobility map ────────────────────────────────────────────────────────────

st.subheader("🗺️ UE Mobility Tracks")
fig_map = px.scatter(
    df_view, x="pos_x", y="pos_y",
    color="ue_id", animation_frame=None,
    symbol="ue_id",
    opacity=0.4,
    labels={"pos_x": "X (m)", "pos_y": "Y (m)", "ue_id": "UE"},
    title="UE Positions (all selected timesteps)",
    height=420,
)

# Add base station markers
for bs_id, (bx, by) in enumerate([(250,250),(750,250),(250,750),(750,750)]):
    fig_map.add_trace(go.Scatter(
        x=[bx], y=[by], mode="markers+text",
        marker=dict(symbol="square", size=18, color="black"),
        text=[f"BS{bs_id}"], textposition="top center",
        name=f"BS{bs_id}", showlegend=False,
    ))
fig_map.update_layout(xaxis_range=[0, 1000], yaxis_range=[0, 1000])
st.plotly_chart(fig_map, use_container_width=True)

st.caption("LTE Handover Prediction Dashboard · Phase 5")
