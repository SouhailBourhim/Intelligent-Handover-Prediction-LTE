"""
Phase 5 — Streamlit Dashboard

Features:
  • Champion model banner  (from models/champion/metadata.json)
  • Network KPI monitor    (RSRP, SINR, RSRQ, CQI per UE over time)
  • Handover event timeline
  • Real-time handover risk gauge
  • Model comparison table
  • MLflow runs table       (reads mlruns/ experiment)
  • SHAP explanation tab    (pre-computed bar / beeswarm / waterfall plots)
  • Interactive filters: model selector, UE filter, time range, risk threshold
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

from src.models import (
    LSTMClassifier, GRUClassifier,
    SequenceDataset, SEQ_LEN,
    _build_seq_row_indices,
)

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
    for name, fname in [
        ("Random Forest",     "random_forest.pkl"),
        ("Logistic Regression", "logistic_regression.pkl"),
        ("XGBoost",           "xgboost.pkl"),
        ("Stacking Ensemble", "stacking_ensemble.pkl"),
    ]:
        try:
            models[name] = joblib.load(ROOT / "models" / fname)
        except Exception:
            pass

    for name, cls, fname in [
        ("LSTM", LSTMClassifier, "lstm.pt"),
        ("GRU",  GRUClassifier,  "gru.pt"),
    ]:
        try:
            net = cls(input_size=len(feat_cols))
            net.load_state_dict(
                torch.load(ROOT / "models" / fname, map_location="cpu")
            )
            net.eval()
            models[name] = net
        except Exception:
            pass
    return models


@st.cache_data
def load_champion_meta():
    path = ROOT / "models" / "champion" / "metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── Prediction helpers ─────────────────────────────────────────────────────────

def predict_proba_sklearn(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def predict_proba_seq(net: torch.nn.Module,
                      df: pd.DataFrame,
                      feat_cols: list[str]) -> tuple[np.ndarray, list[int]]:
    """Return (probs, aligned_row_indices)."""
    ds = SequenceDataset(df, feat_cols, seq_len=SEQ_LEN)
    if len(ds) == 0:
        return np.array([]), []
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            probs.extend(torch.sigmoid(net(xb)).numpy())
    row_indices = _build_seq_row_indices(df, seq_len=SEQ_LEN)
    return np.array(probs), row_indices


def predict_stacking(meta_model,
                     df: pd.DataFrame,
                     feat_cols: list[str],
                     xgb_model, rf_model,
                     lstm_model: torch.nn.Module) -> tuple[np.ndarray, list[int]]:
    """Use LSTM probs as anchor row set; stack XGB + RF + LSTM."""
    lstm_probs, row_indices = predict_proba_seq(lstm_model, df, feat_cols)
    if len(lstm_probs) == 0:
        return np.array([]), []
    sub   = df.loc[row_indices].copy()
    X_sub = sub[feat_cols].fillna(0).values
    xgb_p = xgb_model.predict_proba(X_sub)[:, 1]
    rf_p  = rf_model.predict_proba(X_sub)[:, 1]
    meta_X = np.column_stack([xgb_p, rf_p, lstm_probs])
    probs  = meta_model.predict_proba(meta_X)[:, 1]
    return probs, row_indices


# ── Bootstrap ──────────────────────────────────────────────────────────────────

raw, test, meta = load_data()
feat_cols = [c for c in meta["feature_cols"] if c in test.columns]
models    = load_models(feat_cols)
champion  = load_champion_meta()

# ── Champion banner ────────────────────────────────────────────────────────────

if champion:
    st.success(
        f"🏆 **Champion model:** {champion.get('model_name', '?')}  |  "
        f"ROC-AUC: {champion.get('roc_auc', '?')}  |  "
        f"Promoted: {champion.get('promoted_at', '?')}"
    )

# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.title("📡 LTE Handover Prediction")
st.sidebar.markdown("---")

_sklearn_seq_names = [
    "Random Forest", "Logistic Regression", "XGBoost",
    "LSTM", "GRU", "Stacking Ensemble",
]
available_models = [m for m in _sklearn_seq_names if m in models]
if not available_models:
    available_models = ["No models found"]

selected_model = st.sidebar.selectbox("Active prediction model", options=available_models)

all_ues = sorted(raw["ue_id"].unique())
selected_ues = st.sidebar.multiselect(
    "UE filter", options=all_ues, default=all_ues[:5],
)

t_min, t_max = int(raw["timestamp"].min()), int(raw["timestamp"].max())
time_range = st.sidebar.slider(
    "Time range (s)",
    min_value=t_min, max_value=t_max,
    value=(t_min, t_min + 500), step=10,
)

prob_threshold = st.sidebar.slider("Risk threshold", 0.1, 0.9, 0.5, 0.05)

# ── Filter data ────────────────────────────────────────────────────────────────

mask    = raw["ue_id"].isin(selected_ues) & raw["timestamp"].between(*time_range)
df_view = raw[mask].copy()

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("📡 Intelligent Handover Prediction — LTE Network Dashboard")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total UEs",        len(all_ues))
col2.metric("Timesteps shown",  df_view["timestamp"].nunique())
col3.metric("Handover events",  int(df_view["handover_event"].sum()))
col4.metric("Dataset rows",     f"{len(raw):,}")

st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_kpi, tab_ho, tab_risk, tab_cmp, tab_shap, tab_mlf, tab_map = st.tabs(
    ["📊 Radio KPIs", "🔁 HO Timeline", "⚠️ Risk", "📋 Model Comparison",
     "🔍 SHAP Explanations", "🧪 MLflow Runs", "🗺️ Mobility Map"]
)

# ── Tab 1: KPI Charts ──────────────────────────────────────────────────────────

with tab_kpi:
    st.subheader("Radio KPIs Over Time")
    kpi_tabs = st.tabs(["RSRP", "SINR", "RSRQ", "CQI"])
    kpi_map = {
        "RSRP": ("rsrp_serving", "dBm"),
        "SINR": ("sinr",          "dB"),
        "RSRQ": ("rsrq_serving",  "dB"),
        "CQI":  ("cqi",           ""),
    }
    for kt, (kpi_name, (col_name, unit)) in zip(kpi_tabs, kpi_map.items()):
        with kt:
            fig = go.Figure()
            for ue in selected_ues:
                ue_data = df_view[df_view["ue_id"] == ue]
                fig.add_trace(go.Scatter(
                    x=ue_data["timestamp"], y=ue_data[col_name],
                    mode="lines", name=f"UE {ue}", line=dict(width=1.5),
                ))
            ho_data = df_view[df_view["handover_event"] == 1]
            fig.add_trace(go.Scatter(
                x=ho_data["timestamp"], y=ho_data[col_name],
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

# ── Tab 2: HO Timeline ─────────────────────────────────────────────────────────

with tab_ho:
    st.subheader("Handover Event Timeline")
    ho_events = df_view[df_view["handover_event"] == 1][
        ["timestamp", "ue_id", "serving_cell_id", "target_cell_id",
         "rsrp_serving", "rsrp_neighbor"]
    ].copy()
    if ho_events.empty:
        st.info("No handover events in selected time window.")
    else:
        fig_ho = px.scatter(
            ho_events, x="timestamp", y="ue_id",
            color="target_cell_id", symbol="target_cell_id", size_max=12,
            hover_data=["serving_cell_id", "target_cell_id",
                        "rsrp_serving", "rsrp_neighbor"],
            labels={"timestamp": "Time (s)", "ue_id": "UE ID",
                    "target_cell_id": "Target Cell"},
            title="Handover Events (colour = target cell)",
            height=320,
        )
        fig_ho.update_traces(marker=dict(size=10))
        st.plotly_chart(fig_ho, use_container_width=True)

# ── Tab 3: Risk ────────────────────────────────────────────────────────────────

with tab_risk:
    st.subheader(f"Handover Risk Prediction — {selected_model}")

    if models and selected_model in models and selected_model != "No models found":
        model  = models[selected_model]
        risk_val = 0.0

        _SEQ_MODELS    = {"LSTM", "GRU"}
        _STACK_MODEL   = "Stacking Ensemble"

        test_ues = test[test["ue_id"].isin(selected_ues)]

        if selected_model in _SEQ_MODELS:
            probs, row_idx = predict_proba_seq(model, test_ues, feat_cols)
            if len(probs) > 0:
                risk_val = float(probs.mean())
            st.info(f"{selected_model} risk uses test-set sequences for selected UEs.")

        elif selected_model == _STACK_MODEL:
            xgb_m  = models.get("XGBoost")
            rf_m   = models.get("Random Forest")
            lstm_m = models.get("LSTM")
            if xgb_m and rf_m and lstm_m:
                probs, row_idx = predict_stacking(
                    model, test_ues, feat_cols, xgb_m, rf_m, lstm_m
                )
                if len(probs) > 0:
                    risk_val = float(probs.mean())
            else:
                st.warning("XGBoost, Random Forest, or LSTM model not loaded.")

        else:
            test_view = test_ues.copy()
            X = test_view[feat_cols].fillna(0).values
            if len(X) > 0:
                probs    = predict_proba_sklearn(model, X)
                risk_val = float(probs.mean())
                test_view = test_view.copy()
                test_view["risk"] = probs

                pivot = test_view.pivot_table(
                    index="ue_id", columns="timestamp",
                    values="risk", aggfunc="mean"
                )
                cols_show = pivot.columns[::5]
                fig_heat = px.imshow(
                    pivot[cols_show],
                    color_continuous_scale="RdYlGn_r", zmin=0, zmax=1,
                    labels=dict(x="Timestamp", y="UE ID", color="Risk"),
                    title=f"Handover Risk Heatmap — {selected_model}",
                    aspect="auto", height=300,
                )
                st.plotly_chart(fig_heat, use_container_width=True)

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_val * 100,
            number={"suffix": "%"},
            title={"text": f"Average Handover Risk ({selected_model})"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "darkred"},
                "steps": [
                    {"range": [0,  30],  "color": "#2dc653"},
                    {"range": [30, 60],  "color": "#f0c040"},
                    {"range": [60, 100], "color": "#e34234"},
                ],
                "threshold": {
                    "line":      {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value":     prob_threshold * 100,
                },
            },
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=30, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.warning("Train models first: `python run_pipeline.py`")

# ── Tab 4: Model Comparison ────────────────────────────────────────────────────

with tab_cmp:
    st.subheader("Model Comparison")
    report_path = ROOT / "reports" / "evaluation.txt"
    if report_path.exists():
        report_lines = report_path.read_text().splitlines()
        rows = []
        model_names = [
            "Logistic Regression", "Random Forest", "XGBoost",
            "LSTM", "GRU", "Stacking Ensemble",
        ]
        for name in model_names:
            prec = rec = f1 = auc = None
            for i, ln in enumerate(report_lines):
                if ln.strip().startswith(f"── {name}"):
                    for j in range(i + 1, min(i + 8, len(report_lines))):
                        if "Precision" in report_lines[j]:
                            prec = report_lines[j].split(":")[-1].strip()
                        if "Recall"    in report_lines[j]:
                            rec  = report_lines[j].split(":")[-1].strip()
                        if "F1-score"  in report_lines[j]:
                            f1   = report_lines[j].split(":")[-1].strip()
                        if "ROC-AUC"  in report_lines[j]:
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
        st.info("Run `python run_pipeline.py` to generate evaluation results.")

# ── Tab 5: SHAP Explanations ───────────────────────────────────────────────────

with tab_shap:
    st.subheader("SHAP Feature Explanations")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) shows how much each feature "
        "contributed to a model's prediction — both globally (average over the "
        "test set) and for individual predictions.\n\n"
        "**Supported models:** Logistic Regression · Random Forest · XGBoost  \n"
        "Run `python run_pipeline.py --phase 5` to regenerate plots after retraining."
    )

    SHAP_DIR = ROOT / "reports" / "shap"

    _SHAP_MODELS = {
        "Logistic Regression": "logistic_regression",
        "Random Forest":       "random_forest",
        "XGBoost":             "xgboost",
    }

    shap_model_sel = st.selectbox(
        "Select model to explain",
        options=list(_SHAP_MODELS.keys()),
        key="shap_model_sel",
    )
    slug = _SHAP_MODELS[shap_model_sel]

    plot_type = st.radio(
        "Plot type",
        options=["Feature Importance (bar)", "Beeswarm Summary", "Single Prediction (waterfall)"],
        horizontal=True,
        key="shap_plot_type",
    )

    plot_file_map = {
        "Feature Importance (bar)":           SHAP_DIR / f"shap_bar_{slug}.png",
        "Beeswarm Summary":                   SHAP_DIR / f"shap_summary_{slug}.png",
        "Single Prediction (waterfall)":      SHAP_DIR / f"shap_waterfall_{slug}.png",
    }

    img_path = plot_file_map[plot_type]

    if img_path.exists():
        st.image(str(img_path), use_container_width=True)

        with st.expander("How to read this plot"):
            if "bar" in plot_type.lower():
                st.markdown(
                    "**Bar chart** — each bar is a feature's average absolute SHAP value "
                    "across the test set.  Longer bar = larger average impact on model output.  \n"
                    "Features at the top are the most influential globally."
                )
            elif "beeswarm" in plot_type.lower():
                st.markdown(
                    "**Beeswarm** — each dot is one test sample.  \n"
                    "• **Horizontal position** (x-axis): SHAP value — how much this feature "
                    "pushed the prediction towards *HO soon* (right) or away (left).  \n"
                    "• **Colour**: feature value — red = high, blue = low.  \n"
                    "A cluster of red dots on the right means *high feature value increases "
                    "handover risk*."
                )
            else:
                st.markdown(
                    "**Waterfall** — explains a single positive prediction (a row where the "
                    "model predicted *HO soon*).  \n"
                    "• Bars to the **right** push the prediction higher (more likely HO).  \n"
                    "• Bars to the **left** push it lower.  \n"
                    "• The bottom of the chart shows the model's base value (average prediction "
                    "on the training set); the top shows the final output for this sample."
                )
    else:
        st.info(
            f"SHAP plots not found for **{shap_model_sel}**.  \n"
            "Run:  \n```bash\npython run_pipeline.py --phase 5\n```"
        )

    # ── On-demand re-explanation ───────────────────────────────────────────────
    st.markdown("---")
    with st.expander("⚡ Recompute SHAP explanations now (may take ~30 s)"):
        if st.button("Run Phase 5 — SHAP Explanation", key="run_shap_btn"):
            import subprocess, sys as _sys
            with st.spinner("Computing SHAP values…"):
                result = subprocess.run(
                    [_sys.executable, str(ROOT / "run_pipeline.py"), "--phase", "5"],
                    capture_output=True, text=True, cwd=str(ROOT),
                )
            if result.returncode == 0:
                st.success("Done! Reload the page to see updated plots.")
                st.code(result.stdout[-2000:] if len(result.stdout) > 2000
                        else result.stdout)
            else:
                st.error("SHAP computation failed.")
                st.code(result.stderr[-2000:])


# ── Tab 6: MLflow Runs ─────────────────────────────────────────────────────────

with tab_mlf:
    st.subheader("MLflow Experiment Runs")
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
        client = mlflow.tracking.MlflowClient()
        exp    = client.get_experiment_by_name("lte_handover_prediction")
        if exp is None:
            st.info("No MLflow experiment found. Run the pipeline first.")
        else:
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["metrics.test_roc_auc DESC"],
            )
            if runs:
                rows = []
                for r in runs:
                    rows.append({
                        "Run Name":   r.data.tags.get("mlflow.runName", r.info.run_id[:8]),
                        "Val F1":     r.data.metrics.get("val_f1",       "—"),
                        "Test F1":    r.data.metrics.get("test_f1",      "—"),
                        "Test AUC":   r.data.metrics.get("test_roc_auc", "—"),
                        "Status":     r.info.status,
                        "Run ID":     r.info.run_id[:8],
                    })
                mlf_df = pd.DataFrame(rows)
                st.dataframe(mlf_df, use_container_width=True)
            else:
                st.info("No completed MLflow runs found.")
    except ImportError:
        st.warning("Install mlflow (`pip install mlflow`) to see run history.")
    except Exception as e:
        st.error(f"MLflow error: {e}")

# ── Tab 6: Mobility Map ────────────────────────────────────────────────────────

with tab_map:
    st.subheader("UE Mobility Tracks")
    fig_map = px.scatter(
        df_view, x="pos_x", y="pos_y",
        color="ue_id", symbol="ue_id", opacity=0.4,
        labels={"pos_x": "X (m)", "pos_y": "Y (m)", "ue_id": "UE"},
        title="UE Positions (all selected timesteps)",
        height=460,
    )
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
