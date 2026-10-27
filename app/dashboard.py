"""
Phase 5 — Streamlit Dashboard

Features:
  • Champion model banner  (from models/champion/metadata.json)
  • Live prediction on uploaded raw measurements (src/serving.py — the same
    feature path as predict.py, results joined back by row index)
  • Proactive-HO tab: closed-loop experiment and generalization results
  • Network KPI monitor    (RSRP, SINR, RSRQ, CQI per UE over time)
  • Handover event timeline
  • Real-time handover risk gauge
  • Model comparison table
  • MLflow runs table       (reads mlruns/ experiment)
  • SHAP explanation tab    (pre-computed bar / beeswarm / waterfall plots)
  • Interactive filters: model selector, UE filter, time range, risk threshold
"""

import io
import json
import sys
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

# Allow imports from src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.serving import load_model, predict_scaled, score_measurements  # noqa: E402

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


MODEL_NAMES = [
    "XGBoost", "Random Forest", "Logistic Regression",
    "LSTM", "GRU", "Stacking Ensemble", "Rule (A3 gap)",
]


@st.cache_resource
def load_models():
    models = {}
    for name in MODEL_NAMES:
        try:
            models[name] = load_model(name)
        except Exception:
            pass
    return models


@st.cache_data
def load_report(name: str):
    path = ROOT / "reports" / name
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_champion_meta():
    path = ROOT / "models" / "champion" / "metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_resource
def load_scaler():
    path = ROOT / "models" / "scaler.pkl"
    return joblib.load(path) if path.exists() else None


# ── Live-prediction helpers ────────────────────────────────────────────────────

# Raw measurement columns a live CSV must provide
RAW_SIGNAL_COLS = [
    "rsrp_serving", "rsrq_serving", "sinr", "cqi",
    "rsrp_neighbor", "rsrq_neighbor",
    "l3_rsrp_serving", "l3_rsrp_neighbor",
    "rsrp_diff", "ue_speed", "cell_load_pct",
]


def score_live(df_input: pd.DataFrame, model) -> pd.DataFrame:
    """
    Score raw measurements and return df_input with risk_pct / warm-up columns.

    score_measurements() groups rows by UE, orders them by timestamp and
    returns scores on df_input's own index, so any row order is safe.
    """
    scores = score_measurements(df_input, model)
    out = df_input.join(scores)
    out["risk_pct"] = (out["prob"] * 100).round(1)
    return out


# ── Bootstrap ──────────────────────────────────────────────────────────────────

raw, test, meta = load_data()
feat_cols = [c for c in meta["feature_cols"] if c in test.columns]
models    = load_models()
champion  = load_champion_meta()
scaler    = load_scaler()

# ── Champion banner ────────────────────────────────────────────────────────────

if champion:
    _test = (f"Test F1: {champion['test_f1']:.4f}  |  Test AUC: {champion['test_roc_auc']:.4f}  |  "
             if champion.get("test_f1") is not None else "")
    st.success(
        f"🏆 **Champion model:** {champion.get('display_name', champion.get('model_name', '?'))}  |  "
        f"chosen on validation ({champion.get('promotion_criterion', '')}) = "
        f"{champion.get('champion_score', 0):.4f}  |  {_test}"
        f"Threshold: {champion.get('threshold', 0.5):.2f}  |  "
        f"Promoted: {str(champion.get('promoted_at',''))[:19].replace('T',' ')} UTC"
    )

# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.title("📡 LTE Handover Prediction")
st.sidebar.markdown("---")

available_models = [m for m in MODEL_NAMES if m in models] or ["No models found"]
_default = (champion or {}).get("display_name")
selected_model = st.sidebar.selectbox(
    "Active prediction model", options=available_models,
    index=available_models.index(_default) if _default in available_models else 0,
)

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

_tuned = models[selected_model].threshold if selected_model in models else 0.5
prob_threshold = st.sidebar.slider(
    "Risk threshold", 0.05, 0.95, float(round(_tuned, 2)), 0.01,
    help="Defaults to the threshold tuned on validation for the selected model.",
)

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

(tab_kpi, tab_ho, tab_risk, tab_pred, tab_cmp, tab_shap, tab_mlf, tab_map,
 tab_pho) = st.tabs(
    ["📊 Radio KPIs", "🔁 HO Timeline", "⚠️ Risk", "🔮 Live Prediction",
     "📋 Model Comparison", "🔍 SHAP Explanations", "🧪 MLflow Runs", "🗺️ Mobility Map",
     "🛰️ Proactive HO"]
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

    if models and selected_model in models:
        test_ues = test[test["ue_id"].isin(selected_ues)].copy()
        risk_val = 0.0
        if len(test_ues):
            # Test rows are already scaled; predict_scaled builds per-UE
            # sequences for LSTM / GRU / Stacking and returns probabilities
            # on test_ues' own index.
            test_ues["risk"] = predict_scaled(models[selected_model], test_ues, feat_cols)
            risk_val = float(test_ues["risk"].mean())
            pivot = test_ues.pivot_table(index="ue_id", columns="timestamp",
                                         values="risk", aggfunc="mean")
            fig_heat = px.imshow(
                pivot[pivot.columns[::5]],
                color_continuous_scale="RdYlGn_r", zmin=0, zmax=1,
                labels=dict(x="Timestamp", y="UE ID", color="Risk"),
                title=f"Handover Risk Heatmap — {selected_model} (test period)",
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

# ── Tab 4: Live Prediction ─────────────────────────────────────────────────────

with tab_pred:
    st.subheader("🔮 Live Prediction")
    st.markdown(
        "Score raw UE measurements with the **active model** selected in the sidebar.  \n"
        "Lags, rolling statistics and deltas are engineered per UE from the uploaded "
        "rows (same code path as `predict.py`). The first 3 rows of each UE are "
        "*warm-up* rows: they have no history yet and are not scored."
    )

    if not models or selected_model not in models:
        st.warning("No models loaded. Run `python run_pipeline.py` first.")
    else:
        model = models[selected_model]
        input_mode = st.radio("Input method", ["📂 Upload CSV", "🎛️ Manual input"],
                              horizontal=True, key="pred_input_mode")

        if input_mode == "📂 Upload CSV":
            st.markdown(
                "**Required columns:** " + ", ".join(f"`{c}`" for c in RAW_SIGNAL_COLS) +
                ", plus `ue_id` and `timestamp` for multi-UE files.  \n"
                "Ready-made files: `data/test_scenarios/*.csv` (any row order works)."
            )
            sample = ROOT / "data" / "test_scenarios" / "default_ues5_steps300_seed42.csv"
            if sample.exists():
                st.download_button("⬇️ Download a sample measurement file",
                                   data=sample.read_bytes(), file_name=sample.name,
                                   mime="text/csv")

            uploaded = st.file_uploader("Upload measurements CSV", type=["csv"],
                                        key="pred_upload")
            if uploaded is not None:
                try:
                    df_up = pd.read_csv(uploaded)
                    st.markdown(f"**Loaded:** {len(df_up)} rows × {len(df_up.columns)} columns")
                    missing = [c for c in RAW_SIGNAL_COLS if c not in df_up.columns]
                    if missing:
                        st.error(f"Missing required columns: {missing}")
                    else:
                        with st.spinner(f"Running {selected_model}…"):
                            res = score_live(df_up, model)
                        thr_pct = prob_threshold * 100
                        scored  = res[~res["warmup"]]
                        res["decision"] = np.where(
                            res["warmup"], "warm-up",
                            np.where(res["risk_pct"] >= thr_pct, "⚠️ HANDOVER", "✓ ok"))

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Rows scored", len(scored))
                        m2.metric("Warm-up rows", int(res["warmup"].sum()))
                        m3.metric("⚠️ HO predicted", int((scored["risk_pct"] >= thr_pct).sum()))
                        m4.metric("Avg risk", f"{scored['risk_pct'].mean():.1f}%")

                        show = [c for c in ["ue_id", "timestamp", "rsrp_serving",
                                            "rsrp_neighbor", "sinr", "risk_pct", "decision",
                                            "handover_soon"] if c in res.columns]
                        st.dataframe(res[show], use_container_width=True, height=300)

                        if "timestamp" in res.columns:
                            fig_line = go.Figure()
                            for ue_id, grp in scored.sort_values("timestamp").groupby(
                                    "ue_id" if "ue_id" in scored else lambda _: 0):
                                fig_line.add_trace(go.Scatter(
                                    x=grp["timestamp"], y=grp["risk_pct"],
                                    mode="lines", name=f"UE {ue_id}"))
                            fig_line.add_hline(y=thr_pct, line_dash="dash", line_color="red",
                                               annotation_text=f"Threshold ({thr_pct:.0f}%)")
                            fig_line.update_layout(
                                title=f"Handover Risk Over Time — {selected_model}",
                                xaxis_title="Timestamp (s)", yaxis_title="Risk (%)",
                                yaxis=dict(range=[0, 105]), height=350,
                                legend=dict(orientation="h", y=-0.2), margin=dict(t=40))
                            st.plotly_chart(fig_line, use_container_width=True)

                        st.download_button("⬇️ Download predictions CSV",
                                           data=res[show].to_csv(index=False),
                                           file_name="handover_predictions.csv",
                                           mime="text/csv")
                except Exception as e:
                    st.error(f"Error processing file: {e}")

        else:
            st.markdown(
                "Enter the current measurements of one UE. The values are repeated "
                "for N consecutive seconds to give the model a (flat) history; "
                "upload a CSV to use a real trajectory."
            )
            with st.form("manual_pred_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    rsrp_s = st.slider("RSRP Serving (dBm)", -140, -44, -100)
                    rsrq_s = st.slider("RSRQ Serving (dB)", -20, -3, -12)
                    sinr   = st.slider("SINR (dB)", -20, 30, 3)
                    cqi    = st.slider("CQI", 1, 15, 6)
                with c2:
                    rsrp_n = st.slider("RSRP Neighbour (dBm)", -140, -44, -101)
                    rsrq_n = st.slider("RSRQ Neighbour (dB)", -20, -3, -13)
                    l3_s   = st.slider("L3 RSRP Serving (dBm)", -140, -44, -100)
                    l3_n   = st.slider("L3 RSRP Neighbour (dBm)", -140, -44, -100)
                with c3:
                    speed  = st.slider("UE Speed (m/s)", 0.0, 25.0, 12.0, step=0.5)
                    load   = st.slider("Cell Load (%)", 0, 100, 30)
                    n_rows = st.number_input("Consecutive identical steps", 4, 20, 10)
                submitted = st.form_submit_button("🔮 Predict", type="primary")

            if submitted:
                row = {"ue_id": 0, "rsrp_serving": rsrp_s, "rsrq_serving": rsrq_s,
                       "sinr": sinr, "cqi": cqi, "rsrp_neighbor": rsrp_n,
                       "rsrq_neighbor": rsrq_n, "l3_rsrp_serving": l3_s,
                       "l3_rsrp_neighbor": l3_n, "ue_speed": speed,
                       "rsrp_diff": rsrp_s - rsrp_n, "cell_load_pct": load}
                df_manual = pd.DataFrame([{**row, "timestamp": t} for t in range(int(n_rows))])
                res = score_live(df_manual, model)
                last_risk = float(res["risk_pct"].iloc[-1])
                thr_pct = prob_threshold * 100
                col_g, col_info = st.columns(2)
                with col_g:
                    fig_g = go.Figure(go.Indicator(
                        mode="gauge+number", value=last_risk, number={"suffix": "%"},
                        title={"text": f"Handover Risk — {selected_model}"},
                        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "darkred"},
                               "threshold": {"line": {"color": "black", "width": 4},
                                             "thickness": 0.75, "value": thr_pct}}))
                    fig_g.update_layout(height=280, margin=dict(t=30, b=10))
                    st.plotly_chart(fig_g, use_container_width=True)
                with col_info:
                    st.markdown("### " + ("⚠️ HANDOVER LIKELY" if last_risk >= thr_pct
                                          else "✓ No handover expected"))
                    st.markdown(f"**Risk:** `{last_risk:.1f}%` · **Threshold:** `{thr_pct:.0f}%`")
                    st.markdown(f"- L3 gap (neighbour − serving): `{l3_n - l3_s:+.1f} dB` "
                                f"(A3 triggers above +3 dB)  \n- SINR: `{sinr} dB`  \n"
                                f"- Speed: `{speed} m/s`")


# ── Tab 5: Model Comparison ────────────────────────────────────────────────────

with tab_cmp:
    st.subheader("Model Comparison (test split)")
    ev, op = load_report("evaluation.json"), load_report("operational.json")
    if ev:
        rows = []
        for name, m in ev["models"].items():
            o = (op or {}).get("models", {}).get(name, {})
            rows.append({
                "Model": name, "Threshold": m["threshold"], "Precision": m["precision"],
                "Recall": m["recall"], "F1": m["f1"], "F1 @0.5": m["f1_0.5"],
                "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"],
                "Event recall @3s": o.get("event_recall@3s"),
                "Mean lead (s)": o.get("mean_lead_s"),
                "False-alarm time": o.get("false_alarm_time_share"),
                f"Event recall @≤{(op or {}).get('false_alarm_time_budget', 0.05):.0%} false-alarm time":
                    (op or {}).get("event_recall_at_budget", {}).get(name),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
        st.caption(f"{ev['n_rows']} test rows (same rows for every model), positive rate "
                   f"{ev['pos_rate']:.3f}. Thresholds tuned on validation. "
                   "Rule (A3 gap) is a no-learning baseline.")
        for img in ["roc_curves.png", "pr_curves.png"]:
            path = ROOT / "reports" / img
            if path.exists():
                st.image(str(path), use_column_width=True)
    else:
        st.info("Run `python run_pipeline.py` to generate evaluation results.")

# ── Tab 5: SHAP Explanations ───────────────────────────────────────────────────

with tab_shap:
    st.subheader("SHAP Feature Explanations")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) shows how much each feature "
        "contributed to a model's prediction — both globally (average over the "
        "test set) and for individual predictions.\n\n"
        "**Supported models:** Logistic Regression · Random Forest · XGBoost · Stacking Ensemble  \n"
        "*(Stacking Ensemble SHAP uses KernelExplainer over the 3 meta-features: XGB prob, RF prob, LSTM prob)*  \n"
        "Run `python run_pipeline.py --phase 5` to regenerate plots after retraining."
    )

    SHAP_DIR = ROOT / "reports" / "shap"

    _SHAP_MODELS = {
        "Logistic Regression": "logistic_regression",
        "Random Forest":       "random_forest",
        "XGBoost":             "xgboost",
        "Stacking Ensemble":   "stacking_ensemble",
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
        st.image(str(img_path), use_column_width=True)

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
                        "Val F1 (tuned)": r.data.metrics.get("val_f1_tuned", "—"),
                        "Val AUC":    r.data.metrics.get("val_roc_auc",  "—"),
                        "Test F1":    r.data.metrics.get("test_f1",      "—"),
                        "Test AUC":   r.data.metrics.get("test_roc_auc", "—"),
                        "Test PR-AUC": r.data.metrics.get("test_pr_auc", "—"),
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

# ── Tab 9: Proactive handover (closed loop + generalization) ───────────────────

with tab_pho:
    st.subheader("🛰️ Does prediction improve the network?")
    st.markdown(
        "The closed-loop experiment (`src/closed_loop.py`) re-runs the simulator on "
        "unseen seeds with the champion model **inside** the handover loop. Each "
        "policy sets the time-to-trigger (TTT) of every UE at every second; mobility "
        "and fading are identical across policies, so differences come from the "
        "handover decisions only."
    )
    cl = load_report("closed_loop.json")
    if cl:
        rows = [{"Policy": p, **{k: f"{v['mean']:.3g} ± {v['std']:.2g}"
                                 for k, v in m.items()}}
                for p, m in cl["policies"].items()]
        st.dataframe(pd.DataFrame(rows).set_index("Policy"), use_container_width=True)
        st.caption(cl.get("note", ""))
        img = ROOT / "docs" / "img" / "closed_loop.png"
        if img.exists():
            st.image(str(img), use_column_width=True)
    else:
        st.info("Run `python -m src.closed_loop` to produce reports/closed_loop.json.")

    st.markdown("---")
    st.subheader("Generalization to unseen networks")
    gen = load_report("generalization.json")
    if gen:
        st.dataframe(pd.DataFrame(gen["rows"]).set_index(["dataset", "model"]),
                     use_container_width=True)
    else:
        st.info("Run `python -m src.generalization` to produce reports/generalization.json.")

st.caption("LTE Handover Prediction Dashboard")
