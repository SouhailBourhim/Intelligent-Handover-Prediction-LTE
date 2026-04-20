"""
Phase 5 — SHAP Explanation

Computes and saves SHAP-based feature importance plots for the three
tree / linear models (fast exact explainers):

  Model                  Explainer used
  ─────────────────────  ──────────────────────────────────────────
  Logistic Regression    shap.LinearExplainer
  Random Forest          shap.TreeExplainer  (class-1 SHAP values)
  XGBoost                shap.TreeExplainer

LSTM, GRU, and Stacking Ensemble are excluded: DeepExplainer /
KernelExplainer are orders of magnitude slower and produce noisier
values that add little interpretability over the tree models.

Outputs (saved to reports/shap/):
  shap_bar_<model>.png       — top-20 features ranked by mean |SHAP|
  shap_summary_<model>.png   — beeswarm of top-20 features
  shap_waterfall_<model>.png — single positive-class prediction explained

The explain pipeline can be run standalone:
    python run_pipeline.py --phase 5
or is called automatically at the end of Phase 4.

MLflow integration (optional):
  If MLflow run IDs are present in models/mlflow_run_ids.json, each
  set of plots is logged as artifacts into the corresponding training run.
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT          = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
REPORTS_DIR   = ROOT / "reports" / "shap"
LABEL         = "handover_soon"
TOP_N         = 20      # features shown in summary / bar plots
BG_SAMPLES    = 500     # background / explanation sample size
KERNEL_BG     = 200     # background samples for KernelExplainer (stacking)

# MLflow (optional)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

MLF_TRACKING = f"sqlite:///{ROOT / 'mlflow.db'}"
MLF_RUN_IDS  = MODELS_DIR / "mlflow_run_ids.json"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_test(feat_cols=None):
    with open(PROCESSED_DIR / "meta.json") as f:
        meta = json.load(f)
    feat_cols = feat_cols or meta["feature_cols"]
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    feat_cols = [c for c in feat_cols if c in test.columns]
    X = test[feat_cols].fillna(0).values
    y = test[LABEL].values
    return X, y, feat_cols, test


def _sample(X, y, n=BG_SAMPLES, seed=42):
    """Return a stratified random sample for background / explanation."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos = min(n // 2, len(pos_idx))
    n_neg = min(n - n_pos, len(neg_idx))
    idx = np.concatenate([
        rng.choice(pos_idx, n_pos, replace=False),
        rng.choice(neg_idx, n_neg, replace=False),
    ])
    return X[idx], y[idx], idx


def _first_positive(y, fallback=0) -> int:
    """Index of the first positive label in the sample, or fallback."""
    pos = np.where(y == 1)[0]
    return int(pos[0]) if len(pos) > 0 else fallback


def _save_fig(path: Path, dpi: int = 150):
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close("all")
    print(f"    Saved → {path.relative_to(ROOT)}")


def _mlf_log_artifacts(run_key: str, paths: list[Path]):
    if not MLFLOW_AVAILABLE:
        return
    try:
        run_ids: dict = {}
        if MLF_RUN_IDS.exists():
            with open(MLF_RUN_IDS) as f:
                run_ids = json.load(f)
        rid = run_ids.get(run_key, "")
        if not rid:
            return
        mlflow.set_tracking_uri(MLF_TRACKING)
        with mlflow.start_run(run_id=rid):
            for p in paths:
                if p.exists():
                    mlflow.log_artifact(str(p), artifact_path="shap")
    except Exception:
        pass


# ── Plot helpers ───────────────────────────────────────────────────────────────

def _bar_plot(shap_vals: np.ndarray, feat_cols: list[str],
              model_name: str) -> Path:
    """Horizontal bar chart — mean |SHAP| per feature, top-N."""
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:TOP_N][::-1]   # ascending for barh

    fig, ax = plt.subplots(figsize=(8, max(4, TOP_N * 0.32)))
    colors = plt.cm.RdBu_r(np.linspace(0.15, 0.85, len(order)))
    ax.barh(
        [feat_cols[i] for i in order],
        mean_abs[order],
        color=colors,
    )
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(f"Global Feature Importance — {model_name}", fontsize=12)
    ax.grid(axis="x", alpha=0.3)

    path = REPORTS_DIR / f"shap_bar_{model_name.lower().replace(' ', '_')}.png"
    _save_fig(path)
    return path


def _summary_plot(shap_vals: np.ndarray, X_sample: np.ndarray,
                  feat_cols: list[str], model_name: str) -> Path:
    """Beeswarm summary — top-N features coloured by feature value."""
    import shap

    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:TOP_N]

    shap_top = shap_vals[:, top_idx]
    X_top    = X_sample[:, top_idx]
    names    = [feat_cols[i] for i in top_idx]

    expl = shap.Explanation(
        values          = shap_top,
        base_values     = np.zeros(len(shap_top)),
        data            = X_top,
        feature_names   = names,
    )

    plt.figure(figsize=(9, max(4, TOP_N * 0.35)))
    shap.plots.beeswarm(expl, show=False, max_display=TOP_N)
    plt.title(f"SHAP Beeswarm — {model_name}", fontsize=12, pad=14)

    path = REPORTS_DIR / f"shap_summary_{model_name.lower().replace(' ', '_')}.png"
    _save_fig(path)
    return path


def _waterfall_plot(explainer, X_sample: np.ndarray, y_sample: np.ndarray,
                    feat_cols: list[str], model_name: str) -> Path:
    """Waterfall plot for a single positive-class prediction."""
    import shap

    i = _first_positive(y_sample)

    # Re-compute with Explanation object for waterfall
    sv = explainer(X_sample[i : i + 1])

    # Unpack multi-output (RF returns shape [n, feats, 2])
    if sv.values.ndim == 3:
        sv_single = shap.Explanation(
            values        = sv.values[0, :, 1],
            base_values   = sv.base_values[0, 1],
            data          = sv.data[0],
            feature_names = feat_cols,
        )
    else:
        sv_single = shap.Explanation(
            values        = sv.values[0],
            base_values   = float(sv.base_values[0]) if np.ndim(sv.base_values) > 0 else float(sv.base_values),
            data          = sv.data[0],
            feature_names = feat_cols,
        )

    plt.figure(figsize=(9, max(5, TOP_N * 0.38)))
    shap.plots.waterfall(sv_single, max_display=TOP_N, show=False)
    plt.title(f"Prediction Explanation — {model_name} (positive sample #{i})",
              fontsize=11, pad=14)

    path = REPORTS_DIR / f"shap_waterfall_{model_name.lower().replace(' ', '_')}.png"
    _save_fig(path)
    return path


# ── Per-model explain functions ────────────────────────────────────────────────

def explain_tree_model(model, X: np.ndarray, y: np.ndarray,
                       feat_cols: list[str], model_name: str,
                       cast_float32: bool = False) -> list[Path]:
    """TreeExplainer for RF or XGBoost."""
    import shap

    print(f"  [{model_name}] building TreeExplainer…")
    X_bg, y_bg, _ = _sample(X, y)
    if cast_float32:
        X_bg = X_bg.astype(np.float32)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_bg)

    # SHAP 0.45+: RF may return either a list [cls0, cls1] or a 3-D array
    # (n_samples, n_features, n_classes).  Always extract class-1 values.
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

    paths = [
        _bar_plot(shap_vals, feat_cols, model_name),
        _summary_plot(shap_vals, X_bg, feat_cols, model_name),
        _waterfall_plot(explainer, X_bg, y_bg, feat_cols, model_name),
    ]
    return paths


def explain_linear_model(model, X: np.ndarray, y: np.ndarray,
                         feat_cols: list[str], model_name: str) -> list[Path]:
    """LinearExplainer for Logistic Regression."""
    import shap

    print(f"  [{model_name}] building LinearExplainer…")
    X_bg, y_bg, _ = _sample(X, y)

    masker     = shap.maskers.Independent(X_bg, max_samples=BG_SAMPLES)
    explainer  = shap.LinearExplainer(model, masker)
    shap_vals  = explainer.shap_values(X_bg)

    paths = [
        _bar_plot(shap_vals, feat_cols, model_name),
        _summary_plot(shap_vals, X_bg, feat_cols, model_name),
        _waterfall_plot(explainer, X_bg, y_bg, feat_cols, model_name),
    ]
    return paths


# ── Stacking Ensemble: KernelExplainer on meta-features ───────────────────────

def _build_stacking_meta(feat_cols: list[str], test_df) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute meta-features [xgb_prob, rf_prob, lstm_prob] for all LSTM-valid
    rows in test_df.  The LSTM requires a full SEQ_LEN look-back window, so
    the row count is smaller than len(test_df).
    """
    import torch
    from src.models import LSTMClassifier, SequenceDataset, SEQ_LEN, _build_seq_row_indices

    xgb = joblib.load(MODELS_DIR / "xgboost.pkl")
    rf  = joblib.load(MODELS_DIR / "random_forest.pkl")

    lstm_net = LSTMClassifier(input_size=len(feat_cols))
    lstm_net.load_state_dict(torch.load(MODELS_DIR / "lstm.pt", map_location="cpu"))
    lstm_net.eval()

    ds     = SequenceDataset(test_df, feat_cols, seq_len=SEQ_LEN)
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)
    lstm_probs = []
    with torch.no_grad():
        for xb, _ in loader:
            lstm_probs.extend(torch.sigmoid(lstm_net(xb)).numpy())
    lstm_probs  = np.array(lstm_probs)
    row_indices = _build_seq_row_indices(test_df, seq_len=SEQ_LEN)

    test_sub = test_df.loc[row_indices].copy()
    X_sub    = test_sub[feat_cols].fillna(0).values

    xgb_probs = xgb.predict_proba(X_sub.astype(np.float32))[:, 1]
    rf_probs  = rf.predict_proba(X_sub)[:, 1]

    meta_X = np.column_stack([xgb_probs, rf_probs, lstm_probs]).astype(np.float32)
    y_sub  = test_sub[LABEL].values.astype(int)
    return meta_X, y_sub


def explain_stacking_ensemble(stacking_model, feat_cols: list[str], test_df) -> list[Path]:
    """
    KernelExplainer for the Stacking Ensemble.

    The ensemble's "features" are the three base-model output probabilities
    [XGB prob, RF prob, LSTM prob], so the SHAP plots show which base model
    drives each prediction rather than which raw radio signal does.

    Uses a stratified 200-sample background subset to keep runtime acceptable
    (~30 s for nsamples=200 with 3 features).
    """
    import shap

    print("  [Stacking Ensemble] computing meta-features for test set…")
    meta_X, y_sub = _build_stacking_meta(feat_cols, test_df)

    # Stratified sample of KERNEL_BG rows for background + explanation
    rng     = np.random.default_rng(42)
    pos_idx = np.where(y_sub == 1)[0]
    neg_idx = np.where(y_sub == 0)[0]
    n_pos   = min(KERNEL_BG // 2, len(pos_idx))
    n_neg   = min(KERNEL_BG - n_pos, len(neg_idx))
    bg_idx  = np.concatenate([
        rng.choice(pos_idx, n_pos, replace=False),
        rng.choice(neg_idx, n_neg, replace=False),
    ])
    meta_bg = meta_X[bg_idx]
    y_bg    = y_sub[bg_idx]

    meta_names = ["XGB probability", "RF probability", "LSTM probability"]

    print(f"  [Stacking Ensemble] building KernelExplainer ({len(meta_bg)}-sample background)…")
    # Wrap predict_proba to return class-1 probability (scalar output)
    def _predict(x):
        return stacking_model.predict_proba(x.astype(np.float32))[:, 1]

    explainer = shap.KernelExplainer(_predict, meta_bg)
    shap_vals = explainer.shap_values(meta_bg, nsamples=200, silent=True)
    # KernelExplainer with scalar output returns a single (n, k) array
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    paths = [
        _bar_plot(shap_vals, meta_names, "Stacking Ensemble"),
        _summary_plot(shap_vals, meta_bg, meta_names, "Stacking Ensemble"),
    ]

    # Waterfall — manually construct Explanation to avoid calling explainer() again
    i       = _first_positive(y_bg)
    wf_vals = explainer.shap_values(meta_bg[i : i + 1], nsamples=200, silent=True)
    if isinstance(wf_vals, list):
        wf_vals = wf_vals[0]
    base_val = float(
        explainer.expected_value[0]
        if hasattr(explainer.expected_value, "__len__")
        else explainer.expected_value
    )
    wf_expl = shap.Explanation(
        values        = wf_vals[0],
        base_values   = base_val,
        data          = meta_bg[i],
        feature_names = meta_names,
    )
    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(wf_expl, show=False)
    plt.title(
        f"Prediction Explanation — Stacking Ensemble (positive sample #{i})",
        fontsize=11, pad=14,
    )
    wf_path = REPORTS_DIR / "shap_waterfall_stacking_ensemble.png"
    _save_fig(wf_path)
    paths.append(wf_path)

    return paths


# ── Main entry ─────────────────────────────────────────────────────────────────

def run_explanation():
    print("Phase 5 — SHAP Explanation")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, feat_cols, test_df = _load_test()
    print(f"  Test set: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"  Positive rate: {y.mean():.1%}  |  background sample: {BG_SAMPLES} rows\n")

    all_paths: dict[str, list[Path]] = {}

    # ── Logistic Regression ────────────────────────────────────────────────────
    lr_path = MODELS_DIR / "logistic_regression.pkl"
    if lr_path.exists():
        lr = joblib.load(lr_path)
        paths = explain_linear_model(lr, X, y, feat_cols, "Logistic Regression")
        all_paths["Logistic Regression"] = paths
    else:
        print("  Logistic Regression model not found — skipping.")

    # ── Random Forest ──────────────────────────────────────────────────────────
    rf_path = MODELS_DIR / "random_forest.pkl"
    if rf_path.exists():
        rf = joblib.load(rf_path)
        paths = explain_tree_model(rf, X, y, feat_cols, "Random Forest",
                                   cast_float32=False)
        all_paths["Random Forest"] = paths
    else:
        print("  Random Forest model not found — skipping.")

    # ── XGBoost ────────────────────────────────────────────────────────────────
    xgb_path = MODELS_DIR / "xgboost.pkl"
    if xgb_path.exists():
        xgb = joblib.load(xgb_path)
        paths = explain_tree_model(xgb, X, y, feat_cols, "XGBoost",
                                   cast_float32=True)
        all_paths["XGBoost"] = paths
    else:
        print("  XGBoost model not found — skipping.")

    # ── Stacking Ensemble ──────────────────────────────────────────────────────
    stack_path = MODELS_DIR / "stacking_ensemble.pkl"
    if stack_path.exists():
        try:
            stack = joblib.load(stack_path)
            paths = explain_stacking_ensemble(stack, feat_cols, test_df)
            all_paths["Stacking Ensemble"] = paths
        except Exception as e:
            print(f"  Stacking Ensemble SHAP failed: {e}")
    else:
        print("  Stacking Ensemble model not found — skipping.")

    # ── MLflow: log SHAP plots as artifacts ───────────────────────────────────
    mlf_key_map = {
        "Logistic Regression": "Logistic Regression",
        "Random Forest":       "Random Forest",
        "XGBoost":             "XGBoost",
        "Stacking Ensemble":   "Stacking Ensemble",
    }
    for display_name, run_key in mlf_key_map.items():
        if display_name in all_paths:
            _mlf_log_artifacts(run_key, all_paths[display_name])

    print("\nPhase 5 complete.")
    print(f"  SHAP plots saved to {REPORTS_DIR.relative_to(ROOT)}/")
    return all_paths


if __name__ == "__main__":
    run_explanation()
