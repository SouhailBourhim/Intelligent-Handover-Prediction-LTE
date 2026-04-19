"""
Phase 4 — Evaluation

Metrics per model:
  - Precision, Recall, F1-score (at 0.5 threshold)
  - ROC-AUC
  - Confusion matrix
  - Classification report

Six models evaluated:
  Logistic Regression · Random Forest · XGBoost
  LSTM · GRU · Stacking Ensemble

Generates:
  reports/evaluation.txt
  reports/roc_curves.png
  reports/confusion_matrices.png

MLflow test metrics + plot artifacts are logged into the runs created
during training (run IDs loaded from models/mlflow_run_ids.json).
Wrapped in try/except — runs without MLflow installed.
"""

import json
import os
import numpy as np
import pandas as pd
import joblib

# Must be set before torch (loaded via src.models) to avoid OpenMP segfault
# when running XGBoost after torch on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve,
)

# ── MLflow (optional) ──────────────────────────────────────────────────────────
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

ROOT          = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
REPORTS_DIR   = ROOT / "reports"
LABEL         = "handover_soon"
SEQ_LEN       = 10

MLF_TRACKING  = str(ROOT / "mlruns")
MLF_RUN_IDS   = MODELS_DIR / "mlflow_run_ids.json"


# ── MLflow helpers ─────────────────────────────────────────────────────────────

def _load_run_ids() -> dict:
    try:
        if MLF_RUN_IDS.exists():
            with open(MLF_RUN_IDS) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _mlf_log_test_metrics(run_id: str, metrics: dict, artifacts: list[Path]):
    """Resume a training run and append test metrics + plot artifacts."""
    if not MLFLOW_AVAILABLE or not run_id:
        return
    try:
        mlflow.set_tracking_uri(MLF_TRACKING)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("test_precision", metrics["precision"])
            mlflow.log_metric("test_recall",    metrics["recall"])
            mlflow.log_metric("test_f1",        metrics["f1"])
            mlflow.log_metric("test_roc_auc",   metrics["roc_auc"])
            for path in artifacts:
                if path.exists():
                    mlflow.log_artifact(str(path))
    except Exception:
        pass


# ── Load utilities ─────────────────────────────────────────────────────────────

def load_test(feat_cols: list[str] | None = None):
    with open(PROCESSED_DIR / "meta.json") as f:
        meta = json.load(f)
    feat_cols = feat_cols or meta["feature_cols"]
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    feat_cols = [c for c in feat_cols if c in test.columns]
    return test, feat_cols


def get_sklearn_probs(model, X_test: np.ndarray) -> np.ndarray:
    from xgboost import XGBClassifier
    if isinstance(model, XGBClassifier):
        X_test = np.asarray(X_test, dtype=np.float32)
    return model.predict_proba(X_test)[:, 1]


def _get_seq_model_probs(model_cls, weight_path: Path,
                          feat_cols: list[str],
                          test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Generic sequence-model evaluation helper (LSTM or GRU)."""
    from src.models import SequenceDataset, _build_seq_row_indices
    device = torch.device("cpu")
    net = model_cls(input_size=len(feat_cols))
    net.load_state_dict(torch.load(weight_path, map_location=device))
    net.eval()

    ds     = SequenceDataset(test_df, feat_cols, seq_len=SEQ_LEN)
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)

    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            probs.extend(torch.sigmoid(net(xb)).numpy())

    row_indices = _build_seq_row_indices(test_df, seq_len=SEQ_LEN)
    labels = test_df.loc[row_indices, LABEL].values.astype(int)
    return np.array(probs), labels


def get_lstm_probs(feat_cols: list[str], test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    from src.models import LSTMClassifier
    return _get_seq_model_probs(
        LSTMClassifier, MODELS_DIR / "lstm.pt", feat_cols, test_df
    )


def get_gru_probs(feat_cols: list[str], test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    from src.models import GRUClassifier
    return _get_seq_model_probs(
        GRUClassifier, MODELS_DIR / "gru.pt", feat_cols, test_df
    )


def get_stacking_probs(
    feat_cols: list[str],
    test_df:   pd.DataFrame,
    xgb_model,
    rf_model,
    lstm_probs: np.ndarray,
    lstm_row_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stacking uses LSTM probs as anchor; XGB/RF are restricted to the same rows.
    """
    test_sub  = test_df.loc[lstm_row_indices].copy()
    X_sub     = test_sub[feat_cols].fillna(0).values
    y_sub     = test_sub[LABEL].values.astype(int)

    xgb_p = xgb_model.predict_proba(np.asarray(X_sub, dtype=np.float32))[:, 1]
    rf_p  = rf_model.predict_proba(X_sub)[:, 1]

    meta_X = np.column_stack([xgb_p, rf_p, lstm_probs])
    stack  = joblib.load(MODELS_DIR / "stacking_ensemble.pkl")
    probs  = stack.predict_proba(meta_X)[:, 1]
    return probs, y_sub


# ── Metrics computation ────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true,    y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true,        y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        "cm":        confusion_matrix(y_true, y_pred).tolist(),
        "report":    classification_report(y_true, y_pred, zero_division=0),
    }


# ── ROC plot ───────────────────────────────────────────────────────────────────

_COLORS = {
    "Logistic Regression": "#4e79a7",
    "Random Forest":       "#f28e2b",
    "XGBoost":             "#e15759",
    "LSTM":                "#59a14f",
    "GRU":                 "#76b7b2",
    "Stacking Ensemble":   "#b07aa1",
}


def plot_roc_curves(results: dict, y_labels: dict):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, res in results.items():
        y_true = y_labels[name]
        y_prob = res["probs"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr,
                label=f"{name} (AUC={res['metrics']['roc_auc']:.3f})",
                color=_COLORS.get(name, "gray"), lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Handover Prediction", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = REPORTS_DIR / "roc_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ROC plot saved → {path}")
    return path


def plot_confusion_matrices(results: dict, y_labels: dict):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(n * 4.5, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = np.array(res["metrics"]["cm"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No HO", "HO Soon"],
                    yticklabels=["No HO", "HO Soon"])
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    path = REPORTS_DIR / "confusion_matrices.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix plot saved → {path}")
    return path


# ── Text report ────────────────────────────────────────────────────────────────

def write_report(results: dict):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "=" * 65,
        "  EVALUATION REPORT — Intelligent Handover Prediction (LTE)",
        "=" * 65,
        "",
    ]

    best_model = max(results, key=lambda k: results[k]["metrics"]["f1"])
    best_auc   = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])

    for name, res in results.items():
        m = res["metrics"]
        lines += [
            f"── {name} ──",
            f"  Precision : {m['precision']:.4f}",
            f"  Recall    : {m['recall']:.4f}",
            f"  F1-score  : {m['f1']:.4f}",
            f"  ROC-AUC   : {m['roc_auc']:.4f}",
            "",
            "  Classification Report:",
        ]
        for ln in m["report"].splitlines():
            lines.append("    " + ln)
        lines.append("")

    lines += [
        "=" * 65,
        "  COMPARISON SUMMARY",
        "=" * 65,
        "",
        f"  Best F1      : {best_model}  ({results[best_model]['metrics']['f1']:.4f})",
        f"  Best ROC-AUC : {best_auc}  ({results[best_auc]['metrics']['roc_auc']:.4f})",
        "",
        "  ANALYSIS:",
        "",
        "  • Logistic Regression is a fast linear baseline. It struggles",
        "    with the non-linear RSRP/SINR boundary dynamics but provides",
        "    a useful lower bound and interpretable coefficients.",
        "",
        "  • Random Forest captures feature interactions (RSRP gap,",
        "    rolling trends, delta features) that the linear model misses.",
        "    Ensemble bagging reduces variance on the rare positive class.",
        "    Feature importances expose which KPIs drive the prediction.",
        "",
        "  • XGBoost applies gradient boosting on the same feature set,",
        "    combining early stopping and scale_pos_weight for best",
        "    precision/recall balance among tree-based models.",
        "",
        "  • LSTM models the temporal evolution of radio metrics directly,",
        "    allowing it to detect degrading RSRP trends before the A3",
        "    event fires — suitable for proactive handover prediction.",
        "",
        "  • GRU is a lighter recurrent alternative to LSTM. Comparable",
        "    accuracy with fewer parameters and faster inference.",
        "",
        "  • Stacking Ensemble combines XGBoost, Random Forest, and LSTM",
        "    via a meta Logistic Regression, leveraging complementary",
        "    decision boundaries for higher overall AUC.",
        "",
        "  RECOMMENDATION FOR DEPLOYMENT:",
        "    Use XGBoost or Random Forest for low-latency inference;",
        "    LSTM/GRU when historical context (last 10 s) is available;",
        "    Stacking Ensemble when maximum AUC is the priority.",
        "=" * 65,
    ]

    report_path = REPORTS_DIR / "evaluation.txt"
    report_path.write_text("\n".join(lines))
    print(f"  Report saved → {report_path}")
    print("\n".join(lines))


# ── Main ───────────────────────────────────────────────────────────────────────

def run_evaluation():
    print("Phase 4 — Evaluation")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    test, feat_cols = load_test()
    X_test = test[feat_cols].fillna(0).values
    y_test = test[LABEL].values

    results  = {}
    y_labels = {}
    run_ids  = _load_run_ids()

    # ── Logistic Regression ────────────────────────────────────────────────────
    print("  Evaluating Logistic Regression...")
    lr = joblib.load(MODELS_DIR / "logistic_regression.pkl")
    lr_probs = get_sklearn_probs(lr, X_test)
    results["Logistic Regression"] = {
        "metrics": compute_metrics(y_test, lr_probs), "probs": lr_probs,
    }
    y_labels["Logistic Regression"] = y_test

    # ── Random Forest ──────────────────────────────────────────────────────────
    print("  Evaluating Random Forest...")
    rf = joblib.load(MODELS_DIR / "random_forest.pkl")
    rf_probs = get_sklearn_probs(rf, X_test)
    results["Random Forest"] = {
        "metrics": compute_metrics(y_test, rf_probs), "probs": rf_probs,
    }
    y_labels["Random Forest"] = y_test

    # ── XGBoost ────────────────────────────────────────────────────────────────
    print("  Evaluating XGBoost...")
    xgb = joblib.load(MODELS_DIR / "xgboost.pkl")
    xgb_probs = get_sklearn_probs(xgb, X_test)
    results["XGBoost"] = {
        "metrics": compute_metrics(y_test, xgb_probs), "probs": xgb_probs,
    }
    y_labels["XGBoost"] = y_test

    # ── LSTM ───────────────────────────────────────────────────────────────────
    print("  Evaluating LSTM...")
    lstm_probs, lstm_labels = get_lstm_probs(feat_cols, test)
    from src.models import _build_seq_row_indices
    lstm_row_indices = _build_seq_row_indices(test, seq_len=SEQ_LEN)
    results["LSTM"] = {
        "metrics": compute_metrics(lstm_labels, lstm_probs), "probs": lstm_probs,
    }
    y_labels["LSTM"] = lstm_labels

    # ── GRU ────────────────────────────────────────────────────────────────────
    print("  Evaluating GRU...")
    gru_probs, gru_labels = get_gru_probs(feat_cols, test)
    results["GRU"] = {
        "metrics": compute_metrics(gru_labels, gru_probs), "probs": gru_probs,
    }
    y_labels["GRU"] = gru_labels

    # ── Stacking Ensemble ──────────────────────────────────────────────────────
    print("  Evaluating Stacking Ensemble...")
    stack_probs, stack_labels = get_stacking_probs(
        feat_cols, test, xgb, rf, lstm_probs, lstm_row_indices
    )
    results["Stacking Ensemble"] = {
        "metrics": compute_metrics(stack_labels, stack_probs),
        "probs":   stack_probs,
    }
    y_labels["Stacking Ensemble"] = stack_labels

    # ── Plots + report ─────────────────────────────────────────────────────────
    roc_path = plot_roc_curves(results, y_labels)
    cm_path  = plot_confusion_matrices(results, y_labels)
    write_report(results)

    # ── MLflow: log test metrics + artifacts into training runs ────────────────
    plot_artifacts = [roc_path, cm_path]
    mlf_name_map = {
        "Logistic Regression": "Logistic Regression",
        "Random Forest":       "Random Forest",
        "XGBoost":             "XGBoost",
        "LSTM":                "LSTM",
        "GRU":                 "GRU",
        "Stacking Ensemble":   "Stacking Ensemble",
    }
    for display_name, run_key in mlf_name_map.items():
        if display_name in results:
            rid = run_ids.get(run_key, "")
            _mlf_log_test_metrics(
                rid,
                results[display_name]["metrics"],
                plot_artifacts,
            )

    print("\nPhase 4 complete.\n")
    return results


if __name__ == "__main__":
    run_evaluation()
