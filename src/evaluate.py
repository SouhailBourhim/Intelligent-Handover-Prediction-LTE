"""
Phase 4 — Evaluation

Metrics per model:
  - Precision, Recall, F1-score (at 0.5 threshold)
  - ROC-AUC
  - Confusion matrix
  - Classification report

Generates a human-readable comparison report saved to reports/evaluation.txt
and a plots file reports/roc_curves.png.
"""

import json
import numpy as np
import pandas as pd
import joblib
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

ROOT          = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
REPORTS_DIR   = ROOT / "reports"
LABEL         = "handover_soon"
SEQ_LEN       = 10


# ── Load utilities ─────────────────────────────────────────────────────────────

def load_test(feat_cols: list[str] | None = None):
    with open(PROCESSED_DIR / "meta.json") as f:
        meta = json.load(f)
    feat_cols = feat_cols or meta["feature_cols"]
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    feat_cols = [c for c in feat_cols if c in test.columns]
    return test, feat_cols


def get_sklearn_probs(model, X_test: np.ndarray) -> np.ndarray:
    return model.predict_proba(X_test)[:, 1]


def get_lstm_probs(feat_cols: list[str], test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    from src.models import LSTMClassifier, SequenceDataset
    device = torch.device("cpu")
    model = LSTMClassifier(input_size=len(feat_cols))
    model.load_state_dict(torch.load(MODELS_DIR / "lstm.pt", map_location=device))
    model.eval()

    ds = SequenceDataset(test_df, feat_cols, seq_len=SEQ_LEN)
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)

    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            probs.extend(torch.sigmoid(logits).numpy())
            labels.extend(yb.numpy().astype(int))

    return np.array(probs), np.array(labels)


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

def plot_roc_curves(results: dict, y_labels: dict):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = {"Logistic Regression": "#4e79a7",
              "Random Forest": "#f28e2b",
              "LSTM": "#59a14f"}

    for name, res in results.items():
        y_true = y_labels[name]
        y_prob = res["probs"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['metrics']['roc_auc']:.3f})",
                color=colors.get(name, "gray"), lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Handover Prediction", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = REPORTS_DIR / "roc_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ROC plot saved → {path}")


def plot_confusion_matrices(results: dict, y_labels: dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, res) in zip(axes, results.items()):
        cm = np.array(res["metrics"]["cm"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No HO", "HO Soon"],
                    yticklabels=["No HO", "HO Soon"])
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    path = REPORTS_DIR / "confusion_matrices.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix plot saved → {path}")


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
        for l in m["report"].splitlines():
            lines.append("    " + l)
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
        "  • LSTM models the temporal evolution of radio metrics directly,",
        "    allowing it to detect degrading RSRP trends before the A3",
        "    event fires — this makes it the most suitable model for",
        "    proactive handover prediction in production.",
        "",
        "  RECOMMENDATION FOR DEPLOYMENT:",
        "    Use Random Forest for low-latency inference with explainability;",
        "    LSTM when historical context (last 10 s) is available.",
        "=" * 65,
    ]

    report_path = REPORTS_DIR / "evaluation.txt"
    report_path.write_text("\n".join(lines))
    print(f"  Report saved → {report_path}")
    print("\n".join(lines))


# ── Main ───────────────────────────────────────────────────────────────────────

def run_evaluation():
    print("Phase 4 — Evaluation")
    test, feat_cols = load_test()
    X_test = test[feat_cols].fillna(0).values
    y_test = test[LABEL].values

    results   = {}
    y_labels  = {}

    # Logistic Regression
    print("  Evaluating Logistic Regression...")
    lr = joblib.load(MODELS_DIR / "logistic_regression.pkl")
    lr_probs = get_sklearn_probs(lr, X_test)
    results["Logistic Regression"] = {
        "metrics": compute_metrics(y_test, lr_probs),
        "probs":   lr_probs,
    }
    y_labels["Logistic Regression"] = y_test

    # Random Forest
    print("  Evaluating Random Forest...")
    rf = joblib.load(MODELS_DIR / "random_forest.pkl")
    rf_probs = get_sklearn_probs(rf, X_test)
    results["Random Forest"] = {
        "metrics": compute_metrics(y_test, rf_probs),
        "probs":   rf_probs,
    }
    y_labels["Random Forest"] = y_test

    # LSTM (builds its own sequences from test_df)
    print("  Evaluating LSTM...")
    lstm_probs, lstm_labels = get_lstm_probs(feat_cols, test)
    results["LSTM"] = {
        "metrics": compute_metrics(lstm_labels, lstm_probs),
        "probs":   lstm_probs,
    }
    y_labels["LSTM"] = lstm_labels

    # Plots + report
    plot_roc_curves(results, y_labels)
    plot_confusion_matrices(results, y_labels)
    write_report(results)

    print("\nPhase 4 complete.\n")
    return results


if __name__ == "__main__":
    run_evaluation()
