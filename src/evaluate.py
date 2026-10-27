"""
Phase 4 — Evaluation

Seven models are scored on the held-out test split:
  Logistic Regression · Random Forest · XGBoost · LSTM · GRU ·
  Stacking Ensemble · Rule (A3 gap, no learning)

Every model is scored on the **same rows** (test rows with a full SEQ_LEN
history, which the sequence models need), so the comparison is like-for-like.

Metrics per model:
  - Precision, Recall, F1 at the 0.5 threshold and at the threshold tuned on
    validation (models/thresholds.json)
  - ROC-AUC and PR-AUC (average precision — more informative than ROC-AUC
    when positives are rare)
  - Confusion matrix at the tuned threshold
  - Event-level metrics (src/operational.py): event recall@3 s / @5 s,
    mean lead time, false alarms per UE-hour

Generates:
  reports/evaluation.txt         human-readable report
  reports/evaluation.json        all metrics (read by promotion, docs, dashboard)
  reports/operational.json       event-level metrics per model
  reports/roc_curves.png
  reports/pr_curves.png
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

from src.models import (
    GRUClassifier, LSTMClassifier, SEQ_LEN, THRESHOLDS_PATH,
    _build_seq_row_indices, score_sequence_model, stacking_meta_features,
)
from src.operational import event_metrics, operating_curve, recall_at_false_alarm_budget

# Alarm-cost budget used to compare models: at most 5 % of the time spent
# preparing a handover that does not happen
FA_BUDGET = 0.05

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

MLF_TRACKING  = f"sqlite:///{ROOT / 'mlflow.db'}"
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
            mlflow.log_metrics({
                "test_precision": metrics["precision"],
                "test_recall":    metrics["recall"],
                "test_f1":        metrics["f1"],
                "test_f1_0.5":    metrics["f1_0.5"],
                "test_roc_auc":   metrics["roc_auc"],
                "test_pr_auc":    metrics["pr_auc"],
            })
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


def load_thresholds() -> dict:
    with open(THRESHOLDS_PATH) as f:
        return {name: m["threshold"] for name, m in json.load(f).items()}


def load_seq_model(cls, name: str, n_features: int) -> torch.nn.Module:
    net = cls(input_size=n_features)
    net.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt", map_location="cpu"))
    return net.eval()


def score_all_models(df: pd.DataFrame, feat_cols: list[str]) -> dict[str, pd.Series]:
    """
    Probabilities of all seven models for every row of `df` (Series on df.index).

    Sequence models use their per-UE windows inside `df`; the stacking model
    combines XGBoost, RF and LSTM on the same rows.
    """
    X   = df[feat_cols].fillna(0).values
    idx = df.index
    lr  = joblib.load(MODELS_DIR / "logistic_regression.pkl")
    rf  = joblib.load(MODELS_DIR / "random_forest.pkl")
    xgb = joblib.load(MODELS_DIR / "xgboost.pkl")
    rule  = joblib.load(MODELS_DIR / "rule_a3_gap.pkl")
    stack = joblib.load(MODELS_DIR / "stacking_ensemble.pkl")
    lstm = load_seq_model(LSTMClassifier, "lstm", len(feat_cols))
    gru  = load_seq_model(GRUClassifier,  "gru",  len(feat_cols))

    p_lstm = score_sequence_model(lstm, df, feat_cols)
    meta_X = stacking_meta_features(df, feat_cols, xgb, rf, p_lstm.values)
    return {
        "Logistic Regression": pd.Series(lr.predict_proba(X)[:, 1], idx),
        "Random Forest":       pd.Series(rf.predict_proba(X)[:, 1], idx),
        "XGBoost":             pd.Series(xgb.predict_proba(X.astype(np.float32))[:, 1], idx),
        "LSTM":                p_lstm,
        "GRU":                 score_sequence_model(gru, df, feat_cols),
        "Stacking Ensemble":   pd.Series(stack.predict_proba(meta_X)[:, 1], idx),
        "Rule (A3 gap)":       pd.Series(rule.predict_proba(X)[:, 1], idx),
    }


# ── Metrics computation ────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """Threshold metrics at `threshold` plus F1 at 0.5 and threshold-free AUCs."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 2),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true,    y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true,        y_pred, zero_division=0), 4),
        "f1_0.5":    round(f1_score(y_true, (y_prob >= 0.5).astype(int), zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        "pr_auc":    round(average_precision_score(y_true, y_prob), 4),
        "cm":        confusion_matrix(y_true, y_pred).tolist(),
        "report":    classification_report(y_true, y_pred, zero_division=0),
    }


# ── Plots ──────────────────────────────────────────────────────────────────────

_COLORS = {
    "Logistic Regression": "#4e79a7",
    "Random Forest":       "#f28e2b",
    "XGBoost":             "#e15759",
    "LSTM":                "#59a14f",
    "GRU":                 "#76b7b2",
    "Stacking Ensemble":   "#b07aa1",
    "Rule (A3 gap)":       "#9c9c9c",
}


def plot_roc_curves(results: dict, y_true: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_true, res["probs"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={res['metrics']['roc_auc']:.3f})",
                color=_COLORS.get(name, "gray"), lw=2,
                ls="--" if name.startswith("Rule") else "-")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Handover Prediction (test)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = REPORTS_DIR / "roc_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ROC plot saved → {path}")
    return path


def plot_pr_curves(results: dict, y_true: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, res in results.items():
        prec, rec, _ = precision_recall_curve(y_true, res["probs"])
        ax.plot(rec, prec, label=f"{name} (AP={res['metrics']['pr_auc']:.3f})",
                color=_COLORS.get(name, "gray"), lw=2,
                ls="--" if name.startswith("Rule") else "-")
    ax.axhline(y_true.mean(), color="k", ls=":", lw=1, label=f"chance ({y_true.mean():.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision–Recall Curves (test)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = REPORTS_DIR / "pr_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  PR plot saved → {path}")
    return path


def plot_confusion_matrices(results: dict):
    n = len(results)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.8))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, res) in zip(axes, results.items()):
        cm = np.array(res["metrics"]["cm"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["No HO", "HO Soon"],
                    yticklabels=["No HO", "HO Soon"])
        ax.set_title(f"{name}\n(thr={res['metrics']['threshold']:.2f})", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    path = REPORTS_DIR / "confusion_matrices.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix plot saved → {path}")
    return path


# ── Text report ────────────────────────────────────────────────────────────────

def write_report(results: dict, operational: dict, n_rows: int, pos_rate: float):
    lines = [
        "=" * 72,
        "  EVALUATION REPORT — Intelligent Handover Prediction (LTE)",
        "=" * 72,
        f"  Test rows (full {SEQ_LEN}-step history, same for every model): {n_rows}"
        f"   positive rate: {pos_rate:.3f}",
        "  Thresholds were tuned on validation (models/thresholds.json).",
        "",
        f"  {'Model':<20}{'thr':>5}{'P':>8}{'R':>8}{'F1':>8}{'F1@.5':>8}"
        f"{'ROC':>8}{'PR-AUC':>8}{'EvR@3s':>8}{'FA%':>7}",
    ]
    for name, res in results.items():
        m, o = res["metrics"], operational[name]
        lines.append(
            f"  {name:<20}{m['threshold']:>5.2f}{m['precision']:>8.3f}{m['recall']:>8.3f}"
            f"{m['f1']:>8.3f}{m['f1_0.5']:>8.3f}{m['roc_auc']:>8.3f}{m['pr_auc']:>8.3f}"
            f"{o['event_recall@3s']:>8.3f}{100 * o['false_alarm_time_share']:>7.1f}"
        )
    lines += ["", "  EvR@3s = share of successful handovers with an alarm 1–3 s before;",
              "  FA%    = share of time with an alarm not followed by a handover within 3 s.", ""]

    for name, res in results.items():
        m = res["metrics"]
        lines += [f"── {name} ──",
                  f"  Precision : {m['precision']:.4f}",
                  f"  Recall    : {m['recall']:.4f}",
                  f"  F1-score  : {m['f1']:.4f}",
                  f"  ROC-AUC   : {m['roc_auc']:.4f}",
                  f"  PR-AUC    : {m['pr_auc']:.4f}",
                  "", "  Classification Report:"]
        lines += ["    " + ln for ln in m["report"].splitlines()]
        lines.append("")

    report_path = REPORTS_DIR / "evaluation.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved → {report_path}")
    print("\n".join(lines[:16]))


# ── Main ───────────────────────────────────────────────────────────────────────

def run_evaluation():
    print("Phase 4 — Evaluation")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    test, feat_cols = load_test()
    thresholds = load_thresholds()

    probs_all = score_all_models(test, feat_cols)
    rows   = _build_seq_row_indices(test, seq_len=SEQ_LEN)
    sub    = test.loc[rows]
    y_true = sub[LABEL].values.astype(int)

    results, operational = {}, {}
    for name, p in probs_all.items():
        p_sub = p.loc[rows].values
        thr   = thresholds.get(name, 0.5)
        results[name]     = {"metrics": compute_metrics(y_true, p_sub, thr), "probs": p_sub}
        operational[name] = event_metrics(sub, p_sub, thr)
        m, o = results[name]["metrics"], operational[name]
        print(f"  {name:<20} F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}  "
              f"PR-AUC={m['pr_auc']:.4f}  EvR@3s={o['event_recall@3s']:.3f}  "
              f"FA%={100 * o['false_alarm_time_share']:.1f}")

    roc_path = plot_roc_curves(results, y_true)
    pr_path  = plot_pr_curves(results, y_true)
    cm_path  = plot_confusion_matrices(results)
    write_report(results, operational, len(sub), float(y_true.mean()))

    with open(REPORTS_DIR / "evaluation.json", "w") as f:
        json.dump({
            "n_rows": int(len(sub)), "pos_rate": round(float(y_true.mean()), 4),
            "models": {n: {k: v for k, v in r["metrics"].items() if k != "report"}
                       for n, r in results.items()},
        }, f, indent=2)
    curves = {n: operating_curve(sub, r["probs"]) for n, r in results.items()}
    at_budget = {n: recall_at_false_alarm_budget(c, FA_BUDGET) for n, c in curves.items()}
    print(f"\n  Event recall@3s at <= {FA_BUDGET:.0%} false-alarm time:")
    for n, v in at_budget.items():
        print(f"    {n:<20} {v:.3f}")
    with open(REPORTS_DIR / "operational.json", "w") as f:
        json.dump({"models": operational, "curves": curves,
                   "false_alarm_time_budget": FA_BUDGET,
                   "event_recall_at_budget": at_budget}, f, indent=2)

    # ── MLflow: log test metrics + artifacts into training runs ────────────────
    run_ids = _load_run_ids()
    for name, res in results.items():
        _mlf_log_test_metrics(run_ids.get(name, ""), res["metrics"],
                              [roc_path, pr_path, cm_path])

    print("\nPhase 4 complete.\n")
    return results


if __name__ == "__main__":
    run_evaluation()
