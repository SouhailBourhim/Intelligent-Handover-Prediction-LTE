"""
promote_best_model.py

Queries the MLflow experiment 'lte_handover_prediction', finds the run with
the highest weighted score (0.6 × F1 + 0.4 × AUC), and copies its model
artifact to models/champion/. A metadata.json summary is written alongside.

Promotion criterion — why F1 is weighted higher than AUC:
  In LTE handover prediction a missed handover (false negative) causes an
  immediate connectivity drop and potential RLF, while a false alarm merely
  triggers an unnecessary preparation phase that is quickly aborted.
  F1 penalises false negatives more directly than AUC, so we weight it 60 %
  vs 40 % for AUC, which captures the overall discriminative ability.

Usage:
    python scripts/promote_best_model.py

Requires:
    mlflow>=2.13.0  (falls back gracefully if unavailable)
"""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHAMPION_DIR  = ROOT / "models" / "champion"
MODELS_DIR    = ROOT / "models"
MLF_TRACKING  = f"sqlite:///{ROOT / 'mlflow.db'}"
MLF_EXP_NAME  = "lte_handover_prediction"

# Promotion criterion weights
W_F1  = 0.6
W_AUC = 0.4


def _weighted_score(f1: float, auc: float) -> float:
    return W_F1 * f1 + W_AUC * auc

# Mapping from MLflow run name → saved model file(s)
_MODEL_FILES = {
    "logistic_regression": ["logistic_regression.pkl"],
    "random_forest":       ["random_forest.pkl"],
    "xgboost":             ["xgboost.pkl"],
    "lstm":                ["lstm.pt"],
    "gru":                 ["gru.pt"],
    "stacking_ensemble":   ["stacking_ensemble.pkl"],
}


def _run_name(run) -> str:
    return run.data.tags.get("mlflow.runName", run.info.run_id[:8]).lower()


def promote_via_mlflow():
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed — falling back to evaluation.txt method.")
        return False

    mlflow.set_tracking_uri(MLF_TRACKING)
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(MLF_EXP_NAME)
    if exp is None:
        print(f"Experiment '{MLF_EXP_NAME}' not found in mlflow.db.")
        return False

    # Fetch all runs that have both test metrics so we can compute the
    # weighted score for each rather than relying on a single metric sort.
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="metrics.test_roc_auc > 0",
    )
    if not runs:
        print("No runs with test_roc_auc found.")
        return False

    def _score(run):
        f1  = run.data.metrics.get("test_f1",      0.0)
        auc = run.data.metrics.get("test_roc_auc",  0.0)
        return _weighted_score(f1, auc)

    best  = max(runs, key=_score)
    rname = _run_name(best)
    auc   = best.data.metrics.get("test_roc_auc", 0.0)
    f1    = best.data.metrics.get("test_f1", 0.0)
    score = _weighted_score(f1, auc)

    print(f"Criterion: {W_F1}×F1 + {W_AUC}×AUC")
    for r in sorted(runs, key=_score, reverse=True):
        rn = _run_name(r)
        _f1  = r.data.metrics.get("test_f1",      0.0)
        _auc = r.data.metrics.get("test_roc_auc",  0.0)
        marker = " ← champion" if r.info.run_id == best.info.run_id else ""
        print(f"  {rn:<25} F1={_f1:.4f}  AUC={_auc:.4f}  score={_weighted_score(_f1,_auc):.4f}{marker}")
    print(f"\nBest run: '{rname}'  (score={score:.4f}, AUC={auc:.4f}, F1={f1:.4f})")

    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)

    # Copy model file(s) to champion/
    files = _MODEL_FILES.get(rname, [])
    copied = []
    for fname in files:
        src = MODELS_DIR / fname
        if src.exists():
            dst = CHAMPION_DIR / fname
            shutil.copy2(src, dst)
            copied.append(fname)
            print(f"  Copied {fname} → {dst}")
        else:
            print(f"  Warning: {src} not found — skipping.")

    # Write metadata
    meta = {
        "model_name":         rname,
        "run_id":             best.info.run_id,
        "test_roc_auc":       round(auc, 4),
        "test_f1":            round(f1, 4),
        "champion_score":     round(score, 4),
        "promotion_criterion": f"{W_F1}*F1 + {W_AUC}*AUC",
        "promoted_at":        datetime.now(timezone.utc).isoformat(),
        "model_files":        copied,
    }
    meta_path = CHAMPION_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata → {meta_path}")
    return True


def promote_via_evaluation_txt():
    """
    Fallback: parse reports/evaluation.txt to find best weighted-score model.
    Copies the corresponding pkl/pt to models/champion/.
    """
    report_path = ROOT / "reports" / "evaluation.txt"
    if not report_path.exists():
        print("reports/evaluation.txt not found — run the pipeline first.")
        sys.exit(1)

    lines = report_path.read_text().splitlines()

    # Collect {name: {roc_auc, f1}} from each model section
    auc_scores: dict[str, float] = {}
    f1_scores:  dict[str, float] = {}
    current = None
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("── ") and ln.endswith(" ──"):
            current = ln[3:-3].strip()
        elif current and ln.startswith("ROC-AUC"):
            try:
                auc_scores[current] = float(ln.split(":")[-1].strip())
            except ValueError:
                pass
        elif current and ln.startswith("F1-score"):
            try:
                f1_scores[current] = float(ln.split(":")[-1].strip())
            except ValueError:
                pass

    if not auc_scores:
        print("Could not parse ROC-AUC scores from evaluation.txt.")
        sys.exit(1)

    # Compute weighted score for each model; models missing F1 get 0
    weighted: dict[str, float] = {
        name: _weighted_score(f1_scores.get(name, 0.0), auc)
        for name, auc in auc_scores.items()
    }

    print(f"Criterion: {W_F1}×F1 + {W_AUC}×AUC")
    for name, ws in sorted(weighted.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:<25} F1={f1_scores.get(name,0):.4f}  AUC={auc_scores.get(name,0):.4f}  score={ws:.4f}")

    best_name  = max(weighted, key=weighted.__getitem__)
    best_score = weighted[best_name]
    best_auc   = auc_scores.get(best_name, 0.0)
    best_f1    = f1_scores.get(best_name, 0.0)
    print(f"\nBest model (from evaluation.txt): '{best_name}'  (score={best_score:.4f})")

    rkey  = best_name.lower().replace(" ", "_")
    files = _MODEL_FILES.get(rkey, [])
    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for fname in files:
        src = MODELS_DIR / fname
        if src.exists():
            dst = CHAMPION_DIR / fname
            shutil.copy2(src, dst)
            copied.append(fname)
            print(f"  Copied {fname} → {dst}")

    meta = {
        "model_name":          best_name,
        "run_id":              None,
        "test_roc_auc":        round(best_auc, 4),
        "test_f1":             round(best_f1, 4),
        "champion_score":      round(best_score, 4),
        "promotion_criterion": f"{W_F1}*F1 + {W_AUC}*AUC",
        "promoted_at":         datetime.now(timezone.utc).isoformat(),
        "model_files":         copied,
    }
    meta_path = CHAMPION_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata → {meta_path}")


if __name__ == "__main__":
    print("=" * 50)
    print("  Champion Model Promotion")
    print("=" * 50)

    ok = promote_via_mlflow()
    if not ok:
        promote_via_evaluation_txt()

    print("\nPromotion complete.")
    meta_path = CHAMPION_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            print(json.dumps(json.load(f), indent=2))
