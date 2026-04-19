"""
promote_best_model.py

Queries the MLflow experiment 'lte_handover_prediction', finds the run with
the highest test_roc_auc, and copies its model artifact to models/champion/.
A metadata.json summary is written alongside.

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

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="metrics.test_roc_auc > 0",
        order_by=["metrics.test_roc_auc DESC"],
        max_results=1,
    )
    if not runs:
        print("No runs with test_roc_auc found.")
        return False

    best = runs[0]
    rname = _run_name(best)
    auc   = best.data.metrics.get("test_roc_auc", 0.0)
    f1    = best.data.metrics.get("test_f1", 0.0)

    print(f"Best run: '{rname}'  (AUC={auc:.4f}, F1={f1:.4f})")

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
        "model_name":   rname,
        "run_id":       best.info.run_id,
        "test_roc_auc": round(auc, 4),
        "test_f1":      round(f1, 4),
        "promoted_at":  datetime.now(timezone.utc).isoformat(),
        "model_files":  copied,
    }
    meta_path = CHAMPION_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata → {meta_path}")
    return True


def promote_via_evaluation_txt():
    """
    Fallback: parse reports/evaluation.txt to find best ROC-AUC model.
    Copies the corresponding pkl/pt to models/champion/.
    """
    report_path = ROOT / "reports" / "evaluation.txt"
    if not report_path.exists():
        print("reports/evaluation.txt not found — run the pipeline first.")
        sys.exit(1)

    lines = report_path.read_text().splitlines()

    # Collect {name: roc_auc}
    scores: dict[str, float] = {}
    current = None
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("── ") and ln.endswith(" ──"):
            current = ln[3:-3].strip()
        elif current and ln.startswith("ROC-AUC"):
            try:
                scores[current] = float(ln.split(":")[-1].strip())
            except ValueError:
                pass

    if not scores:
        print("Could not parse ROC-AUC scores from evaluation.txt.")
        sys.exit(1)

    best_name = max(scores, key=scores.__getitem__)
    best_auc  = scores[best_name]
    print(f"Best model (from evaluation.txt): '{best_name}'  (AUC={best_auc:.4f})")

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
        "model_name":   best_name,
        "run_id":       None,
        "test_roc_auc": round(best_auc, 4),
        "test_f1":      None,
        "promoted_at":  datetime.now(timezone.utc).isoformat(),
        "model_files":  copied,
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
