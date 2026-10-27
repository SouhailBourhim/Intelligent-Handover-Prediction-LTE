"""
promote_best_model.py

Selects the champion model on **validation** metrics and copies it to
models/champion/ with a metadata.json summary.

Selection
---------
  score = 0.6 × val_F1 (at the validation-tuned threshold) + 0.4 × val_ROC-AUC

  Validation metrics come from models/thresholds.json, written by training
  (src/models.py).  The test set is never used to choose the model: it is
  only read afterwards (reports/evaluation.json) to report the champion's
  held-out performance.  The rule baseline is reported but not eligible.

Why F1 is weighted higher than AUC:
  In LTE handover prediction a missed handover (false negative) can mean a
  late handover and a radio-link failure, while a false alarm only triggers a
  preparation that is cheaply cancelled.  F1 reflects the operating point the
  model is actually used at; AUC rewards ranking quality across all
  thresholds.  60/40 keeps both in view.

Usage:
    python scripts/promote_best_model.py
"""

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHAMPION_DIR    = ROOT / "models" / "champion"
MODELS_DIR      = ROOT / "models"
THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"
EVAL_PATH       = ROOT / "reports" / "evaluation.json"
RUN_IDS_PATH    = MODELS_DIR / "mlflow_run_ids.json"

W_F1  = 0.6
W_AUC = 0.4

# Display name → files needed to serve that model
MODEL_FILES = {
    "Logistic Regression": ["logistic_regression.pkl"],
    "Random Forest":       ["random_forest.pkl"],
    "XGBoost":             ["xgboost.pkl"],
    "LSTM":                ["lstm.pt"],
    "GRU":                 ["gru.pt"],
    "Stacking Ensemble":   ["stacking_ensemble.pkl", "xgboost.pkl",
                            "random_forest.pkl", "lstm.pt"],
}


def weighted_score(f1: float, auc: float) -> float:
    return W_F1 * f1 + W_AUC * auc


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def promote() -> dict:
    val = _load_json(THRESHOLDS_PATH)
    if not val:
        sys.exit("models/thresholds.json not found — run training first "
                 "(python run_pipeline.py --phase 3).")

    candidates = {n: m for n, m in val.items() if n in MODEL_FILES}
    scores = {n: weighted_score(m["val_f1_tuned"], m["val_roc_auc"])
              for n, m in candidates.items()}
    best = max(scores, key=scores.get)

    print(f"Criterion: {W_F1}×val_F1 + {W_AUC}×val_AUC  (validation split)")
    for n in sorted(scores, key=scores.get, reverse=True):
        m = candidates[n]
        mark = "  ← champion" if n == best else ""
        print(f"  {n:<20} F1={m['val_f1_tuned']:.4f}  AUC={m['val_roc_auc']:.4f}  "
              f"score={scores[n]:.4f}{mark}")

    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)
    for old in CHAMPION_DIR.iterdir():
        if old.suffix in {".pkl", ".pt"}:
            old.unlink()
    copied = []
    for fname in MODEL_FILES[best]:
        src = MODELS_DIR / fname
        if src.exists():
            shutil.copy2(src, CHAMPION_DIR / fname)
            copied.append(fname)
        else:
            print(f"  Warning: {src} not found — skipping.")

    test = _load_json(EVAL_PATH).get("models", {}).get(best, {})
    meta = {
        "model_name":          MODEL_FILES[best][0].rsplit(".", 1)[0],
        "display_name":        best,
        "run_id":              _load_json(RUN_IDS_PATH).get(best) or None,
        "threshold":           candidates[best]["threshold"],
        "val_f1":              candidates[best]["val_f1_tuned"],
        "val_roc_auc":         candidates[best]["val_roc_auc"],
        "val_pr_auc":          candidates[best]["val_pr_auc"],
        "champion_score":      round(scores[best], 4),
        "test_f1":             test.get("f1"),
        "test_roc_auc":        test.get("roc_auc"),
        "test_pr_auc":         test.get("pr_auc"),
        "promotion_criterion": f"{W_F1}*val_F1 + {W_AUC}*val_AUC",
        "promoted_at":         datetime.now(timezone.utc).isoformat(),
        "model_files":         copied,
    }
    with open(CHAMPION_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


if __name__ == "__main__":
    print("=" * 50)
    print("  Champion Model Promotion")
    print("=" * 50)
    meta = promote()
    print("\nPromotion complete.")
    print(json.dumps(meta, indent=2))
