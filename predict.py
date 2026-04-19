"""
predict.py — Champion-model inference for LTE handover risk.

Loads the promoted champion model and outputs a handover-soon probability
for one or more UE measurement rows.

Usage
-----
# Score a single row supplied as JSON:
python predict.py --json '{"rsrp_serving":-85,"rsrq_serving":-12,...}'

# Score a CSV file (must contain the raw signal columns; temporal features
# are computed automatically if history columns are absent):
python predict.py --csv path/to/measurements.csv

# Show which model is currently champion:
python predict.py --info

Outputs
-------
For every input row the script prints:
  ue_id (if present)  |  probability  |  decision (HANDOVER / ok)  |  threshold used

Exit code 0 = success, 1 = champion model not found (run the pipeline first).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Prevent OpenMP conflict between XGBoost and PyTorch on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT         = Path(__file__).resolve().parent
MODELS_DIR   = ROOT / "models"
CHAMPION_DIR = MODELS_DIR / "champion"
META_PATH    = ROOT / "data" / "processed" / "meta.json"
SCALER_PATH  = MODELS_DIR / "scaler.pkl"

DEFAULT_THRESHOLD = 0.5


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_meta() -> dict:
    if not META_PATH.exists():
        sys.exit("❌  data/processed/meta.json not found — run the pipeline first.")
    with open(META_PATH) as f:
        return json.load(f)


def _load_champion():
    """Return (model_object, metadata_dict).  Supports sklearn (.pkl) and
    PyTorch (.pt) champions."""
    meta_file = CHAMPION_DIR / "metadata.json"
    if not meta_file.exists():
        sys.exit(
            "❌  models/champion/metadata.json not found.\n"
            "    Run `python scripts/promote_best_model.py` after the pipeline."
        )
    with open(meta_file) as f:
        champ_meta = json.load(f)

    model_name = champ_meta["model_name"]
    model_files = champ_meta.get("model_files", [])

    # Prefer the file listed in metadata; fall back to guessing by name.
    pkl_path = CHAMPION_DIR / f"{model_name}.pkl"
    pt_path  = CHAMPION_DIR / f"{model_name}.pt"

    if pkl_path.exists():
        model = joblib.load(pkl_path)
        return model, champ_meta, "sklearn"

    if pt_path.exists():
        import torch
        from src.models import LSTMClassifier, GRUClassifier

        feat_cols = _load_meta()["feature_cols"]
        n_features = len(feat_cols)

        if "lstm" in model_name:
            net = LSTMClassifier(input_size=n_features)
        else:
            from src.models import GRUClassifier
            net = GRUClassifier(input_size=n_features)

        net.load_state_dict(torch.load(pt_path, map_location="cpu"))
        net.eval()
        return net, champ_meta, "torch"

    sys.exit(f"❌  No model file found in {CHAMPION_DIR} for '{model_name}'.")


def _load_scaler():
    if not SCALER_PATH.exists():
        return None
    return joblib.load(SCALER_PATH)


def _prepare_features(df: pd.DataFrame, feat_cols: list[str], scaler) -> np.ndarray:
    """
    Fill any missing engineered columns with 0 (graceful degradation when the
    caller supplies only raw signal columns), scale, and return float32 array.
    """
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0.0

    X = df[feat_cols].fillna(0).values.astype(np.float32)

    if scaler is not None:
        import warnings
        # Scaler was fit on a DataFrame; suppress the feature-name warning when
        # we pass a plain numpy array (behaviour is identical).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            X = scaler.transform(X).astype(np.float32)

    return X


def _score_sklearn(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X.astype(np.float32))[:, 1]


def _build_stacking_meta(X: np.ndarray, feat_cols: list[str], scaler) -> np.ndarray:
    """
    Stacking ensemble expects [xgb_prob, rf_prob, lstm_prob] as meta-features.
    Load the three base models and derive those probabilities from X.
    """
    import torch
    from src.models import LSTMClassifier, SEQ_LEN

    xgb_model   = joblib.load(MODELS_DIR / "xgboost.pkl")
    rf_model    = joblib.load(MODELS_DIR / "random_forest.pkl")
    lstm_net    = LSTMClassifier(input_size=X.shape[1])
    lstm_net.load_state_dict(torch.load(MODELS_DIR / "lstm.pt", map_location="cpu"))
    lstm_net.eval()

    xgb_probs  = xgb_model.predict_proba(X.astype(np.float32))[:, 1]
    rf_probs   = rf_model.predict_proba(X)[:, 1]
    lstm_probs = _score_torch(lstm_net, X)

    # Align: LSTM produces one prob per row (with sequence padding applied inside
    # _score_torch), so all three arrays are already the same length.
    meta = np.column_stack([xgb_probs, rf_probs, lstm_probs]).astype(np.float32)
    return meta


def _score_torch(net, X: np.ndarray) -> np.ndarray:
    """Return one probability per input row using a sliding SEQ_LEN window.

    Rows that don't have a full look-back history are padded by repeating the
    first available row — so the output length always equals len(X).
    """
    import torch
    from src.models import SEQ_LEN

    n = len(X)
    # Build a padded array so every row i can look back SEQ_LEN steps.
    # Pad = (SEQ_LEN - 1) copies of the first row prepended.
    pad = np.repeat(X[:1], SEQ_LEN - 1, axis=0)
    X_padded = np.vstack([pad, X])           # shape: (n + SEQ_LEN - 1, features)

    probs = []
    with torch.no_grad():
        for i in range(n):
            seq = torch.tensor(
                X_padded[i : i + SEQ_LEN], dtype=torch.float32
            ).unsqueeze(0)
            logit = net(seq)
            probs.append(torch.sigmoid(logit).item())

    return np.array(probs, dtype=np.float32)


def _print_results(df_input: pd.DataFrame, probs: np.ndarray, threshold: float):
    has_ue = "ue_id" in df_input.columns

    header = f"{'UE':>6}  {'Prob':>7}  {'Decision':>10}" if has_ue else f"{'Row':>4}  {'Prob':>7}  {'Decision':>10}"
    print(f"\n{header}  (threshold={threshold:.2f})")
    print("─" * len(header))

    for i, p in enumerate(probs):
        decision = "⚠  HANDOVER" if p >= threshold else "✓  ok"
        ue_val   = str(df_input["ue_id"].iloc[i]) if has_ue else str(i)
        col1     = f"{ue_val:>6}" if has_ue else f"{ue_val:>4}"
        print(f"{col1}  {p:>7.4f}  {decision}")

    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Score UE measurements against the champion handover model."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--json", metavar="JSON",
        help='Single-row JSON object, e.g. \'{"rsrp_serving":-85, ...}\'',
    )
    group.add_argument(
        "--csv", metavar="FILE",
        help="Path to a CSV file with one or more measurement rows.",
    )
    group.add_argument(
        "--info", action="store_true",
        help="Print champion model info and exit.",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Decision threshold for HANDOVER label (default {DEFAULT_THRESHOLD}).",
    )
    args = parser.parse_args()

    # ── --info ─────────────────────────────────────────────────────────────────
    if args.info or (not args.json and not args.csv):
        meta_file = CHAMPION_DIR / "metadata.json"
        if not meta_file.exists():
            print("No champion model found.  Run `python scripts/promote_best_model.py`.")
            return
        with open(meta_file) as f:
            m = json.load(f)
        print("\n🏆  Champion model")
        print(f"   Name      : {m['model_name']}")
        print(f"   Test AUC  : {m.get('test_roc_auc', 'n/a'):.4f}")
        print(f"   Test F1   : {m.get('test_f1') or 'n/a'}")
        print(f"   Promoted  : {str(m.get('promoted_at',''))[:19].replace('T',' ')}")
        print(f"   Run ID    : {str(m.get('run_id',''))[:16] or 'n/a'}\n")
        return

    # ── load model + meta ──────────────────────────────────────────────────────
    pipe_meta  = _load_meta()
    feat_cols  = pipe_meta["feature_cols"]
    scaler     = _load_scaler()
    model, champ_meta, model_type = _load_champion()

    # ── parse input ───────────────────────────────────────────────────────────
    if args.json:
        try:
            row = json.loads(args.json)
        except json.JSONDecodeError as e:
            sys.exit(f"❌  Invalid JSON: {e}")
        df_input = pd.DataFrame([row])
    else:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            sys.exit(f"❌  File not found: {csv_path}")
        df_input = pd.read_csv(csv_path)

    # ── feature prep + scoring ─────────────────────────────────────────────────
    X = _prepare_features(df_input.copy(), feat_cols, scaler)

    if model_type == "sklearn":
        if champ_meta["model_name"] == "stacking_ensemble":
            meta_X = _build_stacking_meta(X, feat_cols, scaler)
            probs  = _score_sklearn(model, meta_X)
        else:
            probs = _score_sklearn(model, X)
    else:
        probs = _score_torch(model, X)

    # ── output ────────────────────────────────────────────────────────────────
    print(f"\nModel: {champ_meta['model_name']}  |  "
          f"AUC={champ_meta.get('test_roc_auc', 0):.4f}")
    _print_results(df_input, probs, args.threshold)


if __name__ == "__main__":
    main()
