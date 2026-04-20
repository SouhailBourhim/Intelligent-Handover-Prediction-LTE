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

Minimum required input columns
-------------------------------
The following raw signal columns are required for any prediction.  All other
columns (lags, rolling stats, deltas) are engineered automatically from these,
but if more than 10 % of engineered features are missing for a given row, that
row is skipped with a warning rather than silently filled with zeros.

  rsrp_serving     — instantaneous serving-cell RSRP (dBm)
  rsrq_serving     — serving-cell RSRQ (dB)
  sinr             — SINR from L3-filtered RSRP (dB)
  cqi              — Channel Quality Indicator (1–15)
  rsrp_neighbor    — best-neighbour instantaneous RSRP (dBm)
  rsrq_neighbor    — best-neighbour RSRQ (dB)
  rsrp_diff        — rsrp_serving − rsrp_neighbor (dB)
  l3_rsrp_serving  — L3-filtered serving RSRP (dBm)
  l3_rsrp_neighbor — L3-filtered neighbour RSRP (dBm)
  ue_speed         — UE speed (m/s)

Optional context columns (improve lag/delta accuracy when present):
  ue_id, timestamp, serving_cell_id, cell_load_pct

Outputs
-------
For every scored row the script prints:
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


_ENGINEERED_SUFFIXES = ("_lag", "_roll", "_delta")
_MISSING_THRESHOLD   = 0.10   # warn + skip row if > 10 % of engineered features are absent


def _is_engineered(col: str) -> bool:
    return any(col.endswith(s) or s in col for s in _ENGINEERED_SUFFIXES)


def _prepare_features(
    df: pd.DataFrame,
    feat_cols: list[str],
    scaler,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the feature matrix from df.

    Returns
    -------
    X          : float32 array of shape (n_valid, n_features)
    valid_mask : bool array of shape (n_rows,); False rows were skipped
                 because > 10 % of their engineered features were missing.
    """
    eng_cols = [c for c in feat_cols if _is_engineered(c)]
    n_eng    = len(eng_cols) if eng_cols else 1   # avoid div-by-zero on unusual inputs

    # Columns entirely absent from the input → mark as missing for every row
    absent_cols  = [c for c in eng_cols if c not in df.columns]
    n_absent     = len(absent_cols)

    # Fill absent columns with NaN so per-row check works uniformly
    for col in feat_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Per-row missing fraction for engineered features
    eng_present = [c for c in eng_cols if c not in absent_cols]
    per_row_nan = df[eng_present].isna().sum(axis=1) if eng_present else pd.Series(0, index=df.index)
    per_row_missing_frac = (per_row_nan + n_absent) / n_eng

    valid_mask = (per_row_missing_frac <= _MISSING_THRESHOLD).values

    n_skip = (~valid_mask).sum()
    if n_skip > 0:
        print(
            f"  Warning: {n_skip} row(s) have >{_MISSING_THRESHOLD:.0%} engineered "
            f"features missing and will be skipped. Supply lag/rolling/delta history "
            f"columns to score these rows. Missing columns (sample): "
            f"{absent_cols[:5]}{'...' if len(absent_cols) > 5 else ''}"
        )

    df_valid = df[valid_mask].copy()
    X = df_valid[feat_cols].fillna(0).values.astype(np.float32)

    if scaler is not None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            X = scaler.transform(X).astype(np.float32)

    return X, valid_mask


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

_REQUIRED_COLS_HELP = """
minimum required input columns:
  rsrp_serving, rsrq_serving, sinr, cqi,
  rsrp_neighbor, rsrq_neighbor, rsrp_diff,
  l3_rsrp_serving, l3_rsrp_neighbor, ue_speed

rows with >10%% of engineered features missing are skipped with a warning.
"""


def main():
    parser = argparse.ArgumentParser(
        description="Score UE measurements against the champion handover model.",
        epilog=_REQUIRED_COLS_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    X, valid_mask = _prepare_features(df_input.copy(), feat_cols, scaler)

    if len(X) == 0:
        print("No rows left to score after filtering. "
              "Provide engineered-feature columns (lags, rolling stats, deltas).")
        sys.exit(1)

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
    _print_results(df_input[valid_mask].reset_index(drop=True), probs, args.threshold)


if __name__ == "__main__":
    main()
