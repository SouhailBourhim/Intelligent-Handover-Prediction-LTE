"""
Phase 2 — Data Preparation & Feature Engineering

Pipeline:
  1. Load raw CSV
  2. Clean + validate
  3. Scale numeric features
  4. Engineer time-based features (lags, rolling stats, deltas)
  5. Save processed dataset — no leakage guaranteed (all shifts are backward-looking)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "dataset.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# ── Constants ──────────────────────────────────────────────────────────────────
LAG_STEPS = [1, 2, 3]            # backward lags for key signals
ROLL_WINDOWS = [3, 5]            # rolling window sizes (seconds)
DELTA_STEPS = [1, 3]             # delta over N steps

SIGNAL_COLS = ["rsrp_serving", "rsrq_serving", "sinr", "cqi",
               "rsrp_neighbor", "rsrq_neighbor"]

# Features that enter the scaler (constructed after engineering)
SCALE_COLS = SIGNAL_COLS + [
    "ue_speed", "pos_x", "pos_y",
    "rsrp_diff",                  # serving − neighbor gap
] + [
    f"{c}_lag{k}" for c in SIGNAL_COLS for k in LAG_STEPS
] + [
    f"{c}_roll{w}_mean" for c in SIGNAL_COLS for w in ROLL_WINDOWS
] + [
    f"{c}_roll{w}_std"  for c in SIGNAL_COLS for w in ROLL_WINDOWS
] + [
    f"{c}_delta{d}" for c in SIGNAL_COLS for d in DELTA_STEPS
]


# ── Step 1: Load & validate ────────────────────────────────────────────────────

def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "timestamp", "ue_id", "serving_cell_id",
        "rsrp_serving", "rsrq_serving", "sinr", "cqi",
        "best_neighbor_cell_id", "rsrp_neighbor", "rsrq_neighbor",
        "ue_speed", "pos_x", "pos_y",
        "handover_event", "target_cell_id", "handover_soon",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


# ── Step 2: Clean ──────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["ue_id", "timestamp"]).reset_index(drop=True)

    # Drop duplicate (ue_id, timestamp) pairs if any
    df = df.drop_duplicates(subset=["ue_id", "timestamp"])

    # Clip to physically valid ranges
    df["rsrp_serving"]  = df["rsrp_serving"].clip(-140, -30)
    df["rsrp_neighbor"] = df["rsrp_neighbor"].clip(-140, -30)
    df["rsrq_serving"]  = df["rsrq_serving"].clip(-20, -3)
    df["rsrq_neighbor"] = df["rsrq_neighbor"].clip(-20, -3)
    df["sinr"]          = df["sinr"].clip(-20, 30)
    df["cqi"]           = df["cqi"].clip(1, 15).astype(int)
    df["ue_speed"]      = df["ue_speed"].clip(0.5, 25)

    # Fill NaN (none expected, but safety net)
    df[SIGNAL_COLS] = df[SIGNAL_COLS].ffill().bfill()

    return df


# ── Step 3: Feature engineering ───────────────────────────────────────────────

def _per_ue(group: pd.DataFrame) -> pd.DataFrame:
    """All temporal features computed within a single UE's timeline."""
    g = group.copy()

    # -- Domain feature: RSRP gap (positive = neighbor is better)
    g["rsrp_diff"] = g["rsrp_neighbor"] - g["rsrp_serving"]

    for col in SIGNAL_COLS:
        # Lag features (t-k): backward shift → no leakage
        for k in LAG_STEPS:
            g[f"{col}_lag{k}"] = g[col].shift(k)

        # Rolling mean & std (min_periods=1 avoids NaN at start)
        for w in ROLL_WINDOWS:
            g[f"{col}_roll{w}_mean"] = g[col].shift(1).rolling(w, min_periods=1).mean()
            g[f"{col}_roll{w}_std"]  = g[col].shift(1).rolling(w, min_periods=1).std().fillna(0)

        # Delta features: value(t) - value(t-d)
        for d in DELTA_STEPS:
            g[f"{col}_delta{d}"] = g[col] - g[col].shift(d)

    return g


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Use list-concat to avoid pandas 2.x groupby.apply column-consumption bug
    max_lag = max(LAG_STEPS + DELTA_STEPS)
    chunks = []
    for _, group in df.groupby("ue_id", sort=False):
        g = _per_ue(group)
        # Drop first max_lag rows where lag/delta values are NaN
        g = g.iloc[max_lag:].copy()
        chunks.append(g)

    return pd.concat(chunks, ignore_index=True)


# ── Step 4: Scale ──────────────────────────────────────────────────────────────

def split_and_scale(df: pd.DataFrame):
    """
    Temporal train/val/test split (no shuffling — preserves time order).
    Scaler fit on train only → applied to val + test.

    Returns (train, val, test, scaler, feature_cols).
    """
    timestamps = df["timestamp"].unique()
    n = len(timestamps)
    t_train = timestamps[:int(n * 0.70)]
    t_val   = timestamps[int(n * 0.70):int(n * 0.85)]
    t_test  = timestamps[int(n * 0.85):]

    train = df[df["timestamp"].isin(t_train)].copy()
    val   = df[df["timestamp"].isin(t_val)].copy()
    test  = df[df["timestamp"].isin(t_test)].copy()

    # Only scale columns that actually exist (SCALE_COLS may include engineered ones)
    present_scale = [c for c in SCALE_COLS if c in df.columns]

    scaler = StandardScaler()
    train[present_scale] = scaler.fit_transform(train[present_scale])
    val[present_scale]   = scaler.transform(val[present_scale])
    test[present_scale]  = scaler.transform(test[present_scale])

    return train, val, test, scaler, present_scale


# ── Step 5: Save artifacts ─────────────────────────────────────────────────────

def save_processed(train, val, test, scaler):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR  / "val.csv",   index=False)
    test.to_csv(PROCESSED_DIR / "test.csv",  index=False)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print(f"Saved: train={len(train)}, val={len(val)}, test={len(test)} rows")


# ── Main entry ─────────────────────────────────────────────────────────────────

def run_feature_pipeline():
    print("Phase 2 — Feature Engineering")
    df = load_raw()
    print(f"  Loaded {len(df)} rows")

    df = clean(df)
    print(f"  Cleaned: {len(df)} rows")

    df = engineer_features(df)
    total_features = len([c for c in SCALE_COLS if c in df.columns])
    print(f"  Engineered features: {total_features} scaled + categoricals")
    print(f"  Total columns: {len(df.columns)}, rows after lag trim: {len(df)}")

    train, val, test, scaler, feat_cols = split_and_scale(df)
    save_processed(train, val, test, scaler)

    # Save feature list for downstream use
    import json
    meta = {
        "feature_cols": feat_cols,
        "label": "handover_soon",
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "class_balance": {
            "train_pos_rate": round(train["handover_soon"].mean(), 4),
        },
    }
    with open(PROCESSED_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Class balance (train): {meta['class_balance']['train_pos_rate']*100:.1f}% positive")
    print("Phase 2 complete.\n")
    return train, val, test, feat_cols


if __name__ == "__main__":
    run_feature_pipeline()
