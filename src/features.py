"""
Phase 2 — Data Preparation & Feature Engineering

Pipeline:
  1. Load raw CSV
  2. Clean + validate
  3. Engineer time-based features (lags, rolling stats, deltas)
  4. Scale numeric features
  5. Save processed dataset — no leakage guaranteed (all shifts are backward-looking)

SIGNAL_COLS includes l3_rsrp_serving and l3_rsrp_neighbor — the L3-filtered
measurements the handover logic actually uses.

v4: only quantities a real network observes are features.  pos_x, pos_y and
los_flag stay in the raw dataset for analysis but are not model inputs.

build_features() is the single feature builder used by training, predict.py,
the dashboard and the generalization study; OnlineFeatureBuffer computes the
same features one step at a time for the closed-loop simulation.
"""

from collections import deque

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

SIGNAL_COLS = [
    "rsrp_serving", "rsrq_serving", "sinr", "cqi",
    "rsrp_neighbor", "rsrq_neighbor",
    "l3_rsrp_serving", "l3_rsrp_neighbor",   # v3: L3-filtered measurements
]

# Per-row context the network observes (no position or LOS ground truth)
CONTEXT_COLS = [
    "ue_speed",
    "rsrp_diff",                  # serving − neighbor gap (raw, from the simulator)
    "cell_load_pct",              # serving cell load percentage
]

# Rows of history a UE needs before its lag / delta features exist
WARMUP_STEPS = max(LAG_STEPS + DELTA_STEPS)

# Features that enter the scaler (constructed after engineering)
SCALE_COLS = SIGNAL_COLS + CONTEXT_COLS + [
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
    required = {"timestamp", "ue_id", *SIGNAL_COLS, *CONTEXT_COLS,
                "handover_event", "target_cell_id", "handover_soon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


# ── Step 2: Clean ──────────────────────────────────────────────────────────────

def _clip_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Clip measurements to their 3GPP reporting / physical ranges (in place)."""
    for col in ["rsrp_serving", "rsrp_neighbor", "l3_rsrp_serving", "l3_rsrp_neighbor"]:
        df[col] = df[col].clip(-140, -44)
    df["rsrq_serving"]  = df["rsrq_serving"].clip(-20, -3)
    df["rsrq_neighbor"] = df["rsrq_neighbor"].clip(-20, -3)
    df["sinr"]          = df["sinr"].clip(-20, 30)
    df["cqi"]           = df["cqi"].clip(1, 15).astype(int)
    df["ue_speed"]      = df["ue_speed"].clip(0, 25)
    df["cell_load_pct"] = df["cell_load_pct"].clip(0, 100)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["ue_id", "timestamp"]).reset_index(drop=True)

    # Drop duplicate (ue_id, timestamp) pairs if any
    df = df.drop_duplicates(subset=["ue_id", "timestamp"])
    df = _clip_signals(df)

    # Fill NaN (none expected, but safety net)
    df[SIGNAL_COLS] = df[SIGNAL_COLS].ffill().bfill()

    return df


# ── Step 3: Feature engineering ───────────────────────────────────────────────

def _per_ue(group: pd.DataFrame) -> pd.DataFrame:
    """All temporal features computed within a single UE's timeline."""
    g = group.copy()

    # rsrp_diff is already in the raw v3 dataset (serving − neighbor).
    # Positive values mean the serving cell is stronger; negative means a
    # neighbour is better — the natural A3 trigger condition.

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


def build_features(df_raw: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
    """
    Engineer the model features for any measurement table.

    Works on raw measurement CSVs in any row order: rows are grouped by UE and
    ordered by timestamp before lags / rolling stats / deltas are computed.
    The returned frame **keeps the input index**, so callers attach scores
    back to their own rows with `.loc[index]`, never by position.

    Each row gets `history_steps` (earlier rows of the same UE).  Rows with
    fewer than WARMUP_STEPS have undefined lags; they are dropped by default,
    or kept and marked `warmup=True` when drop_warmup=False.
    """
    missing = {"timestamp", "ue_id", *SIGNAL_COLS, *CONTEXT_COLS} - set(df_raw.columns)
    if missing:
        raise ValueError(f"Missing measurement columns: {sorted(missing)}")

    df = _clip_signals(df_raw.copy())
    chunks = []
    # List-concat avoids the pandas 2.x groupby.apply column-consumption bug
    for _, group in df.groupby("ue_id", sort=False):
        g = _per_ue(group.sort_values("timestamp"))
        g["history_steps"] = np.arange(len(g))
        chunks.append(g)
    out = pd.concat(chunks)
    out["warmup"] = out["history_steps"] < WARMUP_STEPS
    if drop_warmup:
        out = out[~out["warmup"]]
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Training-time features: build_features without warm-up rows, fresh index."""
    return build_features(df, drop_warmup=True).reset_index(drop=True)


class OnlineFeatureBuffer:
    """
    Streaming version of build_features for the closed-loop simulation.

    Keeps the last WINDOW measurements of every UE and computes, for the newest
    row, exactly the features _per_ue computes offline (a parity test in
    tests/test_features.py checks this).
    """

    WINDOW = max(ROLL_WINDOWS + LAG_STEPS + DELTA_STEPS)

    def __init__(self):
        self.history: dict[int, deque] = {}

    def push(self, rec: dict) -> dict | None:
        """Add one measurement row; return its features (None during warm-up)."""
        hist = self.history.setdefault(rec["ue_id"], deque(maxlen=self.WINDOW))
        feats = None
        if len(hist) >= WARMUP_STEPS:
            feats = {c: float(rec[c]) for c in SIGNAL_COLS + CONTEXT_COLS}
            for c in SIGNAL_COLS:
                p = np.array([h[c] for h in hist], dtype=float)   # oldest … newest
                for k in LAG_STEPS:
                    feats[f"{c}_lag{k}"] = p[-k]
                for w in ROLL_WINDOWS:
                    win = p[-w:]
                    feats[f"{c}_roll{w}_mean"] = win.mean()
                    feats[f"{c}_roll{w}_std"]  = win.std(ddof=1) if len(win) > 1 else 0.0
                for d in DELTA_STEPS:
                    feats[f"{c}_delta{d}"] = float(rec[c]) - p[-d]
        hist.append({c: float(rec[c]) for c in SIGNAL_COLS})
        return feats


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
