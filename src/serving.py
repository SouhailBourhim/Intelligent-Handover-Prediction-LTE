"""
Inference on raw measurement tables — shared by predict.py, the dashboard,
src/generalization.py and the tests.

    raw measurements (any row order, one or more UEs)
      → build_features()      lags / rolling stats / deltas per UE
      → scaler.transform()    the scaler fitted on the training split
      → model                 sklearn, PyTorch sequence model or stacking
      → probabilities joined back to the input rows **by index**

Rows without enough history for the lag features (the first WARMUP_STEPS rows
of each UE) are returned with prob = NaN and warmup = True rather than being
scored on made-up values.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")

from src.features import build_features  # noqa: E402

ROOT         = Path(__file__).resolve().parents[1]
MODELS_DIR   = ROOT / "models"
CHAMPION_DIR = MODELS_DIR / "champion"
META_PATH    = ROOT / "data" / "processed" / "meta.json"

SKLEARN_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest":       "random_forest.pkl",
    "XGBoost":             "xgboost.pkl",
    "Rule (A3 gap)":       "rule_a3_gap.pkl",
}
TORCH_FILES = {"LSTM": "lstm.pt", "GRU": "gru.pt"}


@dataclass
class LoadedModel:
    name:      str                      # display name, e.g. "XGBoost"
    kind:      str                      # "sklearn" | "torch" | "stacking"
    model:     object
    threshold: float = 0.5
    parts:     dict = field(default_factory=dict)   # stacking base models


def feature_columns() -> list[str]:
    with open(META_PATH) as f:
        return json.load(f)["feature_cols"]


def _load_torch(name: str, path: Path, n_features: int):
    import torch
    from src.models import GRUClassifier, LSTMClassifier
    cls = LSTMClassifier if name == "LSTM" else GRUClassifier
    net = cls(input_size=n_features)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.eval()


def load_model(name: str, models_dir: Path = MODELS_DIR,
               threshold: float | None = None) -> LoadedModel:
    """Load any of the seven pipeline models by display name."""
    if threshold is None:
        thr_path = MODELS_DIR / "thresholds.json"
        thr = json.loads(thr_path.read_text()) if thr_path.exists() else {}
        threshold = thr.get(name, {}).get("threshold", 0.5)

    if name in SKLEARN_FILES:
        return LoadedModel(name, "sklearn", joblib.load(models_dir / SKLEARN_FILES[name]), threshold)
    n_feat = len(feature_columns())
    if name in TORCH_FILES:
        return LoadedModel(name, "torch", _load_torch(name, models_dir / TORCH_FILES[name], n_feat),
                           threshold)
    if name == "Stacking Ensemble":
        parts = {
            "xgb":  joblib.load(models_dir / "xgboost.pkl"),
            "rf":   joblib.load(models_dir / "random_forest.pkl"),
            "lstm": _load_torch("LSTM", models_dir / "lstm.pt", n_feat),
        }
        return LoadedModel(name, "stacking", joblib.load(models_dir / "stacking_ensemble.pkl"),
                           threshold, parts)
    raise ValueError(f"Unknown model '{name}'")


def champion_metadata() -> dict:
    path = CHAMPION_DIR / "metadata.json"
    if not path.exists():
        raise FileNotFoundError("models/champion/metadata.json not found — "
                                "run scripts/promote_best_model.py after the pipeline.")
    return json.loads(path.read_text())


def load_champion() -> LoadedModel:
    meta = champion_metadata()
    return load_model(meta["display_name"], CHAMPION_DIR, meta.get("threshold", 0.5))


def predict_scaled(model: LoadedModel, feats: pd.DataFrame,
                   feat_cols: list[str]) -> pd.Series:
    """Probabilities for an already-scaled feature frame (needs ue_id, timestamp)."""
    from src.models import score_sequence_model, stacking_meta_features

    X = feats[feat_cols].fillna(0).values
    if model.kind == "sklearn":
        return pd.Series(model.model.predict_proba(X.astype(np.float32))[:, 1], feats.index)
    if model.kind == "torch":
        return score_sequence_model(model.model, feats, feat_cols)
    p_lstm = score_sequence_model(model.parts["lstm"], feats, feat_cols)
    meta_X = stacking_meta_features(feats, feat_cols, model.parts["xgb"],
                                    model.parts["rf"], p_lstm.values)
    return pd.Series(model.model.predict_proba(meta_X)[:, 1], feats.index)


def score_measurements(df_raw: pd.DataFrame, model: LoadedModel,
                       feat_cols: list[str] | None = None, scaler=None) -> pd.DataFrame:
    """
    Score a raw measurement table.

    Returns a frame on df_raw's index with columns:
      prob    — P(handover within 3 s), NaN for warm-up rows
      warmup  — True for rows without WARMUP_STEPS of history
      alarm   — prob >= model.threshold
    """
    feat_cols = feat_cols or feature_columns()
    scaler    = scaler if scaler is not None else joblib.load(MODELS_DIR / "scaler.pkl")

    df = df_raw.copy()
    if "ue_id" not in df:
        df["ue_id"] = 0
    if "timestamp" not in df:
        df["timestamp"] = df.groupby("ue_id").cumcount()

    feats = build_features(df, drop_warmup=True)
    out = pd.DataFrame(index=df_raw.index)
    out["prob"]   = np.nan
    out["warmup"] = True
    if len(feats):
        scaled = feats.copy()
        scaled[feat_cols] = scaler.transform(feats[feat_cols])
        out.loc[feats.index, "prob"]   = predict_scaled(model, scaled, feat_cols)
        out.loc[feats.index, "warmup"] = False
    out["alarm"] = out["prob"] >= model.threshold
    return out
