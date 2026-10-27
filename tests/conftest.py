"""Shared fixtures: small simulations and a tiny trained model (no pipeline outputs needed)."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_test_dataset import run as run_scenario  # noqa: E402
from src.simulator import add_handover_soon_label  # noqa: E402


@pytest.fixture(scope="session")
def small_sim() -> pd.DataFrame:
    """6 mixed UEs × 400 s, labelled — enough for handovers, failures, ping-pongs."""
    return add_handover_soon_label(run_scenario("default", 6, 400, seed=7))


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory, small_sim):
    """
    A champion directory with the same layout the pipeline produces
    (scaler.pkl, meta.json, thresholds.json, champion/), built from a quick
    XGBoost fitted on a small simulation.
    """
    import json
    import joblib
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    from src.features import SCALE_COLS, build_features

    d = tmp_path_factory.mktemp("models")
    train_raw = add_handover_soon_label(run_scenario("vehicle", 6, 600, seed=3))
    feats = build_features(train_raw)
    scaler = StandardScaler().fit(feats[SCALE_COLS])
    X = scaler.transform(feats[SCALE_COLS]).astype(np.float32)
    model = XGBClassifier(n_estimators=30, max_depth=3, random_state=0, verbosity=0)
    model.fit(X, feats["handover_soon"].values)

    (d / "champion").mkdir()
    joblib.dump(scaler, d / "scaler.pkl")
    joblib.dump(model, d / "champion" / "xgboost.pkl")
    joblib.dump(model, d / "xgboost.pkl")
    (d / "meta.json").write_text(json.dumps({"feature_cols": SCALE_COLS}))
    (d / "thresholds.json").write_text(json.dumps({"XGBoost": {"threshold": 0.5}}))
    (d / "champion" / "metadata.json").write_text(json.dumps({
        "model_name": "xgboost", "display_name": "XGBoost", "threshold": 0.5,
        "val_f1": 0.0, "val_roc_auc": 0.0, "test_f1": None,
    }))
    return d


@pytest.fixture
def serving_paths(monkeypatch, tiny_model_dir):
    """Point src.serving at the tiny model directory."""
    import src.serving as serving
    monkeypatch.setattr(serving, "MODELS_DIR", tiny_model_dir)
    monkeypatch.setattr(serving, "CHAMPION_DIR", tiny_model_dir / "champion")
    monkeypatch.setattr(serving, "META_PATH", tiny_model_dir / "meta.json")
    return tiny_model_dir
