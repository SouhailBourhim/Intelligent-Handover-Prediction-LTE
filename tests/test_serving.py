import sys

import numpy as np
import pandas as pd
import torch

from src.features import SCALE_COLS


def test_raw_scenario_csv_is_scored(serving_paths, small_sim):
    from src.serving import load_champion, score_measurements
    model = load_champion()
    scores = score_measurements(small_sim, model)
    n_ue = small_sim.ue_id.nunique()
    assert scores.index.equals(small_sim.index)
    assert (~scores["warmup"]).sum() == len(small_sim) - 3 * n_ue
    assert scores.loc[~scores["warmup"], "prob"].between(0, 1).all()


def test_row_order_does_not_change_scores(serving_paths, small_sim):
    """The old dashboard bug: scores attached to the wrong rows for time-sorted input."""
    from src.serving import load_champion, score_measurements
    model = load_champion()
    by_ue   = small_sim.sort_values(["ue_id", "timestamp"])
    by_time = small_sim.sort_values(["timestamp", "ue_id"])
    a = score_measurements(by_ue, model)["prob"]
    b = score_measurements(by_time, model)["prob"]
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())


def test_predict_cli_scores_raw_csv(serving_paths, small_sim, tmp_path, capsys, monkeypatch):
    import predict
    csv = tmp_path / "measurements.csv"
    small_sim.to_csv(csv, index=False)
    monkeypatch.setattr(sys, "argv", ["predict.py", "--csv", str(csv)])
    predict.main()
    out = capsys.readouterr().out
    n_ue = small_sim.ue_id.nunique()
    assert f"{len(small_sim) - 3 * n_ue} of {len(small_sim)} rows scored" in out


def test_sequence_windows_never_cross_ues(small_sim):
    from src.models import LSTMClassifier, score_sequence_model
    torch.manual_seed(0)
    net = LSTMClassifier(input_size=len(SCALE_COLS)).eval()
    df = small_sim.copy()
    df[SCALE_COLS] = np.random.default_rng(0).normal(size=(len(df), len(SCALE_COLS)))
    alone = score_sequence_model(net, df[df.ue_id == 0], SCALE_COLS)
    mixed = score_sequence_model(net, df.sample(frac=1.0, random_state=1), SCALE_COLS)
    pd.testing.assert_series_equal(alone, mixed.loc[alone.index].astype(float),
                                   check_names=False, atol=1e-6)
