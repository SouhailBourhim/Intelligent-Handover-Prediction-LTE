import numpy as np
import pandas as pd

from src.features import (
    SCALE_COLS, SIGNAL_COLS, WARMUP_STEPS, OnlineFeatureBuffer, build_features,
)


def test_no_simulation_only_inputs():
    assert not {"pos_x", "pos_y", "los_flag"} & set(SCALE_COLS)
    assert len(SCALE_COLS) == 83


def test_future_rows_never_change_past_features(small_sim):
    raw = small_sim[small_sim.ue_id == 0].copy()
    base = build_features(raw)
    t_cut = 200
    future = raw.copy()
    future.loc[future.timestamp > t_cut, SIGNAL_COLS] = -100.0   # rewrite the future
    changed = build_features(future)
    past = base.index[base.timestamp <= t_cut]
    pd.testing.assert_frame_equal(base.loc[past, SCALE_COLS], changed.loc[past, SCALE_COLS])


def test_index_is_preserved_and_row_order_irrelevant(small_sim):
    raw = small_sim[small_sim.timestamp < 100]
    shuffled = raw.sample(frac=1.0, random_state=0)
    a = build_features(raw)
    b = build_features(shuffled)
    assert set(a.index) == set(b.index)
    pd.testing.assert_frame_equal(a[SCALE_COLS].sort_index(), b[SCALE_COLS].sort_index())


def test_warmup_rows_are_flagged(small_sim):
    raw = small_sim[small_sim.timestamp < 50]
    full = build_features(raw, drop_warmup=False)
    n_ue = raw.ue_id.nunique()
    assert full["warmup"].sum() == WARMUP_STEPS * n_ue
    assert len(build_features(raw)) == len(raw) - WARMUP_STEPS * n_ue


def test_online_buffer_matches_offline_features(small_sim):
    raw = small_sim[small_sim.timestamp < 150]
    offline = build_features(raw)
    buf, online = OnlineFeatureBuffer(), {}
    for idx, r in raw.sort_values(["timestamp", "ue_id"]).iterrows():
        f = buf.push(r.to_dict())
        if f is not None:
            online[idx] = f
    on = pd.DataFrame.from_dict(online, orient="index").loc[offline.index, SCALE_COLS]
    assert np.abs(on.values - offline[SCALE_COLS].values).max() < 1e-8
