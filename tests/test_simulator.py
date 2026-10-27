import pandas as pd

from generate_test_dataset import run as run_scenario
from src.handover_logic import HO
from src.simulator import add_handover_soon_label


def test_same_seed_gives_identical_output():
    a = run_scenario("cell_edge", 3, 120, seed=11)
    b = run_scenario("cell_edge", 3, 120, seed=11)
    pd.testing.assert_frame_equal(a, b)


def test_policy_does_not_change_mobility_or_channel():
    """Two RNG streams: a TTT policy changes decisions, never trajectories."""
    from src.simulator import make_simulator
    from generate_test_dataset import create_ues_for_scenario
    factory = lambda bss, rng: create_ues_for_scenario("vehicle", 3, bss, rng)
    base = make_simulator(factory, 5).run(200)
    fast = make_simulator(factory, 5, lambda t, recs: {r["ue_id"]: 1 for r in recs}).run(200)
    pd.testing.assert_series_equal(base["pos_x"], fast["pos_x"])
    assert base["handover_event"].sum() != fast["handover_event"].sum()


def test_reported_rsrp_inside_3gpp_range(small_sim):
    for col in ["rsrp_serving", "rsrp_neighbor", "l3_rsrp_serving", "l3_rsrp_neighbor"]:
        assert small_sim[col].between(-140, -44).all()


def test_label_is_successful_handover_in_next_three_steps():
    df = pd.DataFrame({
        "ue_id":            [0] * 8,
        "timestamp":        range(8),
        "handover_event":   [0, 0, 0, 0, 1, 0, 0, 1],
        "handover_failure": [0, 0, 0, 0, 0, 0, 0, 1],   # t=7 is a failure
    })
    out = add_handover_soon_label(df)
    assert out["handover_soon"].tolist() == [0, 1, 1, 1, 0, 0, 0, 0]


def test_rlf_blocks_handovers_for_exactly_three_steps(small_sim):
    df = small_sim.sort_values(["ue_id", "timestamp"])
    fails = df[df["handover_failure"] == 1]
    assert len(fails) > 0
    for _, r in fails.iterrows():
        nxt = df[(df.ue_id == r.ue_id) &
                 df.timestamp.between(r.timestamp + 1, r.timestamp + HO.rlf_duration_steps)]
        if len(nxt) < HO.rlf_duration_steps:
            continue                      # failure at the end of the run
        assert (nxt["handover_event"] == 0).all()
        assert (nxt["rlf_flag"] == 1).all()


def test_reported_speed_is_zero_while_paused():
    df = run_scenario("stable", 3, 400, seed=2)       # pedestrians pause at waypoints
    assert (df["ue_speed"] == 0).any()
    assert df["ue_speed"].max() <= 2.0 + 1e-9
