from types import SimpleNamespace

import numpy as np

from src.handover_logic import HO, evaluate_handover, velocity_ttt
from src.mobility import UE

BSS = [SimpleNamespace(bs_id=i) for i in range(3)]


class NeverFail:
    """rng stub: random() always 0.999, so no handover ever fails."""
    def random(self):
        return 0.999


def make_ue(speed=1.0, serving=0):
    return UE(ue_id=0, x=0, y=0, speed=speed, heading=0, is_vehicle=False,
              serving_cell=serving, current_speed=speed)


def step(ue, l3, t, rng=NeverFail(), **kw):
    return evaluate_handover(ue, BSS, l3, 5.0, t, rng, **kw)


def test_velocity_ttt():
    assert velocity_ttt(0.0) == 3
    assert velocity_ttt(7.0) == 2
    assert velocity_ttt(14.0) == 1
    assert velocity_ttt(30.0) == HO.ttt_min_steps


def test_handover_fires_after_exactly_ttt_steps():
    ue = make_ue(speed=1.0)                     # TTT = 3
    l3 = {0: -100.0, 1: -95.0, 2: -120.0}       # neighbour 1 is 5 dB better (> 3 dB)
    events = [step(ue, l3, t)[0] for t in range(3)]
    assert events == [False, False, True]


def test_ttt_resets_when_condition_drops():
    ue = make_ue(speed=1.0)
    good = {0: -100.0, 1: -95.0, 2: -120.0}
    bad  = {0: -100.0, 1: -99.0, 2: -120.0}     # only 1 dB better: A3 not met
    assert not step(ue, good, 0)[0]
    assert not step(ue, good, 1)[0]
    assert not step(ue, bad, 2)[0]
    assert ue.ttt_counter == 0
    assert [step(ue, good, t)[0] for t in (3, 4, 5)] == [False, False, True]


def test_candidate_change_restarts_ttt():
    ue = make_ue(speed=1.0)
    step(ue, {0: -100.0, 1: -95.0, 2: -120.0}, 0)
    step(ue, {0: -100.0, 1: -95.0, 2: -120.0}, 1)
    step(ue, {0: -100.0, 1: -110.0, 2: -94.0}, 2)   # best neighbour switches to 2
    assert ue.ttt_candidate == 2 and ue.ttt_counter == 1


def test_ttt_override_replaces_velocity_ttt():
    ue = make_ue(speed=1.0)
    l3 = {0: -100.0, 1: -95.0, 2: -120.0}
    assert step(ue, l3, 0, ttt_override=1)[0]


def test_ping_pong_adds_extra_margin_on_the_reverse_pair():
    ue = make_ue(speed=20.0)                    # TTT = 1
    assert step(ue, {0: -100.0, 1: -95.0, 2: -130.0}, 0)[0]        # 0 -> 1
    ue.serving_cell = 1
    ev, _, _, _, pp = step(ue, {0: -95.0, 1: -100.0, 2: -130.0}, 2)  # 1 -> 0 within 10 s
    assert ev and pp
    ue.serving_cell = 0
    # going back 0 -> 1 now needs 3 + 3 = 6 dB: a 5 dB advantage is not enough
    assert not step(ue, {0: -100.0, 1: -95.0, 2: -130.0}, 3)[0]
    assert step(ue, {0: -100.0, 1: -93.0, 2: -130.0}, 4)[0]


def test_failed_handover_starts_rlf():
    class AlwaysFail:
        def random(self):
            return 0.0
    ue = make_ue(speed=20.0)
    ev, _, _, failed, pp = step(ue, {0: -100.0, 1: -95.0, 2: -130.0}, 0, rng=AlwaysFail())
    assert ev and failed and not pp
    assert ue.rlf_remaining == HO.rlf_duration_steps
    assert ue.ho_history == []                  # failed attempts are not handovers
