import math

import numpy as np
import pytest

from src.radio_model import (
    RADIO, compute_rsrp, compute_sinr, los_from_latent, p_los,
    path_loss_los, path_loss_nlos, sinr_to_cqi, update_fast_fading, update_los_latent,
)


def test_p_los_is_a_probability_that_decreases_with_distance():
    d = np.linspace(1, 1000, 200)
    p = np.array([p_los(x) for x in d])
    assert ((0 <= p) & (p <= 1)).all()
    assert (np.diff(p) <= 1e-12).all()


def test_nlos_path_loss_never_below_los():
    for d in [10, 50, 100, 300, 800]:
        assert path_loss_nlos(d) >= path_loss_los(d)


def test_cqi_is_monotone_in_sinr():
    cqis = [sinr_to_cqi(s) for s in np.linspace(-20, 30, 101)]
    assert cqis == sorted(cqis)
    assert cqis[0] == 1 and cqis[-1] == 15


def test_reported_rsrp_is_inside_the_3gpp_range():
    assert compute_rsrp(1.0, True, 10.0, 3.0) == RADIO.max_rsrp == -44.0
    assert compute_rsrp(5000.0, False, -20.0, -8.0) == RADIO.min_rsrp
    # internal (unreported) value is not capped at -44
    assert compute_rsrp(1.0, True, 10.0, 3.0, report=False) > -44.0


def test_rsrp_is_per_resource_element():
    # 46 dBm over 600 subcarriers -> ~18.2 dBm per RE
    assert RADIO.rs_power_dbm == pytest.approx(46 - 10 * math.log10(600), abs=1e-9)


def test_sinr_is_unchanged_by_the_per_re_rescale():
    """S, I and N are all shifted by the same 27.8 dB, so SINR equals the wideband ratio."""
    off = RADIO.re_offset_db
    s_wb, i_wb, load = -70.0, -80.0, 0.5
    sinr = compute_sinr(s_wb - off, [(i_wb - off, load)])
    active = RADIO.load_idle_frac + (1 - RADIO.load_idle_frac) * load
    expected = 10 * math.log10(10 ** (s_wb / 10) /
                               (10 ** (i_wb / 10) * active + 10 ** (RADIO.thermal_noise_dbm / 10)))
    assert sinr == pytest.approx(expected, abs=1e-9)


def test_los_state_does_not_change_for_a_stationary_ue():
    rng = np.random.default_rng(0)
    z = 0.3
    for _ in range(100):
        z_new = update_los_latent(z, 0.0, rng)
        assert z_new == z
    assert los_from_latent(z, 120.0) == los_from_latent(z_new, 120.0)


def test_los_marginal_probability_matches_3gpp():
    rng = np.random.default_rng(1)
    z = rng.normal(size=20000)
    for d in [50, 150, 400]:
        share = np.mean([los_from_latent(v, d) for v in z])
        assert share == pytest.approx(p_los(d), abs=0.02)


def test_fast_fading_averaging_shrinks_with_distance_travelled():
    rng = np.random.default_rng(2)
    slow = [update_fast_fading(0, 0, 1.0, False, rng)[2] for _ in range(2000)]
    fast = [update_fast_fading(0, 0, 15.0, False, rng)[2] for _ in range(2000)]
    assert np.std(fast) < np.std(slow) < 2.0
