"""
Radio propagation model — LTE handover simulation (v4)

Models implemented:
  Path loss  : 3GPP TR 36.873 UMa LOS and NLOS closed-form at 2 GHz
  LOS/NLOS   : 3GPP TR 36.873 distance-dependent probability, spatially
               consistent (latent AR(1) process, 50 m decorrelation, TR 38.901)
  RSRP       : reference-signal power per resource element (TS 36.214), i.e.
               Tx power spread over 600 subcarriers; reported in [−140, −44] dBm
  Shadow     : Gudmundson AR(1) spatial correlation; σ_LOS=4 dB, σ_NLOS=6 dB
  Fast fading: Rayleigh (NLOS) / Rician K=5 dB (LOS), power-averaged over the
               1 s measurement (one independent sample per λ/2 travelled)
  SINR       : S / (Σ active-cell interference × load-fraction + thermal noise)
  RSRQ       : 3GPP TS 36.214 approximation
  CQI        : 3GPP TS 36.213 15-level mapping table
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ── Configuration dataclass ───────────────────────────────────────────────────

@dataclass
class RadioConfig:
    carrier_freq_ghz: float = 2.0           # LTE Band 7
    tx_power_dbm:     float = 46.0          # eNB total Tx power (10 MHz)
    n_subcarriers:    int   = 600           # 50 PRB × 12 subcarriers (10 MHz)
    # kTB over the 9 MHz occupied by 50 PRB (−174 dBm/Hz + 69.5 dB) + 9 dB UE
    # noise figure ≈ −95.5 dBm.  The network is interference-limited, so SINR
    # is almost insensitive to this value.
    thermal_noise_dbm: float = -95.5

    # 3GPP TS 36.133 RSRP reporting range
    min_rsrp: float = -140.0
    max_rsrp: float = -44.0

    # LOS state spatial consistency (TR 38.901 UMa: 50 m)
    los_decorr_m: float = 50.0

    # Shadow fading (Gudmundson AR-1)
    shadow_decorr_m:    float = 100.0       # spatial decorrelation distance
    shadow_sigma_los:   float = 4.0         # dB — 3GPP UMa LOS std dev
    shadow_sigma_nlos:  float = 6.0         # dB — 3GPP UMa NLOS std dev

    # Fast fading, averaged over the 1 s measurement period.  Samples are
    # independent every half wavelength (λ/2 = 7.5 cm at 2 GHz), so a moving UE
    # averages many of them and the residual fading shrinks with speed.
    ff_max_samples: int   = 200             # cap on averaged samples per step
    ff_min_db:      float = -8.0            # deep fade floor (averaged over 1 s)
    ff_max_db:      float = 3.0             # max fast fading gain
    rician_k_db:    float = 5.0             # Rician K-factor for LOS links

    # Interference model
    load_idle_frac: float = 0.10            # fraction of interference when cell idle
    cell_capacity:  int   = 10              # UEs → 100 % load

    @property
    def re_offset_db(self) -> float:
        """dB from total power to power per resource element (≈ 27.8 dB)."""
        return 10.0 * math.log10(self.n_subcarriers)

    @property
    def rs_power_dbm(self) -> float:
        """Reference-signal power per resource element (≈ 18.2 dBm)."""
        return self.tx_power_dbm - self.re_offset_db

    @property
    def noise_re_dbm(self) -> float:
        """Thermal noise per resource element (same SINR as wideband)."""
        return self.thermal_noise_dbm - self.re_offset_db


# Module-level default instance
RADIO = RadioConfig()


# ── LOS probability ────────────────────────────────────────────────────────────

def p_los(d_m: float) -> float:
    """
    3GPP TR 36.873 UMa outdoor LOS probability.

    P(LOS | d) = min(18/d, 1) × (1 − exp(−d/63)) + exp(−d/63)

    Characteristic values:
      d=50 m  → P(LOS) ≈ 0.65
      d=100 m → P(LOS) ≈ 0.35
      d=200 m → P(LOS) ≈ 0.13
      d=400 m → P(LOS) ≈ 0.05
    """
    d = max(d_m, 1.0)
    return min(18.0 / d, 1.0) * (1.0 - math.exp(-d / 63.0)) + math.exp(-d / 63.0)


# ── Path loss ──────────────────────────────────────────────────────────────────

def path_loss_los(d_m: float, fc_ghz: float = 2.0) -> float:
    """3GPP TR 36.873 UMa LOS: PL = 22·log10(d) + 28 + 20·log10(fc_GHz)."""
    d = max(d_m, 1.0)
    return 22.0 * math.log10(d) + 28.0 + 20.0 * math.log10(fc_ghz)


def path_loss_nlos(d_m: float, fc_ghz: float = 2.0) -> float:
    """
    3GPP TR 36.873 UMa NLOS (simplified closed-form, standard params):
      W=20 m, h=20 m, hBS=25 m, hUT=1.5 m
    PL_NLOS = max(PL_LOS, 19.55 + 39.09·log10(d))   [derived analytically]
    """
    d = max(d_m, 1.0)
    pl_los  = path_loss_los(d, fc_ghz)
    pl_nlos = 19.55 + 39.09 * math.log10(d)
    return max(pl_los, pl_nlos)


def path_loss(d_m: float, is_los: bool, fc_ghz: float = 2.0) -> float:
    """Dispatch to LOS or NLOS path loss formula."""
    return path_loss_los(d_m, fc_ghz) if is_los else path_loss_nlos(d_m, fc_ghz)


# ── RSRP ───────────────────────────────────────────────────────────────────────

def compute_rsrp(d_m: float, is_los: bool,
                 shadow_db: float, fast_fade_db: float,
                 cfg: RadioConfig = RADIO, report: bool = True) -> float:
    """
    RSRP = RS_power_per_RE − PL(d, LOS/NLOS) + shadow_db + fast_fade_db

    RSRP is the power of one reference-signal resource element, not the total
    cell power, hence RS_power_per_RE = Tx − 10·log10(600) ≈ 18.2 dBm.

    report=True clamps to the 3GPP reporting range [min_rsrp, max_rsrp].
    report=False only applies the −140 dBm floor: the simulator uses this
    physical value internally (L3 filter, SINR, HO logic) and clamps only
    when writing the dataset.
    """
    pl   = path_loss(d_m, is_los, cfg.carrier_freq_ghz)
    rsrp = cfg.rs_power_dbm - pl + shadow_db + fast_fade_db
    return report_rsrp(rsrp, cfg) if report else float(max(rsrp, cfg.min_rsrp))


def report_rsrp(rsrp_dbm: float, cfg: RadioConfig = RADIO) -> float:
    """Clamp an RSRP value to the UE reporting range (TS 36.133)."""
    return float(np.clip(rsrp_dbm, cfg.min_rsrp, cfg.max_rsrp))


# ── SINR ───────────────────────────────────────────────────────────────────────

def compute_sinr(rsrp_serving_dbm: float,
                 interferers: List[Tuple[float, float]],
                 cfg: RadioConfig = RADIO) -> float:
    """
    SINR = S / (Σ I_k + N)

    Each interferer k contributes:
        I_k = P_k_mW × [load_idle_frac + (1 − load_idle_frac) × load_k]

    This captures that a cell only interferes on PRBs it is actively using.
    An idle cell still radiates reference signals (10 % floor).

    Parameters
    ----------
    rsrp_serving_dbm : received power from serving cell (dBm)
    interferers      : list of (rsrp_dbm, load_fraction) for every neighbour cell
    """
    s_mw = 10.0 ** (rsrp_serving_dbm / 10.0)
    n_mw = 10.0 ** (cfg.noise_re_dbm / 10.0)   # RSRP is per RE, so noise per RE

    i_mw = 0.0
    for rsrp_nb_dbm, load_frac in interferers:
        p_nb_mw     = 10.0 ** (rsrp_nb_dbm / 10.0)
        active_frac = cfg.load_idle_frac + (1.0 - cfg.load_idle_frac) * load_frac
        i_mw       += p_nb_mw * active_frac

    sinr_db = 10.0 * math.log10(s_mw / max(i_mw + n_mw, 1e-30))
    return float(np.clip(sinr_db, -20.0, 30.0))


# ── RSRQ ───────────────────────────────────────────────────────────────────────

def compute_rsrq(rsrp_serving_dbm: float,
                 rsrp_all_dbm: List[float]) -> float:
    """
    RSRQ ≈ RSRP_serving − 10·log10(Σ RSRP_all_mW)   (3GPP TS 36.214 approximation)
    """
    total_mw = sum(10.0 ** (r / 10.0) for r in rsrp_all_dbm)
    rsrq = rsrp_serving_dbm - 10.0 * math.log10(max(total_mw, 1e-30))
    return float(np.clip(rsrq, -20.0, -3.0))


# ── CQI ────────────────────────────────────────────────────────────────────────

_CQI_TABLE: List[Tuple[float, int]] = [
    (-6.7, 1), (-4.7, 2), (-2.3, 3), (0.2, 4), (2.4, 5),
    (4.3, 6),  (5.9, 7),  (8.1, 8),  (10.3, 9), (11.7, 10),
    (14.1, 11),(16.3, 12),(18.7, 13),(21.0, 14),(22.7, 15),
]


def sinr_to_cqi(sinr_db: float) -> int:
    """Map SINR to CQI using 3GPP TS 36.213 15-level table."""
    for thresh, cqi in reversed(_CQI_TABLE):
        if sinr_db >= thresh:
            return cqi
    return 1


# ── LOS state (spatially consistent) ──────────────────────────────────────────

def update_los_latent(z: float, step_dist_m: float, rng: np.random.Generator,
                      cfg: RadioConfig = RADIO) -> float:
    """
    AR(1) update of the latent Gaussian behind a link's LOS state.

        z_new = ρ·z + √(1−ρ²)·N(0,1),   ρ = exp(−Δd / los_decorr_m)

    A UE that does not move keeps its LOS state; one that moves ~50 m has
    largely decorrelated.  Marginally z ~ N(0,1) at every step.
    """
    rho = math.exp(-step_dist_m / cfg.los_decorr_m)
    return rho * z + math.sqrt(max(1.0 - rho ** 2, 0.0)) * rng.normal()


def los_from_latent(z: float, d_m: float) -> bool:
    """LOS if Φ(z) < P_LOS(d), so P(LOS) equals the 3GPP probability exactly."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))) < p_los(d_m)


# ── Shadow fading update ───────────────────────────────────────────────────────

def update_shadow(current_db: float, step_dist_m: float,
                  is_los: bool, rng: np.random.Generator,
                  cfg: RadioConfig = RADIO) -> float:
    """
    Gudmundson AR(1) spatial shadow fading update.

        σ_new = ρ·σ_old + √(1−ρ²)·N(0, σ²)
        ρ = exp(−Δd / d_corr)

    Uses separate σ for LOS (4 dB) and NLOS (6 dB).
    """
    sigma = cfg.shadow_sigma_los if is_los else cfg.shadow_sigma_nlos
    rho   = math.exp(-step_dist_m / cfg.shadow_decorr_m)
    noise = rng.normal(0.0, sigma * math.sqrt(max(1.0 - rho ** 2, 0.0)))
    return rho * current_db + noise


# ── Fast fading update ─────────────────────────────────────────────────────────

def update_fast_fading(
    ff_i: float,
    ff_q: float,
    step_dist_m: float,
    is_los: bool,
    rng: np.random.Generator,
    cfg: RadioConfig = RADIO,
) -> Tuple[float, float, float]:
    """
    Small-scale (fast) fading averaged over one 1-second measurement period.

    The complex channel gain h = I + jQ (unit-variance I, Q) is Rayleigh for
    NLOS links; for LOS links a Rician K-factor adds a fixed in-phase
    component.  Samples decorrelate every half wavelength, so during a step
    of length Δd the UE sees N = 1 + Δd/(λ/2) independent samples and reports
    their mean power, which is what an L1 measurement averages:

        fast_fade_db = 10·log10( mean |h_n|² )      (E|h|² = 1, so ≈ 0 dB)

    A pedestrian at 1 m/s averages ~14 samples (σ ≈ 1.2 dB); a vehicle at
    15 m/s ~200 (σ ≈ 0.3 dB).  A UE that does not move keeps its last value.

    State: ff_i holds the last averaged fading in dB (ff_q is unused and kept
    for interface compatibility).

    Returns
    -------
    new_ff_i, new_ff_q, fast_fade_db
    """
    if step_dist_m <= 0.0:
        ff_db = float(np.clip(ff_i, cfg.ff_min_db, cfg.ff_max_db))
        return ff_db, ff_q, ff_db

    half_wavelength = 3e8 / (cfg.carrier_freq_ghz * 1e9) / 2.0
    n   = int(min(cfg.ff_max_samples, 1 + step_dist_m / half_wavelength))
    i_s = rng.normal(0.0, math.sqrt(0.5), n)       # E|h|² = 1
    q_s = rng.normal(0.0, math.sqrt(0.5), n)
    if is_los:
        K   = 10.0 ** (cfg.rician_k_db / 10.0)
        i_s = i_s / math.sqrt(K + 1.0) + math.sqrt(K / (K + 1.0))
        q_s = q_s / math.sqrt(K + 1.0)
    mean_power = float(np.mean(i_s ** 2 + q_s ** 2))

    ff_db = float(np.clip(10.0 * math.log10(max(mean_power, 1e-8)),
                          cfg.ff_min_db, cfg.ff_max_db))
    return ff_db, ff_q, ff_db
