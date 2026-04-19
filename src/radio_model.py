"""
Radio propagation model — LTE handover simulation (v3)

Models implemented:
  Path loss  : 3GPP TR 36.873 UMa LOS and NLOS closed-form at 2 GHz
  LOS/NLOS   : 3GPP TR 36.873 distance-dependent probability; persistent state
  Shadow     : Gudmundson AR(1) spatial correlation; σ_LOS=4 dB, σ_NLOS=6 dB
  Fast fading: AR(1) complex Gaussian (Rayleigh NLOS / Rician LOS), spatially
               correlated over decorrelation distance ~2 m (few wavelengths)
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
    tx_power_dbm:     float = 46.0          # eNB Tx power
    thermal_noise_dbm: float = -107.0       # kTB for 10 MHz LTE bandwidth

    min_rsrp: float = -140.0
    max_rsrp: float = -25.0

    # Shadow fading (Gudmundson AR-1)
    shadow_decorr_m:    float = 100.0       # spatial decorrelation distance
    shadow_sigma_los:   float = 4.0         # dB — 3GPP UMa LOS std dev
    shadow_sigma_nlos:  float = 6.0         # dB — 3GPP UMa NLOS std dev

    # Fast fading (AR-1 complex Gaussian)
    # Values represent 1-second averaged small-scale fading (not instantaneous).
    # Realistic range for measurement-averaged fast fading: ±4 dB.
    fast_decorr_m:  float = 3.0             # spatial decorr (few wavelengths @ 2 GHz)
    ff_min_db:      float = -8.0            # deep fade floor (averaged over 1 s)
    ff_max_db:      float = 3.0             # max fast fading gain
    rician_k_db:    float = 5.0             # Rician K-factor for LOS links

    # Interference model
    load_idle_frac: float = 0.10            # fraction of interference when cell idle
    cell_capacity:  int   = 10              # UEs → 100 % load


# Module-level default instance
RADIO = RadioConfig()

# Normalisation constant: E[r] for I,Q ~ N(0, 0.5) each
# r = sqrt(I² + Q²) follows Rayleigh with parameter σ_R = 1/√2
# E[r] = σ_R × √(π/2) = (1/√2) × √(π/2) = √(π/4)
_RAYLEIGH_E_AMP = math.sqrt(math.pi / 4.0)   # ≈ 0.8862


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
                 cfg: RadioConfig = RADIO) -> float:
    """
    RSRP = Tx_power − PL(d, LOS/NLOS) + shadow_db + fast_fade_db
    Clamped to [min_rsrp, max_rsrp].
    """
    pl   = path_loss(d_m, is_los, cfg.carrier_freq_ghz)
    rsrp = cfg.tx_power_dbm - pl + shadow_db + fast_fade_db
    return float(np.clip(rsrp, cfg.min_rsrp, cfg.max_rsrp))


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
    n_mw = 10.0 ** (cfg.thermal_noise_dbm / 10.0)

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
    AR(1) complex Gaussian small-scale fading update.

    State (ff_i, ff_q) evolves with spatial correlation ρ = exp(−Δd / d_corr).
    For NLOS: amplitude follows Rayleigh distribution.
    For LOS:  Rician K-factor adds a deterministic in-phase component.

    The output fast_fade_db is normalised so that E[fast_fade_db] ≈ −1 dB
    (Rayleigh average) and slightly positive for Rician LOS.

    Returns
    -------
    new_ff_i, new_ff_q, fast_fade_db
    """
    rho        = math.exp(-step_dist_m / cfg.fast_decorr_m)
    noise_std  = math.sqrt(max(1.0 - rho ** 2, 0.0))

    ff_i = rho * ff_i + rng.normal(0.0, noise_std)
    ff_q = rho * ff_q + rng.normal(0.0, noise_std)

    if is_los:
        # Rician: superimpose deterministic LOS component on I channel
        K          = 10.0 ** (cfg.rician_k_db / 10.0)       # linear K-factor
        scatter    = 1.0 / math.sqrt(K + 1.0)                # scatter amplitude scale
        los_comp   = math.sqrt(K / (K + 1.0))                # LOS component
        I_eff = ff_i * scatter + los_comp
        Q_eff = ff_q * scatter
    else:
        # Rayleigh: normalise so each component has variance 0.5
        I_eff = ff_i / math.sqrt(2.0)
        Q_eff = ff_q / math.sqrt(2.0)

    amplitude   = math.hypot(I_eff, Q_eff)
    # Normalise by expected Rayleigh amplitude (makes NLOS ≈ 0 dB mean)
    amp_norm    = amplitude / max(_RAYLEIGH_E_AMP, 1e-12)
    ff_db       = 20.0 * math.log10(max(amp_norm, 1e-8))
    ff_db       = float(np.clip(ff_db, cfg.ff_min_db, cfg.ff_max_db))

    return ff_i, ff_q, ff_db
