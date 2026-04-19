"""
Handover state machine — LTE handover simulation (v3)

Improvements over v2
--------------------
L3 filtering
    UE applies an exponential moving average (α = 0.5) to raw RSRP measurements
    before feeding them to the HO trigger.  This models the Layer-1 / Layer-3
    measurement filter mandated by 3GPP TS 36.331 and prevents false triggers
    from instantaneous fast-fading dips.

Velocity-aware TTT
    Time-to-trigger is shortened for faster UEs:
        TTT = max(TTT_min, TTT_base − floor(speed / speed_divisor))
    A vehicle at 14 m/s with base TTT=3 steps gets TTT=1 step; a pedestrian
    at 1 m/s keeps TTT=3.  This reduces the "too-late" handover problem.

A3 / A4 / A5 event classification (on L3-filtered RSRP)
    A5  serving_L3 < −68 dBm  AND  neighbour_L3 > −60 dBm  (coverage emergency)
    A4  neighbour_L3 > −55 dBm                               (strong target)
    A3  neighbour_L3 > serving_L3 + margin_db                (relative, default)

Enhanced HO failure probability
    P(fail) is a weighted combination of four factors:
      1. Serving SINR         — low SINR at trigger → hard to execute HO
      2. UE speed             — fast UE may move away before HO completes
      3. Target RSRP          — weak target BS → attachment may fail
      4. Sustained low RSRP   — UE has been at cell edge for many steps

Ping-pong hysteresis
    After a ping-pong is detected the REVERSE cell pair (target → serving) gets
    an extra HO margin added for pp_margin_decay_steps steps.  This prevents
    the UE from immediately bouncing back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class HOConfig:
    # A3 relative threshold
    ho_margin_db: float = 3.0
    ttt_base_steps: int = 3
    ttt_min_steps:  int = 1
    ttt_speed_divisor: float = 7.0   # m/s per TTT step reduction

    # A4 absolute threshold (neighbour RSRP)
    a4_threshold_dbm: float = -55.0

    # A5 double thresholds
    a5_serving_threshold_dbm:  float = -68.0
    a5_neighbor_threshold_dbm: float = -60.0

    # L3 measurement filter coefficient (0 = frozen, 1 = no filter)
    l3_alpha: float = 0.5

    # HO failure model weights
    # Calibrated so typical A3 HO (SINR ≈ 0–5 dB) fails ~8-12 %,
    # A5 HO (SINR ≈ −10–−15 dB) fails ~25-35 %, overall ~15 %.
    sinr_fail_pivot_db: float = -12.0   # SINR at which p_sinr = 0.5
    sinr_fail_scale_db: float = 5.0     # sigmoid steepness (softer)
    speed_fail_pivot:   float = 28.0    # m/s at which p_speed = 0.5 (above veh max)
    speed_fail_scale:   float = 6.0
    target_rsrp_start:  float = -70.0   # RSRP below this starts adding failure risk
    target_rsrp_end:    float = -105.0  # RSRP here → full target contribution
    low_rsrp_decay:     int   = 25      # steps of low RSRP → max sustained contribution
    # weights (must sum to 1.0)
    w_sinr:      float = 0.50
    w_speed:     float = 0.15
    w_target:    float = 0.15
    w_sustained: float = 0.20
    fail_cap:    float = 0.75           # max failure probability

    # Ping-pong
    ping_pong_window:     int   = 10
    pp_extra_margin_db:   float = 3.0   # extra HO margin after ping-pong
    pp_margin_decay_steps: int  = 20    # steps until extra margin expires

    # RLF
    rlf_duration_steps: int = 3

    # Low RSRP threshold (for sustained counter)
    low_rsrp_threshold_dbm: float = -85.0


HO = HOConfig()


# ── L3 measurement filter ─────────────────────────────────────────────────────

def l3_update(current: float, new_meas: float, alpha: float) -> float:
    """Exponential moving average: α × new + (1−α) × old."""
    return alpha * new_meas + (1.0 - alpha) * current


# ── Velocity-aware TTT ────────────────────────────────────────────────────────

def velocity_ttt(speed_ms: float, cfg: HOConfig = HO) -> int:
    """Compute TTT adapted to UE speed (faster → shorter TTT)."""
    reduction = int(speed_ms / cfg.ttt_speed_divisor)
    return max(cfg.ttt_min_steps, cfg.ttt_base_steps - reduction)


# ── Event type classification ─────────────────────────────────────────────────

def classify_event(l3_rsrp_serving: float, l3_rsrp_neighbor: float,
                   cfg: HOConfig = HO) -> str:
    """Classify the dominant 3GPP measurement event on L3-filtered RSRP values."""
    if (l3_rsrp_serving  < cfg.a5_serving_threshold_dbm and
            l3_rsrp_neighbor > cfg.a5_neighbor_threshold_dbm):
        return "A5"
    if l3_rsrp_neighbor > cfg.a4_threshold_dbm:
        return "A4"
    return "A3"


# ── Enhanced HO failure probability ──────────────────────────────────────────

def ho_failure_prob(sinr_db: float, speed_ms: float,
                    target_rsrp_dbm: float, steps_low: int,
                    cfg: HOConfig = HO) -> float:
    """
    Weighted multi-factor HO failure probability.

    Factor 1 (SINR):
        Sigmoid centred at sinr_fail_pivot_db.
        Low SINR → high probability of radio link failure during HO execution.

    Factor 2 (speed):
        Sigmoid centred at speed_fail_pivot m/s.
        Very fast UEs have less time for the HO procedure to complete.

    Factor 3 (target RSRP):
        Linear ramp from 0 at target_rsrp_start to 1 at target_rsrp_end.
        Weak target BS → attachment and RRC reconfiguration may fail.

    Factor 4 (sustained low RSRP):
        Fraction of low_rsrp_decay steps already spent at low RSRP.
        UE stuck at cell edge → likely in deep shadow / coverage hole.
    """
    # 1. SINR sigmoid
    x_sinr  = -(sinr_db - cfg.sinr_fail_pivot_db) / cfg.sinr_fail_scale_db
    p_sinr  = 1.0 / (1.0 + math.exp(-x_sinr))

    # 2. Speed sigmoid
    x_speed = (speed_ms - cfg.speed_fail_pivot) / cfg.speed_fail_scale
    p_speed = 1.0 / (1.0 + math.exp(-x_speed))

    # 3. Target RSRP linear
    span    = cfg.target_rsrp_end - cfg.target_rsrp_start   # negative number
    p_tgt   = float(np.clip((target_rsrp_dbm - cfg.target_rsrp_start) / span, 0.0, 1.0))

    # 4. Sustained low RSRP
    p_sus   = min(1.0, steps_low / max(cfg.low_rsrp_decay, 1))

    p_fail  = (cfg.w_sinr     * p_sinr  +
               cfg.w_speed    * p_speed +
               cfg.w_target   * p_tgt   +
               cfg.w_sustained * p_sus)

    return float(np.clip(p_fail, 0.0, cfg.fail_cap))


# ── Ping-pong detection ───────────────────────────────────────────────────────

def is_ping_pong(ho_history: List[Tuple[int, int, int]],
                 from_cell: int, to_cell: int,
                 current_t: int, cfg: HOConfig = HO) -> bool:
    """
    Return True if this HO (from_cell → to_cell) is a ping-pong:
    the UE recently executed the reverse HO (to_cell → from_cell) within
    the ping-pong detection window.
    """
    for h_from, h_to, h_t in reversed(ho_history):
        if current_t - h_t > cfg.ping_pong_window:
            break
        if h_from == to_cell and h_to == from_cell:
            return True
    return False


# ── Main HO evaluation function ───────────────────────────────────────────────

def evaluate_handover(
    ue,                              # mobility.UE instance
    bss:          list,              # list of BaseStation
    l3_rsrp_map:  Dict[int, float],  # L3-filtered RSRP per bs_id
    raw_sinr_db:  float,             # current serving-cell SINR (for failure model)
    current_t:    int,
    rng:          np.random.Generator,
    cfg:          HOConfig = HO,
) -> Tuple[bool, int, str, bool, bool]:
    """
    Run the A3 TTT state machine and, on TTT expiry, classify the event and
    determine outcome (success / failure / ping-pong).

    Returns
    -------
    ho_event       : True if HO attempt was made this step
    target_cell_id : best neighbour id (−1 if none)
    event_type     : 'A3' | 'A4' | 'A5' | 'none'
    ho_failure     : True if HO attempted but failed (→ RLF)
    ping_pong      : True if this successful HO reversed a recent one
    """
    if ue.rlf_remaining > 0:
        return False, -1, "none", False, False

    l3_srv = l3_rsrp_map.get(ue.serving_cell, -140.0)

    # Find best neighbour by L3-filtered RSRP
    best_nb_id  = -1
    best_nb_l3  = -999.0
    for bs in bss:
        if bs.bs_id == ue.serving_cell:
            continue
        r = l3_rsrp_map.get(bs.bs_id, -140.0)
        if r > best_nb_l3:
            best_nb_l3 = r
            best_nb_id = bs.bs_id

    # Effective A3 margin: base + any active ping-pong hysteresis for this pair
    pp_key       = (ue.serving_cell, best_nb_id)
    extra_margin = 0.0
    if pp_key in ue.pp_extra_margin:
        extra_margin = ue.pp_extra_margin[pp_key]
        # Decay the hysteresis counter
        ue.pp_margin_remaining[pp_key] = ue.pp_margin_remaining.get(pp_key, 0) - 1
        if ue.pp_margin_remaining.get(pp_key, 0) <= 0:
            ue.pp_extra_margin.pop(pp_key, None)
            ue.pp_margin_remaining.pop(pp_key, None)

    effective_margin = cfg.ho_margin_db + extra_margin
    a3_condition     = (best_nb_l3 > l3_srv + effective_margin)

    # TTT state machine
    ttt_steps = velocity_ttt(ue.speed, cfg)

    if a3_condition and ue.ttt_candidate == best_nb_id:
        ue.ttt_counter += 1
    elif a3_condition:
        ue.ttt_candidate = best_nb_id
        ue.ttt_counter   = 1
    else:
        ue.ttt_counter   = 0
        ue.ttt_candidate = None

    if ue.ttt_counter < ttt_steps:
        return False, best_nb_id, "none", False, False

    # ── TTT expired: attempt handover ────────────────────────────────────────
    ue.ttt_counter   = 0
    ue.ttt_candidate = None

    event_type = classify_event(l3_srv, best_nb_l3, cfg)

    # Enhanced failure probability
    target_rsrp = l3_rsrp_map.get(best_nb_id, -140.0) if best_nb_id >= 0 else -140.0
    p_fail      = ho_failure_prob(raw_sinr_db, ue.speed, target_rsrp,
                                  ue.steps_at_low_rsrp, cfg)
    ho_failed   = bool(rng.random() < p_fail)

    # Ping-pong detection
    pp = is_ping_pong(ue.ho_history, ue.serving_cell, best_nb_id, current_t, cfg)

    # Record in HO history
    ue.ho_history.append((ue.serving_cell, best_nb_id, current_t))
    if len(ue.ho_history) > 30:
        ue.ho_history.pop(0)

    # If ping-pong on a successful HO: apply extra margin on the reverse pair
    if pp and not ho_failed:
        rev_key = (best_nb_id, ue.serving_cell)
        ue.pp_extra_margin[rev_key]     = cfg.pp_extra_margin_db
        ue.pp_margin_remaining[rev_key] = cfg.pp_margin_decay_steps

    if ho_failed:
        ue.rlf_remaining = cfg.rlf_duration_steps
        return True, best_nb_id, event_type, True, pp

    return True, best_nb_id, event_type, False, pp
