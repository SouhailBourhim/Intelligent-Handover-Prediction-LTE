"""
Shared LTE simulation engine (v4).

One step loop used by simulate.py (training dataset), generate_test_dataset.py
(scenario CSVs), src/generalization.py and src/closed_loop.py.

Each 1-second step runs in two phases:

  1. measure  — every UE moves, its channel to every eNB is updated
                (LOS, shadowing, fast fading), RSRP is L3-filtered and the
                serving-cell KPIs (SINR, RSRQ, CQI) are computed;
  2. decide   — an optional TTT policy sees all measurements of the step,
                then the A3 handover state machine runs for every UE.

Randomness comes from two independent streams: `rng` drives mobility and the
radio channel, `ho_rng` drives handover-failure draws.  Two runs with the same
seed therefore see identical trajectories and fading even when a policy
changes handover decisions, which is what the closed-loop comparison needs.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.radio_model import (
    RADIO,
    compute_rsrp, compute_rsrq, compute_sinr, sinr_to_cqi, report_rsrp,
    los_from_latent, update_los_latent, update_shadow, update_fast_fading,
)
from src.mobility import MOB, UE, _sample_waypoint
from src.handover_logic import HO, l3_update, evaluate_handover


LABEL_K       = 3       # look-ahead steps for the handover_soon label
CELL_CAPACITY = 10      # UEs corresponding to 100 % load
RLF_SINR_DB   = -15.0   # SINR reported while the UE is in radio-link failure

BS_POSITIONS: List[Tuple[float, float]] = [
    (250.0, 250.0),
    (750.0, 250.0),
    (250.0, 750.0),
    (750.0, 750.0),
]

# A policy receives (t, measurements of every UE this step) and returns
# {ue_id: ttt_steps} for the UEs whose TTT it wants to override.
TTTPolicy = Callable[[int, List[dict]], Dict[int, int]]


# ── Network elements ──────────────────────────────────────────────────────────

@dataclass
class BaseStation:
    bs_id:        int
    x:            float
    y:            float
    tx_power_dbm: float = RADIO.tx_power_dbm

    def distance_to(self, ux: float, uy: float) -> float:
        return math.hypot(self.x - ux, self.y - uy)


def create_base_stations() -> List[BaseStation]:
    """Four eNBs on a square grid, 500 m inter-site distance."""
    return [BaseStation(bs_id=i, x=x, y=y) for i, (x, y) in enumerate(BS_POSITIONS)]


def init_ue(ue_id: int, x: float, y: float, speed_range: Tuple[float, float],
            is_vehicle: bool, bss: List[BaseStation],
            rng: np.random.Generator) -> UE:
    """Create a UE with its mobility state and per-eNB channel state."""
    speed   = float(rng.uniform(*speed_range))
    heading = float(rng.uniform(0.0, 2.0 * math.pi))
    serving = int(np.argmin([bs.distance_to(x, y) for bs in bss]))
    wp_x, wp_y = _sample_waypoint(x, y, is_vehicle, rng, MOB)

    shadow = {bs.bs_id: rng.normal(0.0, RADIO.shadow_sigma_nlos) for bs in bss}
    ff_i   = {bs.bs_id: rng.normal(0.0, 1.0) for bs in bss}
    ff_q   = {bs.bs_id: rng.normal(0.0, 1.0) for bs in bss}
    los_z  = {bs.bs_id: rng.normal() for bs in bss}
    los    = {bs.bs_id: los_from_latent(los_z[bs.bs_id], bs.distance_to(x, y))
              for bs in bss}

    # L3 filter starts from the first measurement
    l3_rsrp = {}
    for bs in bss:
        _, _, ff_db = update_fast_fading(ff_i[bs.bs_id], ff_q[bs.bs_id],
                                         0.0, los[bs.bs_id], rng)
        l3_rsrp[bs.bs_id] = compute_rsrp(bs.distance_to(x, y), los[bs.bs_id],
                                         shadow[bs.bs_id], ff_db, report=False)

    return UE(
        ue_id=ue_id, x=x, y=y, speed=speed, heading=heading,
        is_vehicle=is_vehicle, serving_cell=serving,
        speed_range=speed_range, current_speed=speed,
        wp_x=wp_x, wp_y=wp_y,
        shadow=shadow, ff_i=ff_i, ff_q=ff_q, los=los, los_z=los_z,
        l3_rsrp=l3_rsrp,
    )


def create_mixed_ues(n_ped: int, n_veh: int, bss: List[BaseStation],
                     rng: np.random.Generator) -> List[UE]:
    """The training configuration: pedestrians first, then vehicles."""
    ues = []
    for i in range(n_ped + n_veh):
        is_vehicle = i >= n_ped
        rng_speed  = MOB.veh_speed_range if is_vehicle else MOB.ped_speed_range
        x = float(rng.uniform(50.0, MOB.grid_size - 50.0))
        y = float(rng.uniform(50.0, MOB.grid_size - 50.0))
        ues.append(init_ue(i, x, y, rng_speed, is_vehicle, bss, rng))
    return ues


# ── Engine ────────────────────────────────────────────────────────────────────

class Simulator:
    """Step-by-step LTE network simulation over a fixed set of eNBs and UEs."""

    def __init__(self, bss: List[BaseStation], ues: List[UE],
                 rng: np.random.Generator, ho_rng: np.random.Generator,
                 ttt_policy: Optional[TTTPolicy] = None):
        self.bss        = bss
        self.ues        = ues
        self.rng        = rng
        self.ho_rng     = ho_rng
        self.ttt_policy = ttt_policy

    # Phase 1 ─────────────────────────────────────────────────────────────────
    def _measure(self, ue: UE, cell_load: Dict[int, float]) -> dict:
        step_dist = ue.move(self.rng)

        rsrp: Dict[int, float] = {}
        for bs in self.bss:
            d   = bs.distance_to(ue.x, ue.y)
            bid = bs.bs_id
            ue.los_z[bid] = update_los_latent(ue.los_z[bid], step_dist, self.rng)
            ue.los[bid]   = los_from_latent(ue.los_z[bid], d)
            ue.shadow[bid] = update_shadow(ue.shadow[bid], step_dist,
                                           ue.los[bid], self.rng)
            ue.ff_i[bid], ue.ff_q[bid], ff_db = update_fast_fading(
                ue.ff_i[bid], ue.ff_q[bid], step_dist, ue.los[bid], self.rng)
            # Physical (unclamped) RSRP; clamped only when written out
            rsrp[bid] = compute_rsrp(d, ue.los[bid], ue.shadow[bid], ff_db,
                                     report=False)

        for bid, r in rsrp.items():
            ue.l3_rsrp[bid] = l3_update(ue.l3_rsrp[bid], r, HO.l3_alpha)

        srv       = ue.serving_cell
        rsrp_srv  = rsrp[srv]
        rsrp_all  = list(rsrp.values())
        in_rlf    = ue.rlf_remaining > 0

        ue.steps_at_low_rsrp = (ue.steps_at_low_rsrp + 1
                                if rsrp_srv < HO.low_rsrp_threshold_dbm else 0)

        if in_rlf:
            sinr = RLF_SINR_DB
        else:
            interferers = [(ue.l3_rsrp[bs.bs_id], cell_load[bs.bs_id])
                           for bs in self.bss if bs.bs_id != srv]
            sinr = compute_sinr(ue.l3_rsrp[srv], interferers)

        nb      = max((b for b in rsrp if b != srv), key=lambda b: rsrp[b])
        l3_nb   = max((b for b in rsrp if b != srv), key=lambda b: ue.l3_rsrp[b])
        rsrp_nb = rsrp[nb]

        return {
            "ue_id":                 ue.ue_id,
            "serving_cell_id":       srv,
            "rsrp_serving":          round(report_rsrp(rsrp_srv), 2),
            "rsrq_serving":          round(compute_rsrq(rsrp_srv, rsrp_all), 2),
            "sinr":                  round(sinr, 2),
            "cqi":                   sinr_to_cqi(sinr),
            "best_neighbor_cell_id": nb,
            "rsrp_neighbor":         round(report_rsrp(rsrp_nb), 2),
            "rsrq_neighbor":         round(compute_rsrq(rsrp_nb, rsrp_all), 2),
            "rsrp_diff":             round(report_rsrp(rsrp_srv) - report_rsrp(rsrp_nb), 2),
            "l3_rsrp_serving":       round(report_rsrp(ue.l3_rsrp[srv]), 2),
            "l3_rsrp_neighbor":      round(report_rsrp(ue.l3_rsrp[nb]), 2),
            "ue_speed":              round(ue.current_speed, 2),
            "pos_x":                 round(ue.x, 2),
            "pos_y":                 round(ue.y, 2),
            "los_flag":              int(ue.los[srv]),
            "serving_cell_load":     int(round(cell_load[srv] * CELL_CAPACITY)),
            "cell_load_pct":         round(cell_load[srv] * 100.0, 1),
            # Internal fields for policies (stripped before output)
            "_a3_gap":               ue.l3_rsrp[l3_nb] - ue.l3_rsrp[srv],
            "_in_rlf":               in_rlf,
        }

    # Phase 2 ─────────────────────────────────────────────────────────────────
    def _decide(self, ue: UE, rec: dict, t: int, ttt: Optional[int]) -> None:
        ho_event, target, event_type, failed, pp = evaluate_handover(
            ue, self.bss, ue.l3_rsrp, rec["sinr"], t, self.ho_rng,
            ttt_override=ttt,
        )
        if ho_event and not failed:
            ue.serving_cell = target

        # RLF countdown runs after the HO evaluation, so a failure blocks
        # handovers for exactly rlf_duration_steps steps.  When it ends the
        # UE re-establishes on the strongest cell (RRC re-establishment).
        reestablish = 0
        if rec["_in_rlf"]:
            ue.rlf_remaining -= 1
            if ue.rlf_remaining == 0:
                best = max(ue.l3_rsrp, key=ue.l3_rsrp.get)
                if best != ue.serving_cell:
                    ue.serving_cell = best
                    reestablish = 1

        rec.update({
            "handover_event":   int(ho_event),
            "target_cell_id":   target if ho_event else -1,
            "event_type":       event_type,
            "handover_failure": int(failed),
            "ping_pong":        int(pp),
            "rlf_flag":         int(rec["_in_rlf"] or failed),
            "rrc_reestablish":  reestablish,
        })

    def step(self, t: int) -> List[dict]:
        counts    = Counter(ue.serving_cell for ue in self.ues)
        cell_load = {bs.bs_id: min(1.0, counts[bs.bs_id] / CELL_CAPACITY)
                     for bs in self.bss}

        recs = [self._measure(ue, cell_load) for ue in self.ues]
        for r in recs:
            r["timestamp"] = t

        overrides = self.ttt_policy(t, recs) if self.ttt_policy else {}
        for ue, rec in zip(self.ues, recs):
            self._decide(ue, rec, t, overrides.get(ue.ue_id))
        return recs

    def run(self, n_steps: int) -> pd.DataFrame:
        rows = []
        for t in range(n_steps):
            rows.extend(self.step(t))
        df = pd.DataFrame(rows)
        df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
        first = ["timestamp", "ue_id"]
        return df[first + [c for c in df.columns if c not in first]]


def make_simulator(ue_factory: Callable[[List[BaseStation], np.random.Generator], List[UE]],
                   seed: int, ttt_policy: Optional[TTTPolicy] = None) -> Simulator:
    """Build a Simulator whose channel and HO-failure streams derive from `seed`."""
    rng = np.random.default_rng(seed)
    bss = create_base_stations()
    ues = ue_factory(bss, rng)
    return Simulator(bss, ues, rng=rng,
                     ho_rng=np.random.default_rng([seed, 0x484F]),
                     ttt_policy=ttt_policy)


# ── Label ─────────────────────────────────────────────────────────────────────

def add_handover_soon_label(df: pd.DataFrame, k: int = LABEL_K) -> pd.DataFrame:
    """
    handover_soon = 1 if a *successful* handover happens for the same UE in
    one of the next k steps.  Failed attempts are excluded so the model learns
    to anticipate completed handovers, not radio-link failures.
    """
    df = df.sort_values(["ue_id", "timestamp"]).reset_index(drop=True)
    success = ((df["handover_event"] == 1) & (df["handover_failure"] == 0)).astype(int)
    label = pd.Series(0, index=df.index)
    for s in range(1, k + 1):
        label |= success.groupby(df["ue_id"]).shift(-s).fillna(0).astype(int)
    df["handover_soon"] = label.astype(int)
    return df
