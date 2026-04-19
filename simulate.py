"""
LTE Handover Simulation — v3 (major realism upgrade)

Architecture
------------
  src/radio_model.py    — 3GPP UMa LOS/NLOS path loss, shadow fading (AR-1),
                          Rayleigh/Rician fast fading (AR-1 complex Gaussian),
                          load-weighted per-cell SINR
  src/mobility.py       — Random Waypoint model with direction persistence,
                          pauses, pedestrian/vehicle differentiation
  src/handover_logic.py — L3 EMA filtering, velocity-aware TTT, A3/A4/A5
                          classification, enhanced multi-factor HO failure,
                          ping-pong hysteresis

New columns in v3 dataset (all v2 columns retained)
----------------------------------------------------
  l3_rsrp_serving   — L3-filtered serving RSRP (what HO logic actually sees)
  l3_rsrp_neighbor  — L3-filtered best-neighbour RSRP
  los_flag          — 1 if serving cell link is currently LOS
  cell_load_pct     — serving cell load as percentage (0–100)
  rsrp_diff         — raw rsrp_serving − rsrp_neighbor
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.radio_model import (
    RADIO, RadioConfig,
    p_los,
    compute_rsrp, compute_sinr, compute_rsrq, sinr_to_cqi,
    update_shadow, update_fast_fading,
)
from src.mobility import MOB, MobilityConfig, UE, _sample_waypoint
from src.handover_logic import HO, HOConfig, l3_update, evaluate_handover


# ── Simulation parameters ─────────────────────────────────────────────────────

SIM_STEPS           = 1800    # simulation steps (= seconds at DT=1 s)
NUM_UES_PED         = 8       # pedestrian UEs
NUM_UES_VEH         = 7       # vehicle UEs
LABEL_K             = 3       # look-ahead steps for handover_soon label
LOS_REEVAL_INTERVAL = 5       # steps between LOS state re-evaluations
CELL_CAPACITY       = 10      # UEs corresponding to 100 % load


# ── Network: Base Stations ────────────────────────────────────────────────────

@dataclass
class BaseStation:
    bs_id:         int
    x:             float
    y:             float
    tx_power_dbm:  float = RADIO.tx_power_dbm

    def distance_to(self, ux: float, uy: float) -> float:
        return math.hypot(self.x - ux, self.y - uy)


def create_base_stations() -> List[BaseStation]:
    """Four BSs arranged in a square grid — 500 m inter-site distance."""
    positions = [
        (250.0, 250.0),
        (750.0, 250.0),
        (250.0, 750.0),
        (750.0, 750.0),
    ]
    return [BaseStation(bs_id=i, x=x, y=y) for i, (x, y) in enumerate(positions)]


# ── UE factory ────────────────────────────────────────────────────────────────

def create_ues(bss: List[BaseStation],
               rng: np.random.Generator) -> List[UE]:
    """
    Initialise UEs with Random Waypoint state and radio channel states.
    8 pedestrians (0.5–2 m/s) + 7 vehicles (8–20 m/s).
    """
    ues: List[UE] = []

    for i in range(NUM_UES_PED + NUM_UES_VEH):
        is_vehicle = i >= NUM_UES_PED
        lo, hi     = MOB.veh_speed_range if is_vehicle else MOB.ped_speed_range
        speed      = float(rng.uniform(lo, hi))

        x = float(rng.uniform(50.0, MOB.grid_size - 50.0))
        y = float(rng.uniform(50.0, MOB.grid_size - 50.0))
        heading = float(rng.uniform(0.0, 2.0 * math.pi))

        # Closest BS becomes serving cell
        serving = int(np.argmin([bs.distance_to(x, y) for bs in bss]))

        # Initial waypoint
        wp_x, wp_y = _sample_waypoint(x, y, is_vehicle, rng, MOB)

        # Radio channel initial state (per BS)
        shadow = {bs.bs_id: rng.normal(0.0, RADIO.shadow_sigma_nlos) for bs in bss}
        ff_i   = {bs.bs_id: rng.normal(0.0, 1.0) for bs in bss}
        ff_q   = {bs.bs_id: rng.normal(0.0, 1.0) for bs in bss}

        # Initial LOS state sampled from distance-dependent probability
        los = {bs.bs_id: bool(rng.random() < p_los(bs.distance_to(x, y)))
               for bs in bss}

        # L3 filter initialised with first raw RSRP estimate
        l3_rsrp = {}
        for bs in bss:
            d       = bs.distance_to(x, y)
            _, _, ff_db = update_fast_fading(ff_i[bs.bs_id], ff_q[bs.bs_id],
                                             0.0, los[bs.bs_id], rng)
            rsrp_init = compute_rsrp(d, los[bs.bs_id], shadow[bs.bs_id], ff_db)
            l3_rsrp[bs.bs_id] = rsrp_init

        ues.append(UE(
            ue_id=i, x=x, y=y, speed=speed, heading=heading,
            is_vehicle=is_vehicle, serving_cell=serving,
            wp_x=wp_x, wp_y=wp_y,
            shadow=shadow, ff_i=ff_i, ff_q=ff_q, los=los,
            l3_rsrp=l3_rsrp,
        ))

    return ues


# ── Main simulation loop ──────────────────────────────────────────────────────

def run_simulation(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bss = create_base_stations()
    ues = create_ues(bss, rng)

    records = []

    for t in range(SIM_STEPS):

        # ── Cell load snapshot ────────────────────────────────────────────────
        cell_count = Counter(ue.serving_cell for ue in ues)
        cell_load_frac = {
            bs.bs_id: min(1.0, cell_count[bs.bs_id] / CELL_CAPACITY)
            for bs in bss
        }

        for ue in ues:

            # ── Mobility ──────────────────────────────────────────────────────
            step_dist = ue.move(rng)

            # ── Per-BS radio channel update ───────────────────────────────────
            rsrp_raw: dict = {}

            for bs in bss:
                d = bs.distance_to(ue.x, ue.y)

                # LOS/NLOS state (re-evaluate every N steps to model slow changes)
                if t % LOS_REEVAL_INTERVAL == 0:
                    ue.los[bs.bs_id] = bool(rng.random() < p_los(d))
                is_los = ue.los[bs.bs_id]

                # Shadow fading (Gudmundson AR-1, σ depends on LOS/NLOS)
                ue.shadow[bs.bs_id] = update_shadow(
                    ue.shadow.get(bs.bs_id, 0.0),
                    step_dist, is_los, rng,
                )

                # Fast fading (AR-1 complex Gaussian, Rayleigh/Rician)
                new_i, new_q, ff_db = update_fast_fading(
                    ue.ff_i.get(bs.bs_id, 0.0),
                    ue.ff_q.get(bs.bs_id, 0.0),
                    step_dist, is_los, rng,
                )
                ue.ff_i[bs.bs_id] = new_i
                ue.ff_q[bs.bs_id] = new_q

                rsrp_raw[bs.bs_id] = compute_rsrp(
                    d, is_los, ue.shadow[bs.bs_id], ff_db
                )

            # ── L3 measurement filter (EMA, α = HO.l3_alpha) ─────────────────
            for bs_id, rsrp in rsrp_raw.items():
                ue.l3_rsrp[bs_id] = l3_update(
                    ue.l3_rsrp.get(bs_id, rsrp), rsrp, HO.l3_alpha
                )

            # ── Derived KPIs ──────────────────────────────────────────────────
            rsrp_serving = rsrp_raw[ue.serving_cell]
            rsrp_all     = list(rsrp_raw.values())

            # SINR computed from L3-filtered RSRP values.
            # L3-filtered RSRP averages out fast-fading dips, matching the
            # measurement averaging done by real LTE UEs before reporting SINR.
            interferers_l3 = [
                (ue.l3_rsrp[bs.bs_id], cell_load_frac[bs.bs_id])
                for bs in bss if bs.bs_id != ue.serving_cell
            ]

            # Sustained low RSRP counter (for failure model)
            if rsrp_serving < HO.low_rsrp_threshold_dbm:
                ue.steps_at_low_rsrp += 1
            else:
                ue.steps_at_low_rsrp = 0

            # SINR (clamped to −15 dB during RLF recovery)
            if ue.rlf_remaining > 0:
                sinr = -15.0
                ue.rlf_remaining -= 1
            else:
                sinr = compute_sinr(ue.l3_rsrp[ue.serving_cell], interferers_l3)

            rsrq = compute_rsrq(rsrp_serving, rsrp_all)
            cqi  = sinr_to_cqi(sinr)

            # Best neighbour (by raw RSRP, for output reporting)
            best_nb_id = max(
                (bs.bs_id for bs in bss if bs.bs_id != ue.serving_cell),
                key=lambda bid: rsrp_raw[bid],
                default=-1,
            )
            rsrp_neighbor = rsrp_raw[best_nb_id] if best_nb_id >= 0 else RADIO.min_rsrp
            rsrq_neighbor = (compute_rsrq(rsrp_neighbor, rsrp_all)
                             if best_nb_id >= 0 else -20.0)

            l3_srv = ue.l3_rsrp[ue.serving_cell]
            l3_nb  = ue.l3_rsrp[best_nb_id] if best_nb_id >= 0 else RADIO.min_rsrp

            # ── Handover evaluation ───────────────────────────────────────────
            pre_ho_cell = ue.serving_cell

            ho_event, target_id, event_type, ho_failure, ping_pong = \
                evaluate_handover(ue, bss, ue.l3_rsrp, sinr, t, rng)

            if ho_event and not ho_failure:
                ue.serving_cell = target_id

            # ── Record row ────────────────────────────────────────────────────
            records.append({
                "timestamp":              t,
                "ue_id":                  ue.ue_id,
                "serving_cell_id":        pre_ho_cell,
                # Core radio KPIs (instantaneous)
                "rsrp_serving":           round(rsrp_serving, 2),
                "rsrq_serving":           round(rsrq, 2),
                "sinr":                   round(sinr, 2),
                "cqi":                    cqi,
                "best_neighbor_cell_id":  best_nb_id,
                "rsrp_neighbor":          round(rsrp_neighbor, 2),
                "rsrq_neighbor":          round(rsrq_neighbor, 2),
                "rsrp_diff":              round(rsrp_serving - rsrp_neighbor, 2),
                # L3-filtered measurements (what HO logic sees)
                "l3_rsrp_serving":        round(l3_srv, 2),
                "l3_rsrp_neighbor":       round(l3_nb, 2),
                # UE context
                "ue_speed":               round(ue.speed, 2),
                "pos_x":                  round(ue.x, 2),
                "pos_y":                  round(ue.y, 2),
                # Propagation state
                "los_flag":               int(ue.los.get(pre_ho_cell, False)),
                # Cell load
                "serving_cell_load":      cell_count[pre_ho_cell],
                "cell_load_pct":          round(cell_load_frac[pre_ho_cell] * 100.0, 1),
                # Handover events
                "handover_event":         int(ho_event),
                "target_cell_id":         target_id if ho_event else -1,
                "event_type":             event_type,
                "handover_failure":       int(ho_failure),
                "ping_pong":              int(ping_pong),
                "rlf_flag":               int(ue.rlf_remaining > 0),
            })

    return pd.DataFrame(records)


# ── Label generation ──────────────────────────────────────────────────────────

def add_handover_soon_label(df: pd.DataFrame, k: int = LABEL_K) -> pd.DataFrame:
    """
    handover_soon = 1 if a *successful* (non-failed) HO occurs within the
    next k steps for the same UE.  Failed HOs are excluded so the model learns
    to predict actual serving-cell changes, not attempted-and-failed events.

    No label leakage: shift(-i) for i in 1..k never looks at current step.
    """
    df = df.sort_values(["ue_id", "timestamp"]).reset_index(drop=True)

    df["_successful_ho"] = (
        (df["handover_event"] == 1) & (df["handover_failure"] == 0)
    ).astype(int)

    label = pd.Series(0, index=df.index)
    for shift_k in range(1, k + 1):
        label |= (
            df.groupby("ue_id")["_successful_ho"]
              .shift(-shift_k)
              .fillna(0)
              .astype(int)
        )

    df["handover_soon"] = label.astype(int)
    df.drop(columns=["_successful_ho"], inplace=True)
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("LTE Handover Simulation — v3 (enhanced realism)")
    print("=" * 60)

    df  = run_simulation(seed=42)
    df  = add_handover_soon_label(df, k=LABEL_K)

    out = Path("data/raw/dataset.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    ho = df[df["handover_event"] == 1]

    print(f"\nDataset shape   : {df.shape}")
    print(f"UEs             : {df['ue_id'].nunique()}")
    print(f"Steps           : {df['timestamp'].nunique()}")

    print(f"\nHandover events : {len(ho)}")
    if len(ho) > 0:
        n_succ = (ho["handover_failure"] == 0).sum()
        n_fail = ho["handover_failure"].sum()
        n_pp   = ho["ping_pong"].sum()
        print(f"  Successful    : {n_succ}  ({100*n_succ/len(ho):.1f}%)")
        print(f"  Failed (RLF)  : {n_fail}  ({100*n_fail/len(ho):.1f}%)")
        print(f"  Ping-pong     : {n_pp}  ({100*n_pp/len(ho):.1f}%)")

        print(f"\nEvent type breakdown:")
        for ev, cnt in ho["event_type"].value_counts().items():
            print(f"  {ev}: {cnt:4d}  ({100*cnt/len(ho):.1f}%)")

    hs = df["handover_soon"].sum()
    print(f"\nhandover_soon=1 : {hs:,}  ({100*hs/len(df):.1f}%)")

    print(f"\nLOS fraction    : {df['los_flag'].mean()*100:.1f}%")

    print("\nSignal ranges:")
    for col in ["rsrp_serving", "rsrq_serving", "sinr", "cqi",
                "rsrp_neighbor", "rsrp_diff",
                "l3_rsrp_serving", "cell_load_pct"]:
        if col in df.columns:
            print(f"  {col:22s}: [{df[col].min():7.1f}, {df[col].max():6.1f}]")

    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
