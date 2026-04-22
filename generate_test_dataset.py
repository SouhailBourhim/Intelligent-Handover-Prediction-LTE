"""
generate_test_dataset.py — Generate a small, prediction-ready LTE measurement dataset.

Uses the same 3GPP UMa radio model, Random Waypoint mobility, and A3/A4/A5 handover
logic as the main simulator, but with configurable scenario presets so you can quickly
produce datasets that exercise different prediction conditions.

The output CSV has exactly the columns the dashboard's 🔮 Live Prediction tab expects.
`handover_soon` is included as ground truth so you can check whether predictions match.

Usage
-----
# Default: 5 UEs (mixed ped + veh), 300 steps, saves to data/test_scenarios/
python generate_test_dataset.py

# Vehicle scenario — lots of handovers
python generate_test_dataset.py --scenario vehicle --ues 4 --steps 500

# Stable pedestrian scenario — very few handovers
python generate_test_dataset.py --scenario stable --ues 3 --steps 200

# Cell-edge scenario — UEs placed and steered near boundaries
python generate_test_dataset.py --scenario cell_edge --steps 400

# Custom seed and output path
python generate_test_dataset.py --seed 7 --output my_test.csv

Scenarios
---------
  default      Mixed pedestrians + vehicles, random positions  [~9% HO rate]
  vehicle      All vehicles at high speed (8–20 m/s)           [~15-20% HO rate]
  stable       Slow pedestrians near cell centres              [~2-4% HO rate]
  cell_edge    UEs initialised near cell boundaries, steered   [~12-18% HO rate]
               along the edges — maximises handover density

Output columns
--------------
  Metadata     : timestamp, ue_id, serving_cell_id
  Raw signals  : rsrp_serving, rsrq_serving, sinr, cqi,
                 rsrp_neighbor, rsrq_neighbor, rsrp_diff
  L3-filtered  : l3_rsrp_serving, l3_rsrp_neighbor
  UE context   : ue_speed, pos_x, pos_y, los_flag, cell_load_pct
  HO events    : handover_event, target_cell_id, event_type,
                 handover_failure, ping_pong, rlf_flag
  Ground truth : handover_soon  (1 if successful HO within next 3 steps)
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.radio_model import (
    RADIO,
    p_los,
    compute_rsrp, compute_sinr, compute_rsrq, sinr_to_cqi,
    update_shadow, update_fast_fading,
)
from src.mobility import MOB, UE, _sample_waypoint
from src.handover_logic import HO, l3_update, evaluate_handover

# ── Constants ──────────────────────────────────────────────────────────────────

LABEL_K          = 3    # steps ahead for handover_soon label
CELL_CAPACITY    = 10   # UEs → 100% load
LOS_REEVAL       = 5    # steps between LOS re-evaluations

BS_POSITIONS = [
    (250.0, 250.0),
    (750.0, 250.0),
    (250.0, 750.0),
    (750.0, 750.0),
]

# Cell-boundary midpoints — used to seed UEs near edges in cell_edge scenario
EDGE_SEEDS = [
    (500.0, 250.0),  # between BS0 and BS1
    (250.0, 500.0),  # between BS0 and BS2
    (500.0, 500.0),  # grid centre (between all 4)
    (750.0, 500.0),  # between BS1 and BS3
    (500.0, 750.0),  # between BS2 and BS3
]


# ── Base station dataclass ─────────────────────────────────────────────────────

@dataclass
class BaseStation:
    bs_id: int
    x: float
    y: float
    tx_power_dbm: float = field(default_factory=lambda: RADIO.tx_power_dbm)

    def distance_to(self, ux: float, uy: float) -> float:
        return math.hypot(self.x - ux, self.y - uy)


# ── UE factory ─────────────────────────────────────────────────────────────────

def _init_ue_radio(ue_id: int, x: float, y: float, speed: float,
                   heading: float, is_vehicle: bool,
                   bss: List[BaseStation],
                   rng: np.random.Generator) -> UE:
    """Create a UE with fully initialised radio channel state."""
    serving = int(np.argmin([bs.distance_to(x, y) for bs in bss]))
    wp_x, wp_y = _sample_waypoint(x, y, is_vehicle, rng, MOB)

    shadow = {bs.bs_id: rng.normal(0.0, RADIO.shadow_sigma_nlos) for bs in bss}
    ff_i   = {bs.bs_id: rng.normal(0.0, 1.0) for bs in bss}
    ff_q   = {bs.bs_id: rng.normal(0.0, 1.0) for bs in bss}
    los    = {bs.bs_id: bool(rng.random() < p_los(bs.distance_to(x, y))) for bs in bss}

    l3_rsrp = {}
    for bs in bss:
        d = bs.distance_to(x, y)
        _, _, ff_db = update_fast_fading(ff_i[bs.bs_id], ff_q[bs.bs_id],
                                         0.0, los[bs.bs_id], rng)
        l3_rsrp[bs.bs_id] = compute_rsrp(d, los[bs.bs_id], shadow[bs.bs_id], ff_db)

    return UE(
        ue_id=ue_id, x=x, y=y, speed=speed, heading=heading,
        is_vehicle=is_vehicle, serving_cell=serving,
        wp_x=wp_x, wp_y=wp_y,
        shadow=shadow, ff_i=ff_i, ff_q=ff_q, los=los, l3_rsrp=l3_rsrp,
    )


def create_ues_for_scenario(scenario: str, n_ues: int,
                             bss: List[BaseStation],
                             rng: np.random.Generator) -> List[UE]:
    ues = []

    for i in range(n_ues):
        if scenario == "stable":
            # Slow pedestrians placed near BS centres (low HO probability)
            bs_idx = i % len(bss)
            bx, by = bss[bs_idx].x, bss[bs_idx].y
            x = float(np.clip(rng.normal(bx, 60), 50, MOB.grid_size - 50))
            y = float(np.clip(rng.normal(by, 60), 50, MOB.grid_size - 50))
            lo, hi    = MOB.ped_speed_range
            speed     = float(rng.uniform(lo, min(hi, 1.2)))   # extra slow
            is_vehicle = False

        elif scenario == "vehicle":
            # Fast vehicles, random positions
            x = float(rng.uniform(50, MOB.grid_size - 50))
            y = float(rng.uniform(50, MOB.grid_size - 50))
            lo, hi    = MOB.veh_speed_range
            speed     = float(rng.uniform(lo, hi))
            is_vehicle = True

        elif scenario == "cell_edge":
            # UEs seeded near cell boundaries with speed that varies ped/veh
            seed_pos   = EDGE_SEEDS[i % len(EDGE_SEEDS)]
            x = float(np.clip(seed_pos[0] + rng.normal(0, 30), 50, MOB.grid_size - 50))
            y = float(np.clip(seed_pos[1] + rng.normal(0, 30), 50, MOB.grid_size - 50))
            # Alternate between slower and faster UEs
            if i % 2 == 0:
                lo, hi    = MOB.veh_speed_range
                is_vehicle = True
            else:
                lo, hi    = (3.0, 8.0)   # "fast pedestrian / slow vehicle"
                is_vehicle = False
            speed = float(rng.uniform(lo, hi))

        else:   # default — mixed
            x = float(rng.uniform(50, MOB.grid_size - 50))
            y = float(rng.uniform(50, MOB.grid_size - 50))
            is_vehicle = (i >= n_ues // 2)
            lo, hi = MOB.veh_speed_range if is_vehicle else MOB.ped_speed_range
            speed  = float(rng.uniform(lo, hi))

        heading = float(rng.uniform(0.0, 2.0 * math.pi))
        ues.append(_init_ue_radio(i, x, y, speed, heading, is_vehicle, bss, rng))

    return ues


# ── Simulation loop ────────────────────────────────────────────────────────────

def run(scenario: str, n_ues: int, n_steps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bss = [BaseStation(i, x, y) for i, (x, y) in enumerate(BS_POSITIONS)]
    ues = create_ues_for_scenario(scenario, n_ues, bss, rng)

    records = []

    for t in range(n_steps):
        cell_count    = Counter(ue.serving_cell for ue in ues)
        cell_load_frac = {bs.bs_id: min(1.0, cell_count[bs.bs_id] / CELL_CAPACITY)
                          for bs in bss}

        for ue in ues:
            step_dist = ue.move(rng)

            rsrp_raw: dict = {}
            for bs in bss:
                d = bs.distance_to(ue.x, ue.y)
                if t % LOS_REEVAL == 0:
                    ue.los[bs.bs_id] = bool(rng.random() < p_los(d))
                is_los = ue.los[bs.bs_id]

                ue.shadow[bs.bs_id] = update_shadow(
                    ue.shadow.get(bs.bs_id, 0.0), step_dist, is_los, rng)

                new_i, new_q, ff_db = update_fast_fading(
                    ue.ff_i.get(bs.bs_id, 0.0),
                    ue.ff_q.get(bs.bs_id, 0.0),
                    step_dist, is_los, rng)
                ue.ff_i[bs.bs_id] = new_i
                ue.ff_q[bs.bs_id] = new_q
                rsrp_raw[bs.bs_id] = compute_rsrp(d, is_los, ue.shadow[bs.bs_id], ff_db)

            for bs_id, rsrp in rsrp_raw.items():
                ue.l3_rsrp[bs_id] = l3_update(
                    ue.l3_rsrp.get(bs_id, rsrp), rsrp, HO.l3_alpha)

            rsrp_serving = rsrp_raw[ue.serving_cell]
            rsrp_all     = list(rsrp_raw.values())

            interferers_l3 = [
                (ue.l3_rsrp[bs.bs_id], cell_load_frac[bs.bs_id])
                for bs in bss if bs.bs_id != ue.serving_cell
            ]

            if rsrp_serving < HO.low_rsrp_threshold_dbm:
                ue.steps_at_low_rsrp += 1
            else:
                ue.steps_at_low_rsrp = 0

            if ue.rlf_remaining > 0:
                sinr = -15.0
                ue.rlf_remaining -= 1
            else:
                sinr = compute_sinr(ue.l3_rsrp[ue.serving_cell], interferers_l3)

            rsrq = compute_rsrq(rsrp_serving, rsrp_all)
            cqi  = sinr_to_cqi(sinr)

            best_nb_id = max(
                (bs.bs_id for bs in bss if bs.bs_id != ue.serving_cell),
                key=lambda bid: rsrp_raw[bid], default=-1,
            )
            rsrp_neighbor = rsrp_raw[best_nb_id] if best_nb_id >= 0 else RADIO.min_rsrp
            rsrq_neighbor = (compute_rsrq(rsrp_neighbor, rsrp_all)
                             if best_nb_id >= 0 else -20.0)
            l3_srv = ue.l3_rsrp[ue.serving_cell]
            l3_nb  = ue.l3_rsrp[best_nb_id] if best_nb_id >= 0 else RADIO.min_rsrp

            pre_cell = ue.serving_cell
            ho_event, target_id, event_type, ho_failure, ping_pong = \
                evaluate_handover(ue, bss, ue.l3_rsrp, sinr, t, rng)
            if ho_event and not ho_failure:
                ue.serving_cell = target_id

            records.append({
                "timestamp":             t,
                "ue_id":                 ue.ue_id,
                "serving_cell_id":       pre_cell,
                "rsrp_serving":          round(rsrp_serving, 2),
                "rsrq_serving":          round(rsrq, 2),
                "sinr":                  round(sinr, 2),
                "cqi":                   cqi,
                "best_neighbor_cell_id": best_nb_id,
                "rsrp_neighbor":         round(rsrp_neighbor, 2),
                "rsrq_neighbor":         round(rsrq_neighbor, 2),
                "rsrp_diff":             round(rsrp_serving - rsrp_neighbor, 2),
                "l3_rsrp_serving":       round(l3_srv, 2),
                "l3_rsrp_neighbor":      round(l3_nb, 2),
                "ue_speed":              round(ue.speed, 2),
                "pos_x":                 round(ue.x, 2),
                "pos_y":                 round(ue.y, 2),
                "los_flag":              int(ue.los.get(pre_cell, False)),
                "cell_load_pct":         round(cell_load_frac[pre_cell] * 100.0, 1),
                "handover_event":        int(ho_event),
                "target_cell_id":        target_id if ho_event else -1,
                "event_type":            event_type,
                "handover_failure":      int(ho_failure),
                "ping_pong":             int(ping_pong),
                "rlf_flag":              int(ue.rlf_remaining > 0),
            })

    return pd.DataFrame(records)


def add_label(df: pd.DataFrame, k: int = LABEL_K) -> pd.DataFrame:
    """handover_soon = 1 if a successful HO occurs within the next k steps."""
    df = df.sort_values(["ue_id", "timestamp"]).reset_index(drop=True)
    df["_suc"] = ((df["handover_event"] == 1) & (df["handover_failure"] == 0)).astype(int)
    label = pd.Series(0, index=df.index)
    for s in range(1, k + 1):
        label |= df.groupby("ue_id")["_suc"].shift(-s).fillna(0).astype(int)
    df["handover_soon"] = label.astype(int)
    df.drop(columns=["_suc"], inplace=True)
    return df


# ── Summary printer ────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, path: Path, scenario: str) -> None:
    ho = df[df["handover_event"] == 1]
    hs = df["handover_soon"].sum()

    print(f"\n{'─'*55}")
    print(f"  Scenario  : {scenario}")
    print(f"  Rows      : {len(df):,}  ({df['ue_id'].nunique()} UEs × "
          f"{df['timestamp'].nunique()} steps)")
    print(f"  HO events : {len(ho)}", end="")

    if len(ho):
        n_suc = (ho["handover_failure"] == 0).sum()
        n_pp  = ho["ping_pong"].sum()
        print(f"  (success={n_suc}, ping-pong={n_pp})")
        ev_counts = ho["event_type"].value_counts().to_dict()
        print(f"  Event types: " + ", ".join(f"{k}={v}" for k, v in ev_counts.items()))
    else:
        print()

    print(f"  handover_soon=1 : {hs:,}  ({100*hs/len(df):.1f}%)")
    print(f"  LOS fraction    : {df['los_flag'].mean()*100:.1f}%")
    print(f"\n  Signal ranges:")
    for col in ["rsrp_serving", "rsrp_neighbor", "rsrp_diff", "sinr", "l3_rsrp_serving"]:
        print(f"    {col:<22} [{df[col].min():7.1f}, {df[col].max():6.1f}]")

    print(f"\n  ✅ Saved → {path}")
    print(f"{'─'*55}\n")
    print("  Upload this file in the dashboard 🔮 Live Prediction tab")
    print("  to score predictions and compare against ground truth.\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a prediction-ready LTE measurement dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios:
  default    Mixed pedestrians + vehicles, random start positions  (~9% HO rate)
  vehicle    All vehicles at high speed                            (~15-20% HO rate)
  stable     Slow pedestrians near BS centres                      (~2-4% HO rate)
  cell_edge  UEs seeded near cell boundaries                       (~12-18% HO rate)

Examples:
  python generate_test_dataset.py
  python generate_test_dataset.py --scenario vehicle --ues 4 --steps 500
  python generate_test_dataset.py --scenario cell_edge --steps 400 --seed 99
  python generate_test_dataset.py --scenario stable --ues 2 --output quiet_ues.csv
        """,
    )
    p.add_argument("--scenario", choices=["default", "vehicle", "stable", "cell_edge"],
                   default="default", help="Mobility and placement scenario (default: default)")
    p.add_argument("--ues",   type=int, default=5,
                   help="Number of UEs to simulate (default: 5)")
    p.add_argument("--steps", type=int, default=300,
                   help="Number of simulation steps / seconds (default: 300)")
    p.add_argument("--seed",  type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path (default: data/test_scenarios/<scenario>_<seed>.csv)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve output path
    if args.output:
        out = Path(args.output)
    else:
        out_dir = Path("data/test_scenarios")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{args.scenario}_ues{args.ues}_steps{args.steps}_seed{args.seed}.csv"

    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*55}")
    print(f"  LTE Test Dataset Generator")
    print(f"  Scenario : {args.scenario}")
    print(f"  UEs      : {args.ues}   Steps : {args.steps}   Seed : {args.seed}")
    print(f"{'─'*55}")
    print("  Simulating…", end=" ", flush=True)

    df = run(args.scenario, args.ues, args.steps, args.seed)
    df = add_label(df)

    df.to_csv(out, index=False)
    print("done.")

    print_summary(df, out, args.scenario)


if __name__ == "__main__":
    main()
