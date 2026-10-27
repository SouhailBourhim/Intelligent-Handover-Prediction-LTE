"""
generate_test_dataset.py — Generate a small, prediction-ready LTE measurement dataset.

Uses the shared simulation engine (src/simulator.py), so the radio model, mobility
and A3/A4/A5 handover logic are exactly those of the training simulator, but with configurable scenario presets so you can quickly
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
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.mobility import MOB, UE  # noqa: E402
from src.simulator import (  # noqa: E402
    LABEL_K, BaseStation, add_handover_soon_label, init_ue, make_simulator,
)

# Cell-boundary midpoints — used to seed UEs near edges in cell_edge scenario
EDGE_SEEDS = [
    (500.0, 250.0),  # between BS0 and BS1
    (250.0, 500.0),  # between BS0 and BS2
    (500.0, 500.0),  # grid centre (between all 4)
    (750.0, 500.0),  # between BS1 and BS3
    (500.0, 750.0),  # between BS2 and BS3
]


# ── UE factory ─────────────────────────────────────────────────────────────────

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
            speed_range = (MOB.ped_speed_range[0], min(MOB.ped_speed_range[1], 1.2))
            is_vehicle  = False

        elif scenario == "vehicle":
            # Fast vehicles, random positions
            x = float(rng.uniform(50, MOB.grid_size - 50))
            y = float(rng.uniform(50, MOB.grid_size - 50))
            speed_range = MOB.veh_speed_range
            is_vehicle  = True

        elif scenario == "cell_edge":
            # UEs seeded near cell boundaries, alternating fast / medium speed
            seed_pos = EDGE_SEEDS[i % len(EDGE_SEEDS)]
            x = float(np.clip(seed_pos[0] + rng.normal(0, 30), 50, MOB.grid_size - 50))
            y = float(np.clip(seed_pos[1] + rng.normal(0, 30), 50, MOB.grid_size - 50))
            if i % 2 == 0:
                speed_range, is_vehicle = MOB.veh_speed_range, True
            else:
                speed_range, is_vehicle = (3.0, 8.0), False   # fast ped / slow veh

        else:   # default — mixed
            x = float(rng.uniform(50, MOB.grid_size - 50))
            y = float(rng.uniform(50, MOB.grid_size - 50))
            is_vehicle  = (i >= n_ues // 2)
            speed_range = MOB.veh_speed_range if is_vehicle else MOB.ped_speed_range

        ues.append(init_ue(i, x, y, speed_range, is_vehicle, bss, rng))

    return ues


# ── Simulation ─────────────────────────────────────────────────────────────────

def run(scenario: str, n_ues: int, n_steps: int, seed: int) -> pd.DataFrame:
    """Simulate one scenario with the shared engine (src/simulator.py)."""
    factory = lambda bss, rng: create_ues_for_scenario(scenario, n_ues, bss, rng)
    return make_simulator(factory, seed).run(n_steps)


def add_label(df: pd.DataFrame, k: int = LABEL_K) -> pd.DataFrame:
    """handover_soon = 1 if a successful HO occurs within the next k steps."""
    return add_handover_soon_label(df, k)


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
