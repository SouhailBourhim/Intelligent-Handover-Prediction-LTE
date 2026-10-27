"""
LTE Handover Simulation — v4 (training dataset)

Runs the shared engine in src/simulator.py on the training configuration
(4 eNBs, 8 pedestrians + 7 vehicles, 3,600 s) and writes data/raw/dataset.csv.

Architecture
------------
  src/radio_model.py    — 3GPP UMa LOS/NLOS path loss, spatially consistent
                          LOS state, shadow fading (AR-1), Rayleigh/Rician fast
                          fading, per-RE RSRP, load-weighted SINR, RSRQ, CQI
  src/mobility.py       — Random Waypoint with direction persistence, pauses,
                          a new speed per waypoint leg
  src/handover_logic.py — L3 filtering, velocity-aware TTT, A3/A4/A5 events,
                          multi-factor HO failure, ping-pong hysteresis
  src/simulator.py      — the step loop, RLF and RRC re-establishment, label

v4 changes vs v3
----------------
  * RSRP is reference-signal power per resource element (≈ 28 dB below the
    v3 total-power value) and is reported in the 3GPP range [−140, −44] dBm.
  * LOS state is spatially consistent: it only changes as the UE moves.
  * ue_speed is the UE's actual speed (0 when paused, new speed per leg).
  * RLF lasts exactly 3 steps and ends with re-establishment on the best cell.
  * New column rrc_reestablish.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.simulator import (  # noqa: E402
    LABEL_K, add_handover_soon_label, create_mixed_ues, make_simulator,
)


# ── Simulation parameters ─────────────────────────────────────────────────────

SIM_STEPS   = 3600     # simulation steps (= seconds at DT = 1 s)
NUM_UES_PED = 8        # pedestrian UEs
NUM_UES_VEH = 7        # vehicle UEs
SEED        = 42


def create_ues(bss, rng):
    """Training UE population: 8 pedestrians (ids 0–7) + 7 vehicles (8–14)."""
    return create_mixed_ues(NUM_UES_PED, NUM_UES_VEH, bss, rng)


def run_simulation(seed: int = SEED, n_steps: int = SIM_STEPS,
                   ttt_policy=None) -> pd.DataFrame:
    """Simulate the training network and return the raw (unlabelled) dataset."""
    return make_simulator(create_ues, seed, ttt_policy).run(n_steps)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("LTE Handover Simulation - v4")
    print("=" * 60)

    df = add_handover_soon_label(run_simulation(seed=SEED), k=LABEL_K)

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

        print("\nEvent type breakdown:")
        for ev, cnt in ho["event_type"].value_counts().items():
            print(f"  {ev}: {cnt:4d}  ({100*cnt/len(ho):.1f}%)")

    hs = df["handover_soon"].sum()
    print(f"\nhandover_soon=1 : {hs:,}  ({100*hs/len(df):.1f}%)")
    print(f"LOS fraction    : {df['los_flag'].mean()*100:.1f}%")

    print("\nSignal ranges:")
    for col in ["rsrp_serving", "rsrq_serving", "sinr", "cqi",
                "rsrp_neighbor", "rsrp_diff", "l3_rsrp_serving", "cell_load_pct"]:
        print(f"  {col:22s}: [{df[col].min():7.1f}, {df[col].max():6.1f}]")

    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
