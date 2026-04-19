"""
LTE Handover Simulation — Phase 1: Data Generation

Simulation layout:
  - 4 base stations on a 1000x1000m grid
  - 15 UEs (mix of pedestrian and vehicle speeds)
  - 1-second timestep, ~1700 steps → ~25k rows
  - A3-event handover trigger (TTT=3 steps, margin=3 dB)
  - Label: handover_soon=1 if handover in next K=3 steps
"""

import numpy as np
import pandas as pd
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ──────────────────────────────────────────────
# Constants & Configuration
# ──────────────────────────────────────────────

GRID_SIZE = 1000.0          # metres
SIM_STEPS = 1800            # seconds  (~25k rows for 15 UEs)
DT = 1.0                    # step duration (s)
NOISE_STD = 2.0             # dB — shadow fading σ
HO_MARGIN_DB = 3.0          # A3 event offset (dB)
TTT_STEPS = 3               # Time-To-Trigger (steps)
LABEL_K = 3                 # lookahead window for handover_soon
THERMAL_NOISE_DBM = -107.0  # dBm — LTE 10 MHz BW
MIN_RSRP = -140.0           # dBm floor
MAX_RSRP = -30.0            # dBm ceiling (very close to BS)

# 3GPP CQI table: maps SINR (dB) thresholds to CQI 1-15
_CQI_SINR_THRESHOLDS = [
    (-6.7, 1), (-4.7, 2), (-2.3, 3), (0.2, 4), (2.4, 5),
    (4.3, 6),  (5.9, 7),  (8.1, 8), (10.3, 9),(11.7, 10),
    (14.1,11),(16.3,12), (18.7,13),(21.0,14),(22.7,15),
]

def sinr_to_cqi(sinr_db: float) -> int:
    for thresh, cqi in reversed(_CQI_SINR_THRESHOLDS):
        if sinr_db >= thresh:
            return cqi
    return 1


# ──────────────────────────────────────────────
# Radio Model
# ──────────────────────────────────────────────

def path_loss_db(d_m: float) -> float:
    """
    3GPP TR 25.814 Urban Macro path loss (LTE 2 GHz band):
      PL = 128.1 + 37.6 * log10(d_km)
    Gives RSRP ≈ -44 dBm at 100m, -71 dBm at 500m, -82 dBm at 1km.
    """
    d_km = max(d_m, 1.0) / 1000.0
    return 128.1 + 37.6 * math.log10(d_km)

def compute_rsrp(d_m: float, tx_power_dbm: float = 46.0, noise: float = 0.0) -> float:
    rsrp = tx_power_dbm - path_loss_db(d_m) + noise
    return float(np.clip(rsrp, MIN_RSRP, MAX_RSRP))

def compute_sinr(rsrp_serving: float, rsrp_others: List[float]) -> float:
    """
    SINR = S / (I + N)
    Convert dBm → mW, sum interference, return SINR in dB.
    """
    s_mw = 10 ** (rsrp_serving / 10.0)
    i_mw = sum(10 ** (r / 10.0) for r in rsrp_others)
    n_mw = 10 ** (THERMAL_NOISE_DBM / 10.0)
    sinr_db = 10.0 * math.log10(s_mw / (i_mw + n_mw + 1e-12))
    return float(np.clip(sinr_db, -20.0, 30.0))

def compute_rsrq(rsrp_serving: float, rsrp_all_cells: List[float]) -> float:
    """
    RSRQ = N_RB * RSRP / RSSI
    Approximated: RSRQ = RSRP_serving - 10*log10(sum of all RSRP in mW)
    Range: -3 to -19.5 dBm (3GPP)
    """
    total_mw = sum(10 ** (r / 10.0) for r in rsrp_all_cells)
    total_mw = max(total_mw, 1e-12)
    rsrq = rsrp_serving - 10.0 * math.log10(total_mw)
    return float(np.clip(rsrq, -20.0, -3.0))


# ──────────────────────────────────────────────
# Network: Base Stations
# ──────────────────────────────────────────────

@dataclass
class BaseStation:
    bs_id: int
    x: float
    y: float
    tx_power_dbm: float = 46.0

    def distance_to(self, ux: float, uy: float) -> float:
        return math.hypot(self.x - ux, self.y - uy)


def create_base_stations() -> List[BaseStation]:
    """4 BSs at roughly symmetric positions in a 1000x1000m area."""
    positions = [
        (250.0, 250.0),
        (750.0, 250.0),
        (250.0, 750.0),
        (750.0, 750.0),
    ]
    return [BaseStation(bs_id=i, x=x, y=y) for i, (x, y) in enumerate(positions)]


# ──────────────────────────────────────────────
# UE Mobility Model
# ──────────────────────────────────────────────

@dataclass
class UE:
    ue_id: int
    x: float
    y: float
    speed: float            # m/s
    heading: float          # radians
    is_vehicle: bool
    serving_cell: int       # BS id
    ttt_counter: int = 0    # Time-To-Trigger counter
    ttt_candidate: Optional[int] = None  # candidate target cell

    def move(self, rng: np.random.Generator) -> None:
        """Linear motion with random walk heading perturbation."""
        turn_sigma = 0.15 if self.is_vehicle else 0.4
        self.heading += rng.normal(0, turn_sigma)

        self.x += self.speed * DT * math.cos(self.heading)
        self.y += self.speed * DT * math.sin(self.heading)

        # Reflect off boundaries
        if self.x < 0 or self.x > GRID_SIZE:
            self.heading = math.pi - self.heading
            self.x = float(np.clip(self.x, 0, GRID_SIZE))
        if self.y < 0 or self.y > GRID_SIZE:
            self.heading = -self.heading
            self.y = float(np.clip(self.y, 0, GRID_SIZE))


def create_ues(bss: List[BaseStation], rng: np.random.Generator) -> List[UE]:
    """15 UEs: 8 pedestrian (1-2 m/s) + 7 vehicle (10-20 m/s)."""
    ues = []
    for i in range(15):
        is_vehicle = i >= 8
        speed = rng.uniform(10, 20) if is_vehicle else rng.uniform(1, 2)
        x = rng.uniform(50, GRID_SIZE - 50)
        y = rng.uniform(50, GRID_SIZE - 50)
        heading = rng.uniform(0, 2 * math.pi)

        # Assign initial serving cell (nearest BS)
        dists = [bs.distance_to(x, y) for bs in bss]
        serving = int(np.argmin(dists))

        ues.append(UE(
            ue_id=i, x=x, y=y, speed=speed,
            heading=heading, is_vehicle=is_vehicle,
            serving_cell=serving,
        ))
    return ues


# ──────────────────────────────────────────────
# Handover Logic (A3 Event)
# ──────────────────────────────────────────────

def evaluate_handover(
    ue: UE,
    bss: List[BaseStation],
    rsrp_map: dict,
) -> Tuple[bool, int]:
    """
    A3 event: trigger HO if best_neighbor_RSRP > serving_RSRP + margin
    for TTT_STEPS consecutive steps.

    Returns (handover_triggered, target_cell_id).
    """
    serving_rsrp = rsrp_map[ue.serving_cell]
    best_neighbor_id = -1
    best_neighbor_rsrp = -999.0

    for bs in bss:
        if bs.bs_id == ue.serving_cell:
            continue
        r = rsrp_map[bs.bs_id]
        if r > best_neighbor_rsrp:
            best_neighbor_rsrp = r
            best_neighbor_id = bs.bs_id

    condition_met = best_neighbor_rsrp > serving_rsrp + HO_MARGIN_DB

    if condition_met and ue.ttt_candidate == best_neighbor_id:
        ue.ttt_counter += 1
    elif condition_met:
        ue.ttt_candidate = best_neighbor_id
        ue.ttt_counter = 1
    else:
        ue.ttt_counter = 0
        ue.ttt_candidate = None

    if ue.ttt_counter >= TTT_STEPS:
        ue.ttt_counter = 0
        ue.ttt_candidate = None
        return True, best_neighbor_id

    return False, best_neighbor_id


# ──────────────────────────────────────────────
# Simulation Loop
# ──────────────────────────────────────────────

def run_simulation(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bss = create_base_stations()
    ues = create_ues(bss, rng)

    records = []

    for t in range(SIM_STEPS):
        for ue in ues:
            # Move UE
            ue.move(rng)

            # Compute RSRP for every BS with independent noise per sample
            rsrp_map = {}
            for bs in bss:
                d = bs.distance_to(ue.x, ue.y)
                noise = rng.normal(0, NOISE_STD)
                rsrp_map[bs.bs_id] = compute_rsrp(d, bs.tx_power_dbm, noise)

            rsrp_serving = rsrp_map[ue.serving_cell]
            rsrp_all = list(rsrp_map.values())
            rsrp_others = [rsrp_map[bs.bs_id] for bs in bss if bs.bs_id != ue.serving_cell]

            sinr = compute_sinr(rsrp_serving, rsrp_others)
            rsrq = compute_rsrq(rsrp_serving, rsrp_all)
            cqi = sinr_to_cqi(sinr)

            # Capture serving cell BEFORE handover for the record
            pre_ho_serving_cell = ue.serving_cell

            # Handover evaluation
            ho_event, best_neighbor_id = evaluate_handover(ue, bss, rsrp_map)
            target_cell = best_neighbor_id if ho_event else -1

            if ho_event:
                ue.serving_cell = best_neighbor_id

            rsrp_neighbor = rsrp_map.get(best_neighbor_id, MIN_RSRP) if best_neighbor_id != -1 else MIN_RSRP
            rsrq_neighbor = compute_rsrq(rsrp_neighbor, rsrp_all) if best_neighbor_id != -1 else -20.0

            records.append({
                "timestamp":            t,
                "ue_id":                ue.ue_id,
                "serving_cell_id":      pre_ho_serving_cell,
                "rsrp_serving":         round(rsrp_serving, 2),
                "rsrq_serving":         round(rsrq, 2),
                "sinr":                 round(sinr, 2),
                "cqi":                  cqi,
                "best_neighbor_cell_id":best_neighbor_id,
                "rsrp_neighbor":        round(rsrp_neighbor, 2),
                "rsrq_neighbor":        round(rsrq_neighbor, 2),
                "ue_speed":             round(ue.speed, 2),
                "pos_x":                round(ue.x, 2),
                "pos_y":                round(ue.y, 2),
                "handover_event":       int(ho_event),
                "target_cell_id":       target_cell,
            })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────
# Label Generation (no leakage)
# ──────────────────────────────────────────────

def add_handover_soon_label(df: pd.DataFrame, k: int = LABEL_K) -> pd.DataFrame:
    """
    handover_soon = 1 if any handover_event=1 occurs in the NEXT k rows
    for the same UE (future rows only — no leakage).

    Strategy: for each UE, shift handover_event backwards by 1..k steps,
    OR them together.
    """
    df = df.sort_values(["ue_id", "timestamp"]).reset_index(drop=True)

    label = pd.Series(0, index=df.index)

    for shift in range(1, k + 1):
        label |= (
            df.groupby("ue_id")["handover_event"]
              .shift(-shift)
              .fillna(0)
              .astype(int)
        )

    df["handover_soon"] = label.astype(int)
    return df


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("Running LTE handover simulation...")
    df = run_simulation(seed=42)

    print(f"Raw rows: {len(df)}")
    print(f"Handover events: {df['handover_event'].sum()}")

    df = add_handover_soon_label(df, k=LABEL_K)

    print(f"Rows with handover_soon=1: {df['handover_soon'].sum()} "
          f"({100*df['handover_soon'].mean():.1f}%)")

    out_path = "lte_handover_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows → {out_path}")

    # Quick sanity check
    print("\n--- Value ranges ---")
    for col in ["rsrp_serving", "rsrq_serving", "sinr", "cqi",
                "rsrp_neighbor", "ue_speed"]:
        print(f"  {col:25s}: [{df[col].min():.1f}, {df[col].max():.1f}]")

    print("\n--- Class balance ---")
    print(df["handover_soon"].value_counts(normalize=True).rename("ratio").to_string())


if __name__ == "__main__":
    main()
