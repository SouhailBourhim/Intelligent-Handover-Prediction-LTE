"""
LTE Handover Simulation — Phase 1: Data Generation  (v2 — enhanced realism)

Improvements over v1:
  1. Correlated shadow fading  — Gudmundson AR(1) model; spatially smooth
  2. A3 + A4 + A5 event types  — each event classified at trigger time
  3. Cell load tracking        — co-channel UEs degrade SINR realistically
  4. Handover failure          — sigmoid probability based on serving RSRP
  5. Ping-pong detection       — HO back to previous cell within TTT window

New output columns (backward-compatible — all v1 columns retained):
  serving_cell_load  | event_type | handover_failure | ping_pong | rlf_flag
"""

import numpy as np
import pandas as pd
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Constants & Configuration
# ──────────────────────────────────────────────────────────────────────────────

GRID_SIZE   = 1000.0      # metres
SIM_STEPS   = 1800        # seconds
DT          = 1.0         # step duration (s)
NOISE_STD   = 2.0         # dB — shadow fading standard deviation
MIN_RSRP    = -140.0      # dBm floor
MAX_RSRP    = -30.0       # dBm ceiling (very close to BS)
THERMAL_NOISE_DBM = -107.0

LABEL_K     = 3           # lookahead window for handover_soon

# ── A3 (relative threshold) ───────────────────────────────────────────────────
HO_MARGIN_DB  = 3.0       # dB — A3 offset
TTT_STEPS     = 3         # time-to-trigger (steps)

# ── A4 (absolute threshold — neighbor very strong, UE close to target BS) ────
A4_RSRP_THRESHOLD_DBM   = -55.0   # dBm  → ~15% of HOs

# ── A5 (double-threshold — serving weak AND neighbor adequate) ────────────────
A5_SERVING_THRESHOLD_DBM  = -68.0  # dBm — serving must drop below this → ~6% of HOs
A5_NEIGHBOR_THRESHOLD_DBM = -60.0  # dBm — neighbor must exceed this

# ── Handover failure ──────────────────────────────────────────────────────────
# Sigmoid P(fail) centred at HO_FAIL_PIVOT_DBM with scale HO_FAIL_SCALE_DB.
# P(fail | RSRP=-90) ≈ 0.88 · P(fail | RSRP=-80) ≈ 0.50 · P(fail | RSRP=-65) ≈ 0.05
HO_FAIL_PIVOT_DBM = -80.0
HO_FAIL_SCALE_DB  =  5.0
RLF_DURATION_STEPS = 3    # steps UE spends in RLF recovery after failed HO

# ── Ping-pong ─────────────────────────────────────────────────────────────────
PING_PONG_WINDOW = 10     # if HO returns to previous cell within this many steps → ping-pong

# ── Correlated shadow fading (Gudmundson model) ───────────────────────────────
SHADOW_DECORR_DIST_M = 100.0   # metres — spatial decorrelation distance


# ── 3GPP CQI lookup (TS 36.213) ──────────────────────────────────────────────
_CQI_SINR = [
    (-6.7,1),(-4.7,2),(-2.3,3),(0.2,4),(2.4,5),
    (4.3,6),(5.9,7),(8.1,8),(10.3,9),(11.7,10),
    (14.1,11),(16.3,12),(18.7,13),(21.0,14),(22.7,15),
]

def sinr_to_cqi(sinr_db: float) -> int:
    for thresh, cqi in reversed(_CQI_SINR):
        if sinr_db >= thresh:
            return cqi
    return 1


# ──────────────────────────────────────────────────────────────────────────────
# Radio Model
# ──────────────────────────────────────────────────────────────────────────────

def path_loss_db(d_m: float) -> float:
    """3GPP TR 25.814 Urban Macro: PL = 128.1 + 37.6·log10(d_km)"""
    d_km = max(d_m, 1.0) / 1000.0
    return 128.1 + 37.6 * math.log10(d_km)


def compute_rsrp(d_m: float, tx_power_dbm: float = 46.0,
                 shadow_db: float = 0.0) -> float:
    rsrp = tx_power_dbm - path_loss_db(d_m) + shadow_db
    return float(np.clip(rsrp, MIN_RSRP, MAX_RSRP))


def compute_sinr(rsrp_serving: float, rsrp_others: List[float],
                 load_penalty_db: float = 0.0) -> float:
    """
    SINR = S / (I_inter + I_intra + N)

    load_penalty_db accounts for co-channel UE interference on the serving cell.
    Each additional UE on the same cell is modelled as an independent interferer
    whose power is approximated by INTRA_UE_OFFSET below the serving RSRP.
    """
    s_mw  = 10 ** (rsrp_serving / 10.0)
    i_mw  = sum(10 ** (r / 10.0) for r in rsrp_others)
    n_mw  = 10 ** (THERMAL_NOISE_DBM / 10.0)
    sinr  = 10.0 * math.log10(s_mw / (i_mw + n_mw + 1e-12))
    sinr -= load_penalty_db
    return float(np.clip(sinr, -20.0, 30.0))


def compute_rsrq(rsrp_serving: float, rsrp_all: List[float]) -> float:
    """RSRQ ≈ RSRP_serving − 10·log10(Σ RSRP_all_mW)  (3GPP approximation)"""
    total_mw = sum(10 ** (r / 10.0) for r in rsrp_all)
    rsrq = rsrp_serving - 10.0 * math.log10(max(total_mw, 1e-12))
    return float(np.clip(rsrq, -20.0, -3.0))


def ho_failure_probability(rsrp_serving: float) -> float:
    """
    Sigmoid P(HO failure) centred at HO_FAIL_PIVOT_DBM.
    Represents probability that a radio link failure occurs during execution
    because the serving cell is already too weak.
    """
    x = -(rsrp_serving - HO_FAIL_PIVOT_DBM) / HO_FAIL_SCALE_DB
    return 1.0 / (1.0 + math.exp(-x))


# ──────────────────────────────────────────────────────────────────────────────
# Network: Base Stations
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BaseStation:
    bs_id: int
    x: float
    y: float
    tx_power_dbm: float = 46.0

    def distance_to(self, ux: float, uy: float) -> float:
        return math.hypot(self.x - ux, self.y - uy)


def create_base_stations() -> List[BaseStation]:
    positions = [(250.0, 250.0), (750.0, 250.0),
                 (250.0, 750.0), (750.0, 750.0)]
    return [BaseStation(bs_id=i, x=x, y=y) for i, (x, y) in enumerate(positions)]


# ──────────────────────────────────────────────────────────────────────────────
# UE Mobility Model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class UE:
    ue_id:         int
    x:             float
    y:             float
    speed:         float           # m/s
    heading:       float           # radians
    is_vehicle:    bool
    serving_cell:  int

    # A3 TTT state
    ttt_counter:   int            = 0
    ttt_candidate: Optional[int]  = None

    # Improvement 1: per-BS shadow fading state (Gudmundson AR(1))
    shadow_state:  dict           = field(default_factory=dict)   # {bs_id: dB}

    # Improvement 4 & 5: handover history + RLF state
    ho_history:    list           = field(default_factory=list)   # [(from, to, t)]
    rlf_remaining: int            = 0   # steps remaining in RLF recovery

    def move(self, rng: np.random.Generator) -> float:
        """Move one step; return distance travelled (metres)."""
        turn_sigma = 0.15 if self.is_vehicle else 0.4
        self.heading += rng.normal(0, turn_sigma)

        dx = self.speed * DT * math.cos(self.heading)
        dy = self.speed * DT * math.sin(self.heading)
        self.x += dx
        self.y += dy

        if self.x < 0 or self.x > GRID_SIZE:
            self.heading = math.pi - self.heading
            self.x = float(np.clip(self.x, 0, GRID_SIZE))
        if self.y < 0 or self.y > GRID_SIZE:
            self.heading = -self.heading
            self.y = float(np.clip(self.y, 0, GRID_SIZE))

        return math.hypot(dx, dy)


def create_ues(bss: List[BaseStation], rng: np.random.Generator) -> List[UE]:
    """15 UEs: 8 pedestrian (1–2 m/s) + 7 vehicle (10–20 m/s)."""
    ues = []
    for i in range(15):
        is_vehicle = i >= 8
        speed      = rng.uniform(10, 20) if is_vehicle else rng.uniform(1, 2)
        x = rng.uniform(50, GRID_SIZE - 50)
        y = rng.uniform(50, GRID_SIZE - 50)
        heading = rng.uniform(0, 2 * math.pi)
        serving = int(np.argmin([bs.distance_to(x, y) for bs in bss]))

        # Initialise shadow state: small random values per BS
        shadow_state = {bs.bs_id: rng.normal(0, NOISE_STD) for bs in bss}

        ues.append(UE(
            ue_id=i, x=x, y=y, speed=speed, heading=heading,
            is_vehicle=is_vehicle, serving_cell=serving,
            shadow_state=shadow_state,
        ))
    return ues


# ──────────────────────────────────────────────────────────────────────────────
# Improvement 1 — Correlated Shadow Fading (Gudmundson AR(1))
# ──────────────────────────────────────────────────────────────────────────────

def update_shadow_state(ue: UE, step_dist: float,
                        rng: np.random.Generator) -> None:
    """
    Evolve per-BS shadow fading using a first-order autoregressive model.

    rho = exp(-d / d_corr)  — spatial correlation coefficient
    σ_new = rho·σ_old + sqrt(1−rho²)·N(0, σ²)

    When d << d_corr  →  rho≈1  →  shadow barely changes  (smooth areas)
    When d >> d_corr  →  rho≈0  →  shadow fully refreshes (fast decorrelation)
    """
    rho = math.exp(-step_dist / SHADOW_DECORR_DIST_M)
    noise_scale = NOISE_STD * math.sqrt(max(1.0 - rho ** 2, 0.0))
    for bs_id in ue.shadow_state:
        ue.shadow_state[bs_id] = (
            rho * ue.shadow_state[bs_id] + rng.normal(0, noise_scale)
        )


# ──────────────────────────────────────────────────────────────────────────────
# Improvement 2 — A3 / A4 / A5 Event Classification
# ──────────────────────────────────────────────────────────────────────────────

def classify_event_type(rsrp_serving: float,
                        rsrp_neighbor: float) -> str:
    """
    Classify the 3GPP measurement event that best describes this handover.

    A5 — serving is critically weak AND neighbor is adequate (coverage emergency)
    A4 — neighbor signal is very strong (UE moving close to target BS)
    A3 — standard relative-threshold handover (default, most common)

    Priority: A5 > A4 > A3
    Expected distribution: ~79% A3 · ~15% A4 · ~6% A5
    """
    if (rsrp_serving  < A5_SERVING_THRESHOLD_DBM and
            rsrp_neighbor > A5_NEIGHBOR_THRESHOLD_DBM):
        return "A5"
    if rsrp_neighbor > A4_RSRP_THRESHOLD_DBM:
        return "A4"
    return "A3"


# ──────────────────────────────────────────────────────────────────────────────
# Improvement 5 — Ping-Pong Detection
# ──────────────────────────────────────────────────────────────────────────────

def check_ping_pong(ho_history: list, from_cell: int,
                    to_cell: int, current_t: int) -> bool:
    """
    Return True if this handover (from_cell → to_cell) is a ping-pong:
    the UE already handed over FROM to_cell TO from_cell within PING_PONG_WINDOW steps.
    """
    for (h_from, h_to, h_t) in reversed(ho_history):
        if current_t - h_t > PING_PONG_WINDOW:
            break
        if h_from == to_cell and h_to == from_cell:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Handover Logic (A3 TTT — trigger; A3/A4/A5 classification after trigger)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_handover(
    ue: UE,
    bss: List[BaseStation],
    rsrp_map: dict,
    current_t: int,
    rng: np.random.Generator,
) -> Tuple[bool, int, str, bool, bool]:
    """
    Run the A3 TTT state machine and, when it fires, apply:
      • event classification  (A3 / A4 / A5)
      • handover failure      (sigmoid probability)
      • ping-pong detection

    Returns:
        ho_event        — True if handover attempt was made
        target_cell_id  — best neighbour id (-1 if none)
        event_type      — 'A3' | 'A4' | 'A5' | 'none'
        ho_failure      — True if HO was attempted but failed (RLF)
        ping_pong       — True if this HO is a ping-pong
    """
    # UE in RLF — no new HO measurement
    if ue.rlf_remaining > 0:
        return False, -1, "none", False, False

    serving_rsrp   = rsrp_map[ue.serving_cell]
    best_nb_id     = -1
    best_nb_rsrp   = -999.0

    for bs in bss:
        if bs.bs_id == ue.serving_cell:
            continue
        r = rsrp_map[bs.bs_id]
        if r > best_nb_rsrp:
            best_nb_rsrp = r
            best_nb_id   = bs.bs_id

    # A3 entry condition
    a3_condition = best_nb_rsrp > serving_rsrp + HO_MARGIN_DB

    if a3_condition and ue.ttt_candidate == best_nb_id:
        ue.ttt_counter += 1
    elif a3_condition:
        ue.ttt_candidate = best_nb_id
        ue.ttt_counter   = 1
    else:
        ue.ttt_counter   = 0
        ue.ttt_candidate = None

    if ue.ttt_counter < TTT_STEPS:
        return False, best_nb_id, "none", False, False

    # TTT expired — HO attempt
    ue.ttt_counter   = 0
    ue.ttt_candidate = None

    event_type = classify_event_type(serving_rsrp, best_nb_rsrp)

    # Improvement 4: handover failure
    p_fail    = ho_failure_probability(serving_rsrp)
    ho_failed = rng.random() < p_fail

    # Improvement 5: ping-pong detection
    ping_pong = check_ping_pong(ue.ho_history, ue.serving_cell,
                                best_nb_id, current_t)

    # Record this attempt (regardless of failure — for future ping-pong checks)
    ue.ho_history.append((ue.serving_cell, best_nb_id, current_t))
    if len(ue.ho_history) > 20:
        ue.ho_history.pop(0)

    if ho_failed:
        ue.rlf_remaining = RLF_DURATION_STEPS
        return True, best_nb_id, event_type, True, ping_pong

    return True, best_nb_id, event_type, False, ping_pong


# ──────────────────────────────────────────────────────────────────────────────
# Simulation Loop
# ──────────────────────────────────────────────────────────────────────────────

def run_simulation(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bss = create_base_stations()
    ues = create_ues(bss, rng)

    records = []

    for t in range(SIM_STEPS):

        # Improvement 3: cell load snapshot at start of each step
        cell_load: Counter = Counter(ue.serving_cell for ue in ues)

        for ue in ues:
            # ── Mobility ────────────────────────────────────────────────────
            step_dist = ue.move(rng)

            # ── Improvement 1: update correlated shadow fading ───────────────
            update_shadow_state(ue, step_dist, rng)

            # ── Radio measurements ───────────────────────────────────────────
            rsrp_map: dict[int, float] = {}
            for bs in bss:
                d = bs.distance_to(ue.x, ue.y)
                rsrp_map[bs.bs_id] = compute_rsrp(
                    d, bs.tx_power_dbm, ue.shadow_state[bs.bs_id]
                )

            rsrp_serving = rsrp_map[ue.serving_cell]
            rsrp_all     = list(rsrp_map.values())
            rsrp_others  = [rsrp_map[bs.bs_id]
                            for bs in bss if bs.bs_id != ue.serving_cell]

            # Improvement 3: load-based SINR penalty
            # Each co-cell UE contributes ~1.5 dB interference degradation.
            n_co_ue         = max(0, cell_load[ue.serving_cell] - 1)
            load_penalty_db = 1.5 * n_co_ue

            # RLF recovery: SINR is floored at -15 dB during radio link failure
            if ue.rlf_remaining > 0:
                sinr = -15.0
                ue.rlf_remaining -= 1
            else:
                sinr = compute_sinr(rsrp_serving, rsrp_others, load_penalty_db)

            rsrq = compute_rsrq(rsrp_serving, rsrp_all)
            cqi  = sinr_to_cqi(sinr)

            # ── Handover evaluation ──────────────────────────────────────────
            pre_ho_cell = ue.serving_cell

            ho_event, best_nb_id, event_type, ho_failure, ping_pong = \
                evaluate_handover(ue, bss, rsrp_map, t, rng)

            if ho_event and not ho_failure:
                ue.serving_cell = best_nb_id

            # Neighbour metrics
            if best_nb_id != -1:
                rsrp_neighbor = rsrp_map[best_nb_id]
                rsrq_neighbor = compute_rsrq(rsrp_neighbor, rsrp_all)
            else:
                rsrp_neighbor = MIN_RSRP
                rsrq_neighbor = -20.0

            records.append({
                "timestamp":             t,
                "ue_id":                 ue.ue_id,
                "serving_cell_id":       pre_ho_cell,
                # ── core radio KPIs ──────────────────────────────────────────
                "rsrp_serving":          round(rsrp_serving, 2),
                "rsrq_serving":          round(rsrq, 2),
                "sinr":                  round(sinr, 2),
                "cqi":                   cqi,
                "best_neighbor_cell_id": best_nb_id,
                "rsrp_neighbor":         round(rsrp_neighbor, 2),
                "rsrq_neighbor":         round(rsrq_neighbor, 2),
                # ── UE context ───────────────────────────────────────────────
                "ue_speed":              round(ue.speed, 2),
                "pos_x":                 round(ue.x, 2),
                "pos_y":                 round(ue.y, 2),
                # ── Improvement 3: cell load ──────────────────────────────────
                "serving_cell_load":     cell_load[pre_ho_cell],
                # ── Handover outcomes ─────────────────────────────────────────
                "handover_event":        int(ho_event),
                "target_cell_id":        best_nb_id if ho_event else -1,
                # ── New realism columns ───────────────────────────────────────
                "event_type":            event_type,
                "handover_failure":      int(ho_failure),
                "ping_pong":             int(ping_pong),
                "rlf_flag":              int(ue.rlf_remaining > 0),
            })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Label Generation (no leakage)
# ──────────────────────────────────────────────────────────────────────────────

def add_handover_soon_label(df: pd.DataFrame, k: int = LABEL_K) -> pd.DataFrame:
    """
    handover_soon = 1 if a *successful* handover occurs within the next k steps
    for the same UE.  Failed HOs (handover_failure=1) are excluded so the model
    learns to predict events that actually change the serving cell.
    """
    df = df.sort_values(["ue_id", "timestamp"]).reset_index(drop=True)

    # successful_ho: HO event that was not a failure
    df["_successful_ho"] = ((df["handover_event"] == 1) &
                            (df["handover_failure"] == 0)).astype(int)

    label = pd.Series(0, index=df.index)
    for shift in range(1, k + 1):
        label |= (
            df.groupby("ue_id")["_successful_ho"]
              .shift(-shift)
              .fillna(0)
              .astype(int)
        )

    df["handover_soon"] = label.astype(int)
    df.drop(columns=["_successful_ho"], inplace=True)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Running LTE handover simulation (v2 — enhanced realism)...")
    df = run_simulation(seed=42)
    df = add_handover_soon_label(df, k=LABEL_K)

    out_path = "data/raw/dataset.csv"
    df.to_csv(out_path, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    ho = df[df["handover_event"] == 1]
    print(f"\nRows            : {len(df):,}")
    print(f"HO attempts     : {df['handover_event'].sum()}")
    print(f"  ├─ Successful : {(ho['handover_failure']==0).sum()}")
    print(f"  ├─ Failed     : {ho['handover_failure'].sum()}  "
          f"({100*ho['handover_failure'].mean():.1f}% failure rate)")
    print(f"  └─ Ping-pong  : {ho['ping_pong'].sum()}  "
          f"({100*ho['ping_pong'].mean():.1f}% of HO attempts)")

    print(f"\nEvent type breakdown (of HO attempts):")
    for ev, cnt in ho["event_type"].value_counts().items():
        print(f"  {ev}: {cnt}  ({100*cnt/len(ho):.1f}%)")

    print(f"\nhandover_soon=1 : {df['handover_soon'].sum():,}  "
          f"({100*df['handover_soon'].mean():.1f}%)")

    print("\nSignal ranges:")
    for col in ["rsrp_serving", "rsrq_serving", "sinr", "cqi",
                "rsrp_neighbor", "serving_cell_load"]:
        print(f"  {col:25s}: [{df[col].min():.1f}, {df[col].max():.1f}]")

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
