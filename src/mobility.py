"""
UE mobility model — LTE handover simulation (v3)

Model: Random Waypoint (RWP) with direction persistence and optional pauses.

Behaviour
---------
  Pedestrian : slow (0.5–2 m/s), short waypoints (~200 m radius), random turns,
               pause 0–10 s on arrival
  Vehicle    : fast (8–20 m/s), long waypoints (full grid), smooth direction
               blending, no pauses

Direction persistence
  On each step the UE blends its current heading toward the waypoint with a
  small heading noise term.  Vehicles blend 80 % toward target; pedestrians 50 %.
  This produces realistic curvilinear trajectories rather than straight lines.

Boundary handling
  Elastic reflection: UE bounces off the grid edge and picks a new waypoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class MobilityConfig:
    grid_size: float = 1000.0          # metres

    # Speed ranges (m/s)
    ped_speed_range: Tuple[float, float] = (0.5, 2.0)
    veh_speed_range: Tuple[float, float] = (8.0, 20.0)

    # Pauses at waypoint
    ped_pause_max_s: int = 10           # pedestrians pause up to 10 s
    veh_pause_max_s: int = 0            # vehicles never pause

    # Waypoint radius: max distance from current position when picking next WP
    ped_wp_radius: float = 200.0        # shorter excursions for pedestrians
    veh_wp_radius: float = 1000.0       # vehicles can target anywhere

    # Heading noise (radians per step)
    ped_heading_sigma: float = 0.30     # more random
    veh_heading_sigma: float = 0.06     # smooth

    # Direction blending: fraction of target heading applied each step
    ped_heading_blend: float = 0.50
    veh_heading_blend: float = 0.80

    dt: float = 1.0                     # step duration (s)


MOB = MobilityConfig()


# ── UE dataclass ──────────────────────────────────────────────────────────────

@dataclass
class UE:
    ue_id:       int
    x:           float
    y:           float
    speed:       float           # m/s
    heading:     float           # radians
    is_vehicle:  bool
    serving_cell: int

    # Waypoint state
    wp_x:             float = 0.0
    wp_y:             float = 0.0
    pause_remaining:  int   = 0

    # Radio channel state — dicts keyed by bs_id
    shadow: Dict[int, float] = field(default_factory=dict)   # shadow fading (dB)
    ff_i:   Dict[int, float] = field(default_factory=dict)   # fast fading I component
    ff_q:   Dict[int, float] = field(default_factory=dict)   # fast fading Q component
    los:    Dict[int, bool]  = field(default_factory=dict)   # LOS state per BS

    # L3 measurement filter (EMA) — keyed by bs_id
    l3_rsrp: Dict[int, float] = field(default_factory=dict)

    # HO / TTT state
    ttt_counter:   int           = 0
    ttt_candidate: Optional[int] = None

    # HO history for ping-pong detection: list of (from_cell, to_cell, timestamp)
    ho_history: List[Tuple[int, int, int]] = field(default_factory=list)

    # Ping-pong hysteresis: extra HO margin per directed cell pair
    pp_extra_margin:      Dict[Tuple[int, int], float] = field(default_factory=dict)
    pp_margin_remaining:  Dict[Tuple[int, int], int]   = field(default_factory=dict)

    # RLF recovery counter
    rlf_remaining: int = 0

    # Sustained low RSRP counter (for enhanced failure probability)
    steps_at_low_rsrp: int = 0

    def move(self, rng: np.random.Generator,
             cfg: MobilityConfig = MOB) -> float:
        """
        Execute one Random Waypoint step.

        Returns the distance travelled in metres (0 during a pause).
        """
        if self.pause_remaining > 0:
            self.pause_remaining -= 1
            return 0.0

        dx_wp = self.wp_x - self.x
        dy_wp = self.wp_y - self.y
        dist_to_wp = math.hypot(dx_wp, dy_wp)
        step       = self.speed * cfg.dt

        if dist_to_wp <= step:
            # ── Arrive at waypoint ───────────────────────────────────────────
            traveled    = dist_to_wp
            self.x      = self.wp_x
            self.y      = self.wp_y

            # Pedestrian pause at destination
            if not self.is_vehicle and cfg.ped_pause_max_s > 0:
                self.pause_remaining = int(rng.integers(0, cfg.ped_pause_max_s + 1))

            # Pick next waypoint
            self.wp_x, self.wp_y = _sample_waypoint(
                self.x, self.y, self.is_vehicle, rng, cfg
            )
        else:
            # ── Move toward waypoint ─────────────────────────────────────────
            target_heading = math.atan2(dy_wp, dx_wp)
            blend  = cfg.veh_heading_blend if self.is_vehicle else cfg.ped_heading_blend
            sigma  = cfg.veh_heading_sigma if self.is_vehicle else cfg.ped_heading_sigma

            # Blend current heading toward target then add noise
            delta          = _wrap_angle(target_heading - self.heading)
            self.heading  += blend * delta + rng.normal(0.0, sigma)

            self.x += step * math.cos(self.heading)
            self.y += step * math.sin(self.heading)
            traveled = step

        # ── Boundary elastic reflection ──────────────────────────────────────
        if self.x < 0.0:
            self.x       = -self.x
            self.heading = math.pi - self.heading
            self.wp_x, self.wp_y = _sample_waypoint(self.x, self.y, self.is_vehicle, rng, cfg)
        elif self.x > cfg.grid_size:
            self.x       = 2.0 * cfg.grid_size - self.x
            self.heading = math.pi - self.heading
            self.wp_x, self.wp_y = _sample_waypoint(self.x, self.y, self.is_vehicle, rng, cfg)

        if self.y < 0.0:
            self.y       = -self.y
            self.heading = -self.heading
            self.wp_x, self.wp_y = _sample_waypoint(self.x, self.y, self.is_vehicle, rng, cfg)
        elif self.y > cfg.grid_size:
            self.y       = 2.0 * cfg.grid_size - self.y
            self.heading = -self.heading
            self.wp_x, self.wp_y = _sample_waypoint(self.x, self.y, self.is_vehicle, rng, cfg)

        self.x = float(np.clip(self.x, 0.0, cfg.grid_size))
        self.y = float(np.clip(self.y, 0.0, cfg.grid_size))

        return traveled


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sample_waypoint(x: float, y: float, is_vehicle: bool,
                     rng: np.random.Generator,
                     cfg: MobilityConfig) -> Tuple[float, float]:
    """Sample a new waypoint for the UE."""
    margin = 30.0
    lo, hi = margin, cfg.grid_size - margin

    if is_vehicle:
        # Vehicles can target anywhere on the grid
        wp_x = float(rng.uniform(lo, hi))
        wp_y = float(rng.uniform(lo, hi))
    else:
        # Pedestrians pick a point within ped_wp_radius
        r     = rng.uniform(20.0, cfg.ped_wp_radius)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        wp_x  = float(np.clip(x + r * math.cos(angle), lo, hi))
        wp_y  = float(np.clip(y + r * math.sin(angle), lo, hi))

    return wp_x, wp_y


def _wrap_angle(angle: float) -> float:
    """Wrap angle to [−π, π]."""
    while angle >  math.pi: angle -= 2.0 * math.pi
    while angle < -math.pi: angle += 2.0 * math.pi
    return angle
