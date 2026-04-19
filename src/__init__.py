"""
src — LTE Handover Prediction: simulation modules + ML pipeline

Simulation (used by simulate.py)
---------------------------------
  radio_model     3GPP UMa LOS/NLOS path loss, Gudmundson shadow fading,
                  AR(1) fast fading (Rayleigh/Rician), load-weighted SINR,
                  RSRQ, CQI mapping.

  mobility        Random Waypoint UE mobility with direction persistence and
                  elastic boundary reflection.  Pedestrian and vehicle profiles.

  handover_logic  L3 EMA measurement filter, velocity-aware TTT, A3/A4/A5 event
                  classification, multi-factor HO failure probability, ping-pong
                  hysteresis.

ML pipeline (used by run_pipeline.py)
--------------------------------------
  features        Phase 2 — data cleaning, lag/rolling/delta feature engineering,
                  temporal train/val/test split, StandardScaler.

  models          Phase 3 — Logistic Regression, Random Forest, LSTM training
                  and serialisation helpers.

  evaluate        Phase 4 — precision/recall/F1/AUC metrics, ROC curve and
                  confusion matrix plots, text evaluation report.
"""

# Simulation public API
from .radio_model import (
    RADIO, RadioConfig,
    p_los,
    path_loss, path_loss_los, path_loss_nlos,
    compute_rsrp, compute_sinr, compute_rsrq, sinr_to_cqi,
    update_shadow, update_fast_fading,
)
from .mobility import MOB, MobilityConfig, UE
from .handover_logic import HO, HOConfig, l3_update, velocity_ttt, evaluate_handover

__all__ = [
    # radio_model
    "RADIO", "RadioConfig",
    "p_los",
    "path_loss", "path_loss_los", "path_loss_nlos",
    "compute_rsrp", "compute_sinr", "compute_rsrq", "sinr_to_cqi",
    "update_shadow", "update_fast_fading",
    # mobility
    "MOB", "MobilityConfig", "UE",
    # handover_logic
    "HO", "HOConfig", "l3_update", "velocity_ttt", "evaluate_handover",
]
