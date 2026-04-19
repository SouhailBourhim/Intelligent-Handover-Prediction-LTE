# Intelligent Handover Prediction in LTE Networks using ML

A full machine-learning pipeline that predicts **imminent handover events** (`handover_soon`) in LTE networks from simulated UE radio measurements.

---

## Project Overview

| Item | Detail |
|------|--------|
| Task | Binary time-series classification |
| Label | `handover_soon = 1` if a successful HO occurs within the next 3 steps |
| Dataset | 27,000 rows · 15 UEs · 1,800 s simulation |
| Simulator | v3 — 3GPP UMa LOS/NLOS, Random Waypoint mobility, A3/A4/A5 events |
| Models | Logistic Regression · Random Forest · LSTM |
| Best model | Random Forest (F1 = 0.521, ROC-AUC = 0.909) |

---

## Repository Structure

```
├── simulate.py              # Phase 1 — LTE network simulation (v3)
├── run_pipeline.py          # Master runner (phases 2–4)
├── requirements.txt
│
├── src/
│   ├── radio_model.py       # 3GPP UMa path loss, shadow/fast fading, SINR, CQI
│   ├── mobility.py          # Random Waypoint UE mobility (pedestrian + vehicle)
│   ├── handover_logic.py    # A3/A4/A5 events, L3 filter, TTT, HO failure, ping-pong
│   ├── features.py          # Phase 2 — feature engineering + train/val/test split
│   ├── models.py            # Phase 3 — Logistic Regression, Random Forest, LSTM
│   └── evaluate.py          # Phase 4 — metrics, ROC curves, confusion matrices
│
├── data/
│   ├── raw/dataset.csv      # 27k-row simulated dataset
│   └── processed/           # train/val/test splits + meta.json
│
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_dashboard_preview.ipynb
│   ├── 06_simulation_realism.ipynb   ← v3 radio model analysis
│   └── 07_model_impact.ipynb         ← v2 vs v3 model comparison
│
├── models/                  # Saved model artifacts (.pkl / .pt)
├── reports/                 # Evaluation report + plots
├── docs/
│   └── technical_report.md  # Full write-up
└── app/
    └── dashboard.py         # Phase 5 — Streamlit dashboard
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/SouhailBourhim/Intelligent-Handover-Prediction-LTE.git
cd Intelligent-Handover-Prediction-LTE

# 2. Create environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Generate dataset
python simulate.py

# 4. Run full pipeline (feature engineering → training → evaluation)
python run_pipeline.py

# 5. Launch dashboard
streamlit run app/dashboard.py
```

Run individual phases:

```bash
python run_pipeline.py --phase 2   # feature engineering only
python run_pipeline.py --phase 3   # training only
python run_pipeline.py --phase 4   # evaluation only
```

---

## Phases

### Phase 1 — Data Generation (`simulate.py`)

Simulates a 1000×1000 m LTE network using three modules in `src/`:

**Network topology:**
- 4 base stations at symmetric grid positions — (250,250), (750,250), (250,750), (750,750) m
- 15 UEs: 8 pedestrians (0.5–2 m/s) + 7 vehicles (8–20 m/s)
- Carrier: 2 GHz, TX power: 46 dBm, bandwidth: 10 MHz (noise floor −107 dBm)

**Radio model** (`src/radio_model.py`) — 3GPP TR 36.873 UMa:
- **LOS/NLOS** state per link, sampled from distance-dependent probability `P(LOS|d) = min(18/d,1)·(1−e^{−d/63}) + e^{−d/63}`
- **Path loss** — separate LOS (`22·log10(d)+28+20·log10(fc)`) and NLOS formulas
- **Shadow fading** — Gudmundson AR(1), σ_LOS=4 dB / σ_NLOS=6 dB, decorr=100 m
- **Fast fading** — AR(1) complex Gaussian (Rayleigh NLOS / Rician K=5 dB LOS), 1 s averaged
- **SINR** — load-weighted: `S / (Σ P_nb × (0.10 + 0.90 × cell_load) + N)`
- **L3 filter** — EMA α=0.5 applied to raw RSRP before HO logic (models 3GPP measurement averaging)

**Mobility model** (`src/mobility.py`) — Random Waypoint with direction persistence:
- Pedestrians: short waypoints (200 m radius), high heading noise, pauses on arrival
- Vehicles: full-grid waypoints, smooth heading blend (80% toward target), no pauses
- Elastic boundary reflection

**Handover logic** (`src/handover_logic.py`):
- **A3/A4/A5 event classification** on L3-filtered RSRP
- **Velocity-aware TTT** — `TTT = max(1, 3 − floor(speed/7))` steps
- **Multi-factor HO failure** — weighted combination of SINR sigmoid (50%), speed (15%), target RSRP (15%), sustained low RSRP (20%); ~20% overall failure rate
- **Ping-pong hysteresis** — +3 dB extra margin on the reverse cell pair for 20 steps after a ping-pong

**Dataset columns:**

| Column | Description |
|--------|-------------|
| `timestamp` | Simulation step (s) |
| `ue_id` | UE identifier (0–14) |
| `serving_cell_id` | Current serving BS (0–3) |
| `rsrp_serving` | Instantaneous RSRP from serving BS (dBm) |
| `rsrq_serving` | RSRQ from serving BS (dB) |
| `sinr` | SINR computed from L3-filtered RSRP (dB) |
| `cqi` | Channel Quality Indicator (1–15) |
| `best_neighbor_cell_id` | Strongest neighbour BS |
| `rsrp_neighbor` | Instantaneous RSRP from best neighbour (dBm) |
| `rsrq_neighbor` | RSRQ from best neighbour (dB) |
| `rsrp_diff` | `rsrp_serving − rsrp_neighbor` (dB) |
| `l3_rsrp_serving` | L3-filtered serving RSRP — what HO logic sees (dBm) |
| `l3_rsrp_neighbor` | L3-filtered neighbour RSRP (dBm) |
| `ue_speed` | UE speed (m/s) |
| `pos_x`, `pos_y` | UE position (m) |
| `los_flag` | 1 if serving link is currently LOS |
| `serving_cell_load` | UEs on serving cell |
| `cell_load_pct` | Serving cell load as percentage (0–100) |
| `handover_event` | 1 if HO attempt made this step |
| `target_cell_id` | Target BS (−1 if no HO) |
| `event_type` | A3 / A4 / A5 / none |
| `handover_failure` | 1 if HO attempt failed (→ RLF) |
| `ping_pong` | 1 if this HO reversed a recent one |
| `rlf_flag` | 1 if UE is in RLF recovery this step |
| `handover_soon` | **Label** — successful HO within next 3 steps |

### Phase 2 — Feature Engineering (`src/features.py`)

From 26 raw columns → **86 engineered features**:

| Type | Examples |
|------|---------|
| Lag (t−1, t−2, t−3) | `rsrp_serving_lag1`, `sinr_lag3` |
| Rolling mean (3 s, 5 s) | `rsrp_serving_roll5_mean` |
| Rolling std | `sinr_roll3_std` |
| Delta (1 s, 3 s) | `rsrp_serving_delta3` |
| Domain | `rsrp_diff` = neighbour − serving |

Temporal 70/15/15 split · StandardScaler fit on train only · no future leakage.

### Phase 3 — Modeling (`src/models.py`)

| Model | Key settings |
|-------|-------------|
| Logistic Regression | C=0.5, saga solver, class_weight balanced |
| Random Forest | 300 trees, max_depth=12, class_weight balanced |
| LSTM | 2 layers × 64 hidden, seq_len=10, WeightedRandomSampler |

### Phase 4 — Evaluation (`src/evaluate.py`)

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|----|---------|
| Logistic Regression | 0.315 | 0.852 | 0.460 | 0.894 |
| **Random Forest** | **0.408** | **0.720** | **0.521** | **0.909** |
| LSTM | 0.320 | 0.827 | 0.462 | 0.876 |

**Random Forest** wins on F1 and AUC. **Logistic Regression** maximises recall at the cost of false positives. **LSTM** captures temporal RSRP trajectories without manual feature engineering.

### Phase 5 — Dashboard (`app/dashboard.py`)

Interactive Streamlit app with:
- Real-time KPI charts (RSRP, SINR, RSRQ, CQI) with handover markers
- Handover event timeline
- Per-UE risk heatmap + gauge
- UE mobility map with BS positions
- Model comparison table

---

## Results

![ROC Curves](reports/roc_curves.png)
![Confusion Matrices](reports/confusion_matrices.png)

---

## Requirements

- Python 3.10+
- pandas · numpy · scikit-learn · torch · streamlit · plotly · seaborn · joblib · statsmodels

See [`requirements.txt`](requirements.txt) for pinned versions.

---

## Author

**Souhail Bourhim** — LTE network simulation + ML pipeline
