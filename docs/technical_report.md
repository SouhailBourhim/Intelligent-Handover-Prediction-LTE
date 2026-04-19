# Technical Report — Intelligent Handover Prediction in LTE Networks

**Author:** Souhail Bourhim  
**Repository:** [github.com/SouhailBourhim/Intelligent-Handover-Prediction-LTE](https://github.com/SouhailBourhim/Intelligent-Handover-Prediction-LTE)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Project Architecture](#2-project-architecture)
3. [Phase 1 — LTE Network Simulation](#3-phase-1--lte-network-simulation)
4. [Simulation Realism Improvements (v2)](#4-simulation-realism-improvements-v2)
5. [Phase 2 — Feature Engineering](#5-phase-2--feature-engineering)
6. [Phase 3 — Modelling](#6-phase-3--modelling)
7. [Phase 4 — Evaluation](#7-phase-4--evaluation)
8. [Phase 5 — Dashboard](#8-phase-5--dashboard)
9. [Results Summary](#9-results-summary)
10. [Design Decisions & Trade-offs](#10-design-decisions--trade-offs)
11. [Reproduction Guide](#11-reproduction-guide)

---

## 1. Problem Statement

Handover in LTE networks is the process by which a User Equipment (UE) transfers its connection from one base station (eNodeB) to another. Poorly timed handovers cause:

- **Dropped calls** — if the serving cell degrades before HO completes
- **Ping-pong** — rapid back-and-forth between two cells at a boundary
- **Radio Link Failure (RLF)** — HO attempt fails because the serving signal is already too weak

**Goal:** predict `handover_soon = 1` — whether a successful handover will occur within the next K = 3 seconds — using only measurements available at time *t* (no future leakage).

This is a **binary time-series classification** problem on a heavily imbalanced dataset (~97% negative class).

---

## 2. Project Architecture

```
lte_handover_prediction/
│
├── simulate.py              # Phase 1 — synthetic LTE dataset generation
├── run_pipeline.py          # orchestrates phases 2–4
│
├── src/
│   ├── features.py          # Phase 2 — cleaning + feature engineering
│   ├── models.py            # Phase 3 — LR, Random Forest, LSTM
│   └── evaluate.py          # Phase 4 — metrics, plots, report
│
├── app/
│   └── dashboard.py         # Phase 5 — Streamlit dashboard
│
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_dashboard_preview.ipynb
│   ├── 06_simulation_realism.ipynb   ← v2 improvements analysis
│   └── 07_model_impact.ipynb         ← v1 vs v2 model comparison
│
├── data/
│   ├── raw/dataset.csv      # 27,000-row simulated dataset
│   └── processed/           # train/val/test + meta.json
│
├── models/                  # serialised model artefacts (.pkl / .pt)
├── reports/                 # evaluation report + ROC/confusion plots
└── docs/
    └── technical_report.md  ← this document
```

**Data flow:**

```
simulate.py ──► data/raw/dataset.csv
                      │
              src/features.py (Phase 2)
                      │
         data/processed/{train,val,test}.csv
                      │
              src/models.py (Phase 3)
                      │
           models/{lr,rf,lstm}.{pkl,pt}
                      │
             src/evaluate.py (Phase 4)
                      │
           reports/{roc_curves,confusion_matrices}.png
           reports/evaluation.txt
```

---

## 3. Phase 1 — LTE Network Simulation

### 3.1 Network Topology

| Parameter | Value |
|-----------|-------|
| Grid size | 1000 × 1000 m |
| Base stations | 4 (symmetric: (250,250), (750,250), (250,750), (750,750)) |
| TX power | 46 dBm |
| Frequency | 2 GHz (LTE Band 1) |
| Bandwidth | 10 MHz (noise floor −107 dBm) |

### 3.2 UE Mobility

| Type | Count | Speed | Heading dynamics |
|------|-------|-------|-----------------|
| Pedestrian | 8 | 1–2 m/s | High random-walk variance (σ=0.4 rad) |
| Vehicle | 7 | 10–20 m/s | Low random-walk variance (σ=0.15 rad) |

Boundary reflection: heading mirrors off grid edges.

### 3.3 Radio Model

**Path loss** (3GPP TR 25.814 Urban Macro):

```
PL(d) = 128.1 + 37.6 · log₁₀(d_km)   [dB]
```

At reference distances: d=100m → −44 dBm · d=500m → −71 dBm · d=1000m → −82 dBm

**RSRP:**
```
RSRP = P_tx − PL(d) + σ_shadow   [dBm]
Clipped to [−140, −30] dBm
```

**SINR** (inter-cell + load-based intra-cell):
```
SINR = 10·log₁₀( S / (Σ I_inter + I_load + N) )
I_load = 1.5 dB · (n_co_UE − 1)
```

**RSRQ** (3GPP approximation):
```
RSRQ = RSRP_serving − 10·log₁₀(Σ RSRP_all_mW)
Clipped to [−20, −3] dB
```

**CQI** mapped from SINR using 3GPP TS 36.213 lookup table (values 1–15).

### 3.4 Handover Logic (A3 Event + TTT)

The A3 Time-To-Trigger state machine:

```
condition: RSRP_neighbor > RSRP_serving + 3 dB

if condition met AND same candidate for TTT=3 consecutive steps:
    → trigger handover attempt
    → classify event type (A3/A4/A5)
    → apply failure probability
    → check ping-pong

if condition breaks before TTT expires:
    → reset counter
```

### 3.5 Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | int | Simulation step (seconds) |
| `ue_id` | int | UE identifier (0–14) |
| `serving_cell_id` | int | Current serving BS (0–3) |
| `rsrp_serving` | float | RSRP from serving BS (dBm) |
| `rsrq_serving` | float | RSRQ from serving BS (dB) |
| `sinr` | float | Signal-to-Interference+Noise Ratio (dB) |
| `cqi` | int | Channel Quality Indicator (1–15) |
| `best_neighbor_cell_id` | int | ID of strongest neighbour BS |
| `rsrp_neighbor` | float | RSRP from best neighbour (dBm) |
| `rsrq_neighbor` | float | RSRQ from best neighbour (dB) |
| `ue_speed` | float | UE speed (m/s) |
| `pos_x`, `pos_y` | float | UE position (metres) |
| `serving_cell_load` | int | UEs currently on serving cell |
| `handover_event` | 0/1 | HO attempt made at this step |
| `target_cell_id` | int | Target BS (−1 if no HO) |
| `event_type` | str | A3 / A4 / A5 / none |
| `handover_failure` | 0/1 | HO attempt failed (RLF) |
| `ping_pong` | 0/1 | HO is a ping-pong |
| `rlf_flag` | 0/1 | UE in RLF recovery this step |
| `handover_soon` | 0/1 | **Label** — successful HO in next 3 steps |

---

## 4. Simulation Realism Improvements (v2)

### 4.1 Correlated Shadow Fading

**Motivation:** in v1, each step added independent Gaussian noise. In reality, shadow fading varies smoothly as a UE moves through the environment — nearby positions are correlated.

**Implementation:** Gudmundson AR(1) model with decorrelation distance d_corr = 100 m:

```python
rho = exp(−step_dist / d_corr)
σ_new = rho · σ_old + sqrt(1 − rho²) · N(0, σ²)
```

**Effect:** pedestrian at 1 m/s has ρ ≈ 0.990 (barely changes per step); vehicle at 15 m/s has ρ ≈ 0.861 (faster decorrelation). RSRP traces are now spatially smooth.

### 4.2 A3 / A4 / A5 Event Classification

Three 3GPP measurement event types are now distinguished at each handover trigger:

| Event | Condition | Share in data | Network meaning |
|-------|-----------|---------------|-----------------|
| **A3** | `RSRP_nb > RSRP_srv + 3 dB` | 79% | Standard best-server reselection |
| **A4** | `RSRP_nb > −55 dBm` | 15% | UE moving close to a new BS |
| **A5** | `RSRP_srv < −68` AND `RSRP_nb > −60` | 6% | Coverage emergency |

Classification priority: A5 > A4 > A3 (most critical condition wins).

### 4.3 Cell Load & SINR Degradation

**Motivation:** cells with more UEs experience intra-cell interference.

**Model:** each additional UE on the serving cell adds 1.5 dB SINR penalty.

```
SINR_effective = SINR_baseline − 1.5 × (n_co_UE − 1)
```

This creates realistic SINR variance correlated with time-varying cell occupancy (1–9 UEs observed per cell).

### 4.4 Handover Failure & RLF

**Motivation:** when the serving RSRP is already weak, the UE may lose radio contact during the HO execution window.

**Model:** sigmoid failure probability centred at −80 dBm:

```
P(fail) = σ(−(RSRP_srv − (−80)) / 5)
```

| RSRP | P(fail) |
|------|---------|
| −60 dBm | 1.8% |
| −70 dBm | 12% |
| −80 dBm | 50% |
| −90 dBm | 88% |

On failure: UE enters **RLF recovery** for 3 steps with SINR = −15 dB.

**Data impact:** 3.7% failure rate observed (10/268 attempts). `handover_soon` label is defined on *successful* HOs only — failed HO attempts are not labelled positive.

### 4.5 Ping-Pong Detection

**Definition:** a handover (A→B) is a ping-pong if the UE had previously handed over from B→A within the last 10 steps.

**Detection:**

```python
for (h_from, h_to, h_t) in reversed(ho_history):
    if current_t − h_t > PING_PONG_WINDOW: break
    if h_from == to_cell and h_to == from_cell: return True
```

0.7% of HO attempts were ping-pongs (2/268). This is a real LTE KPI — operators track it to detect over-tight handover margins.

---

## 5. Phase 2 — Feature Engineering

### 5.1 Pipeline (`src/features.py`)

```
load_raw() → clean() → engineer_features() → split_and_scale()
```

No data leakage: all temporal features (lags, rolling windows, deltas) use `shift(k)` with k ≥ 1, meaning they reference only past values. The StandardScaler is fit on the training set and applied to val/test.

### 5.2 Feature Types

Starting from 16 raw columns, 71 features are produced:

| Type | Formula | Example |
|------|---------|---------|
| Raw signals | — | `rsrp_serving`, `sinr` |
| **Lag** (k=1,2,3) | `signal(t−k)` | `rsrp_serving_lag2` |
| **Rolling mean** (w=3,5) | `mean(signal, t−w..t−1)` | `sinr_roll5_mean` |
| **Rolling std** (w=3,5) | `std(signal, t−w..t−1)` | `rsrp_serving_roll3_std` |
| **Delta** (d=1,3) | `signal(t) − signal(t−d)` | `rsrp_serving_delta3` |
| **Domain feature** | `rsrp_neighbor − rsrp_serving` | `rsrp_diff` |

### 5.3 Signals Featured

`rsrp_serving`, `rsrq_serving`, `sinr`, `cqi`, `rsrp_neighbor`, `rsrq_neighbor`  
Plus: `ue_speed`, `pos_x`, `pos_y`, `serving_cell_load`, `handover_failure` (lag/rolling), `ping_pong` (lag/rolling), `rlf_flag` (lag)

### 5.4 Train / Val / Test Split

Temporal 70/15/15 split on unique timestamps (preserves time order — no shuffling).

| Split | Rows | Positive rate |
|-------|------|--------------|
| Train | 18,855 | 2.8% |
| Val | 4,050 | ~2.8% |
| Test | 4,050 | ~2.6% |

---

## 6. Phase 3 — Modelling

### 6.1 Class Imbalance Strategy

Positive rate ≈ 2.7% → 37× imbalance. Each model uses a different mechanism:

| Model | Imbalance handling |
|-------|--------------------|
| Logistic Regression | `class_weight = {0: 1.0, 1: 37.0}` |
| Random Forest | `class_weight = {0: 1.0, 1: 37.0}` |
| LSTM | `WeightedRandomSampler` + `BCEWithLogitsLoss(pos_weight=37)` |

### 6.2 Logistic Regression

- Solver: `saga` (supports L1/L2, converges on large sparse data)
- Regularisation: C = 0.5 (moderate L2)
- Role: linear baseline; coefficients identify which engineered features drive HO risk linearly

### 6.3 Random Forest

- 300 estimators, max depth 12, min_samples_leaf = 5
- Captures non-linear RSRP/SINR threshold effects and feature interactions
- Feature importances provide network-engineer-interpretable explanations

### 6.4 LSTM

**Architecture:**

```
Input: [batch, seq_len=10, n_features=64]
         │
    LSTM(64 hidden, 2 layers, dropout=0.3)
         │  (last hidden state)
    Linear(64 → 32) → ReLU → Dropout(0.3)
         │
    Linear(32 → 1) → sigmoid
```

**Training details:**

| Hyperparameter | Value |
|---------------|-------|
| Sequence length | 10 steps (= 10 seconds of history) |
| Batch size | 256 |
| Optimiser | Adam (lr=1e-3, weight_decay=1e-5) |
| LR scheduler | ReduceLROnPlateau (mode=max, patience=3) |
| Early stopping | patience=5 on val F1 |
| Max epochs | 30 |

Each training sample is a `(10 × 64)` tensor representing one UE's last 10 steps.

---

## 7. Phase 4 — Evaluation

### 7.1 Metrics

Evaluated on the held-out test set at threshold 0.5:

| Model | Precision | Recall | **F1** | **ROC-AUC** |
|-------|-----------|--------|--------|-------------|
| Logistic Regression | 0.394 | 0.971 | 0.560 | 0.986 |
| **Random Forest** | **0.552** | **0.857** | **0.672** | **0.992** |
| LSTM | 0.458 | 0.912 | 0.610 | 0.987 |

### 7.2 Model Analysis

**Logistic Regression:**  
Highest recall (0.97) — catches almost all true handovers at the cost of many false positives (precision 0.39). Useful when missed handovers are more costly than unnecessary preparations (conservative deployment). Coefficient analysis confirms `rsrp_diff` and `rsrp_serving_delta3` as dominant linear predictors.

**Random Forest:**  
Best F1 (0.672) and AUC (0.992). Balances precision and recall well. Feature importances reveal that `rsrp_diff` (serving−neighbour gap), `sinr_roll5_mean`, and `serving_cell_load`-derived features dominate. The ensemble captures the non-linear boundary: HO risk spikes when RSRP gap > 3 dB AND SINR is declining.

**LSTM:**  
Close to Random Forest (F1 = 0.610). Its advantage is learning the temporal *trajectory* of RSRP degradation directly from sequences, rather than relying on manually crafted delta/rolling features. With more training data or hyperparameter tuning, LSTM would likely outperform the RF.

### 7.3 Threshold Recommendations

| Deployment scenario | Recommended threshold | Effect |
|--------------------|-----------------------|--------|
| Conservative (safety-first) | 0.3 | High recall, more false alarms |
| Balanced (default) | 0.5 | Best F1 |
| Precision-focused | 0.65 | Fewer unnecessary preparations |

---

## 8. Phase 5 — Dashboard

**File:** `app/dashboard.py`  
**Launch:** `streamlit run app/dashboard.py`

| Component | Description |
|-----------|-------------|
| KPI monitor | RSRP, SINR, RSRQ, CQI over time per UE with HO event markers |
| HO timeline | Scatter plot of HO events coloured by target cell |
| Risk heatmap | Per-UE predicted probability across all test timestamps |
| Risk gauge | Average risk for selected UEs and model |
| Model comparison | Table pulled from `reports/evaluation.txt` |
| Mobility map | UE tracks on 1000×1000m grid with BS positions |

---

## 9. Results Summary

| Metric | Value |
|--------|-------|
| Dataset size | 27,000 rows (15 UEs × 1,800 s) |
| Positive label rate | 2.9% |
| Handover attempts | 268 |
| HO failure rate | 3.7% |
| Ping-pong rate | 0.7% of HO attempts |
| Event split | A3: 79% · A4: 15% · A5: 6% |
| Best model | Random Forest |
| Best F1 | **0.672** |
| Best ROC-AUC | **0.992** |
| Engineered features | 71 (from 16 raw) |

---

## 10. Design Decisions & Trade-offs

### Why simulate rather than use real data?

Real LTE measurement logs (MDT data) are proprietary and require network operator access. Simulation allows full ground-truth labelling (serving cell, HO events) and controlled comparison of realism levels.

### Why K=3 for the label?

3 seconds is sufficient lead time for proactive resource reservation (typical LTE HO preparation takes 1–2 s). Larger K (e.g., K=10) would increase positive rate but make the prediction task harder and less actionable.

### Why Random Forest over XGBoost?

XGBoost 3.2.0 has a known segmentation fault on Python 3.14 (as of April 2026). Random Forest from scikit-learn is stable and achieves comparable performance on this dataset size.

### Why not tune LSTM further?

The current LSTM already matches the RF without feature engineering — further tuning (attention, Transformer, TFT) is deferred to future work where it would constitute a research contribution.

### Handover failure in the label

`handover_soon` counts only *successful* HOs. This is intentional — the network wants to predict handovers that will actually change the serving cell, not failed attempts that leave the UE on a degraded link.

---

## 11. Reproduction Guide

```bash
# 1. Clone repository
git clone https://github.com/SouhailBourhim/Intelligent-Handover-Prediction-LTE.git
cd Intelligent-Handover-Prediction-LTE/lte_handover_prediction

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Generate dataset (Phase 1)
python simulate.py

# 4. Run full ML pipeline (Phases 2–4)
python run_pipeline.py

# 5. Run individual phases
python run_pipeline.py --phase 2   # feature engineering only
python run_pipeline.py --phase 3   # training only
python run_pipeline.py --phase 4   # evaluation only

# 6. Launch dashboard (Phase 5)
streamlit run app/dashboard.py

# 7. Open notebooks
jupyter notebook notebooks/
# Select kernel: "LTE (venv)"
```

### Expected outputs

| File | Description |
|------|-------------|
| `data/raw/dataset.csv` | 27,000-row raw dataset |
| `data/processed/train.csv` | 18,855 rows, 71 features |
| `data/processed/meta.json` | Feature list + split sizes |
| `models/random_forest.pkl` | Best performing model |
| `models/lstm.pt` | LSTM state dict |
| `models/scaler.pkl` | StandardScaler (fit on train) |
| `reports/evaluation.txt` | Full metrics report |
| `reports/roc_curves.png` | ROC curves for 3 models |
| `reports/confusion_matrices.png` | Confusion matrices |
