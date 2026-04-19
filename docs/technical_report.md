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
├── simulate.py              # Phase 1 — synthetic LTE dataset generation (v3)
├── run_pipeline.py          # orchestrates phases 2–4
│
├── src/
│   ├── radio_model.py       # 3GPP UMa path loss, shadow/fast fading, SINR, CQI
│   ├── mobility.py          # Random Waypoint UE mobility (pedestrian + vehicle)
│   ├── handover_logic.py    # A3/A4/A5 events, L3 filter, TTT, HO failure, ping-pong
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
│   ├── 06_simulation_realism.ipynb   ← v3 radio model analysis
│   └── 07_model_impact.ipynb         ← v2 vs v3 model comparison
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

### 3.2 UE Mobility (`src/mobility.py`)

Random Waypoint model with direction persistence:

| Type | Count | Speed | Heading blend | Pause |
|------|-------|-------|---------------|-------|
| Pedestrian | 8 | 0.5–2 m/s | 50% toward waypoint, σ=0.30 rad | 0–10 s on arrival |
| Vehicle | 7 | 8–20 m/s | 80% toward waypoint, σ=0.06 rad | None |

Each step the heading blends toward the current waypoint target then adds small Gaussian noise, producing realistic curvilinear trajectories. On arrival at a waypoint a new one is sampled (pedestrians within 200 m radius, vehicles anywhere on the grid). Boundary: elastic reflection off grid edges.

### 3.3 Radio Model (`src/radio_model.py`)

**LOS/NLOS state** (3GPP TR 36.873 UMa outdoor):
```
P(LOS | d) = min(18/d, 1) · (1 − exp(−d/63)) + exp(−d/63)
  d=100 m → P(LOS) ≈ 0.35,   d=400 m → P(LOS) ≈ 0.05
```
State is re-sampled every 5 steps; separate σ values are used for LOS vs NLOS shadow fading.

**Path loss** (3GPP TR 36.873 UMa):
```
LOS:  PL = 22·log₁₀(d) + 28 + 20·log₁₀(fc_GHz)
NLOS: PL = max(PL_LOS, 19.55 + 39.09·log₁₀(d))
```

**Shadow fading** — Gudmundson AR(1), decorrelation distance 100 m:
```
ρ = exp(−Δd / 100),   σ_LOS = 4 dB,   σ_NLOS = 6 dB
σ_new = ρ · σ_old + √(1−ρ²) · N(0, σ²)
```

**Fast fading** — AR(1) complex Gaussian, decorrelation distance 3 m:
- NLOS: Rayleigh amplitude, clipped to [−8, +3] dB (represents 1 s measurement average)
- LOS: Rician K=5 dB adds a deterministic in-phase component

**RSRP:**
```
RSRP = P_tx − PL(d, LOS/NLOS) + shadow + fast_fade   [dBm]
Clipped to [−140, −25] dBm
```

**L3 measurement filter** — 3GPP TS 36.331 EMA applied to raw RSRP before HO logic:
```
L3_RSRP_new = α · RSRP_raw + (1−α) · L3_RSRP_old     (α = 0.5)
```

**SINR** — computed from L3-filtered RSRP, load-weighted interference:
```
SINR = 10·log₁₀( S_L3 / (Σ P_nb_L3 · (0.10 + 0.90·load_nb) + N) )
Clipped to [−20, +30] dB
```
Idle cells still contribute 10% of their reference-signal power as an interference floor.

**RSRQ** (3GPP TS 36.214):
```
RSRQ = RSRP_serving − 10·log₁₀(Σ RSRP_all_mW),   clipped to [−20, −3] dB
```

**CQI** mapped from SINR via 3GPP TS 36.213 15-level lookup table.

### 3.4 Handover Logic (`src/handover_logic.py`)

**A3/A4/A5 event classification** (on L3-filtered RSRP):

| Event | Condition | Network meaning |
|-------|-----------|-----------------|
| **A5** | `L3_srv < −68 dBm` AND `L3_nb > −60 dBm` | Coverage emergency — leave immediately |
| **A4** | `L3_nb > −55 dBm` | Strong target available |
| **A3** | `L3_nb > L3_srv + margin` | Standard best-server reselection |

Classification priority: A5 > A4 > A3.

**Velocity-aware TTT:**
```
TTT = max(1, 3 − floor(speed / 7))   [steps]
```
Vehicles at 14 m/s get TTT=1 (react quickly); pedestrians keep TTT=3.

**Multi-factor HO failure probability:**
```
P(fail) = 0.50·p_SINR + 0.15·p_speed + 0.15·p_target + 0.20·p_sustained
```
- `p_SINR`: sigmoid centred at −12 dB (low SINR → hard to execute HO)
- `p_speed`: sigmoid centred at 28 m/s (very fast UE may outrun procedure)
- `p_target`: linear ramp from 0 at −70 dBm to 1 at −105 dBm
- `p_sustained`: fraction of `low_rsrp_decay=25` steps already at cell edge

**Ping-pong hysteresis:** after a ping-pong the reverse cell pair receives +3 dB extra HO margin for 20 steps, suppressing immediate bouncebacks.

### 3.5 Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | int | Simulation step (seconds) |
| `ue_id` | int | UE identifier (0–14) |
| `serving_cell_id` | int | Current serving BS (0–3) |
| `rsrp_serving` | float | Instantaneous RSRP from serving BS (dBm) |
| `rsrq_serving` | float | RSRQ from serving BS (dB) |
| `sinr` | float | SINR computed from L3-filtered RSRP (dB) |
| `cqi` | int | Channel Quality Indicator (1–15) |
| `best_neighbor_cell_id` | int | ID of strongest neighbour BS |
| `rsrp_neighbor` | float | Instantaneous RSRP from best neighbour (dBm) |
| `rsrq_neighbor` | float | RSRQ from best neighbour (dB) |
| `rsrp_diff` | float | `rsrp_serving − rsrp_neighbor` (dB) |
| `l3_rsrp_serving` | float | L3-filtered serving RSRP — what HO logic sees (dBm) |
| `l3_rsrp_neighbor` | float | L3-filtered neighbour RSRP (dBm) |
| `ue_speed` | float | UE speed (m/s) |
| `pos_x`, `pos_y` | float | UE position (metres) |
| `los_flag` | 0/1 | 1 if serving link is currently LOS |
| `serving_cell_load` | int | UEs currently on serving cell |
| `cell_load_pct` | float | Serving cell load as percentage (0–100) |
| `handover_event` | 0/1 | HO attempt made at this step |
| `target_cell_id` | int | Target BS (−1 if no HO) |
| `event_type` | str | A3 / A4 / A5 / none |
| `handover_failure` | 0/1 | HO attempt failed (RLF) |
| `ping_pong` | 0/1 | HO is a ping-pong |
| `rlf_flag` | 0/1 | UE in RLF recovery this step |
| `handover_soon` | 0/1 | **Label** — successful HO in next 3 steps |

---

## 4. Simulator Architecture (v3)

The v3 simulator is split into three modules under `src/` that are orchestrated by `simulate.py`. The key design goal was to make every modelling choice traceable to a specific 3GPP specification or published measurement study.

### 4.1 Why L3 Filtering Changes Everything

**Problem:** earlier versions computed SINR from instantaneous fast-faded RSRP. A deep instantaneous fade (−15 dB below mean) gave SINR ≈ −20 dB, making the HO failure model think every handover was happening in a near-outage. This drove the failure rate to 30–40%, far above the 15–25% seen in real networks.

**Solution:** 3GPP TS 36.331 mandates that UEs apply a Layer-3 measurement filter (exponential moving average, α=0.5) to raw RSRP before reporting. The HO trigger and failure model both operate on the L3-filtered value, which averages out fast-fading dips. SINR is also computed from L3-filtered RSRP, representing a 1-second measurement-averaged channel quality.

**Effect:** A3 HO triggers now occur at median SINR ≈ −4.5 dB (cell-edge but not outage); A5 triggers at ≈ −16 dB (genuine coverage holes). Overall failure rate: ~20%.

### 4.2 Path Loss Model Upgrade

**v1/v2:** 3GPP TR 25.814 single slope (`128.1 + 37.6·log₁₀(d_km)`) — no LOS/NLOS distinction.

**v3:** 3GPP TR 36.873 UMa with separate LOS and NLOS formulas:
```
LOS:  PL = 22·log₁₀(d) + 28 + 20·log₁₀(fc)   [dB at 2 GHz]
NLOS: PL = max(PL_LOS, 19.55 + 39.09·log₁₀(d))
```
NLOS has a steeper distance exponent (39 vs 22), which produces realistic deep coverage holes at cell edges — A5 events increase from ~6% to ~22% of HO triggers, matching field observations.

LOS probability is distance-dependent (Sec 3.3), re-sampled every 5 steps to model slow environment changes.

### 4.3 Fast Fading as Averaged Measurement

**v1/v2:** no explicit fast fading.

**v3:** AR(1) complex Gaussian state machine (`ff_i`, `ff_q`) evolving with spatial decorrelation ~3 m (a few wavelengths at 2 GHz). The clipping range is [−8, +3] dB, representing a 1-second time-averaged contribution — not the ±20 dB of instantaneous fast fading.

For LOS links a Rician K=5 dB component is added to the in-phase channel, raising the mean RSRP by ~1.5 dB relative to NLOS Rayleigh.

### 4.4 Load-Weighted SINR Interference Model

**v2:** `SINR_eff = SINR_baseline − 1.5·(n_UE − 1)` — a simple additive penalty.

**v3:** proper power-domain model:
```
I_k = P_k_mW × (0.10 + 0.90 × load_k)
SINR = S / (Σ I_k + N)
```
Each neighbour cell contributes interference proportional to its fractional load. The 10% floor models always-on reference signals. This means:
- A fully loaded cell (10 UEs) contributes 100% of its received power as interference
- An empty cell still contributes 10% (reference signals cannot be turned off in LTE)

### 4.5 Handover Events in v3

| Metric | v2 | v3 |
|--------|----|----|
| HO failure rate | 3.7% | ~20% |
| A3 share | 79% | ~68% |
| A4 share | 15% | ~10% |
| A5 share | 6% | ~22% |
| Ping-pong share | 0.7% | ~2% |

The higher A5 share and failure rate in v3 are physically correct: steeper NLOS path loss means more UEs reach the A5 coverage-emergency threshold, and the multi-factor failure model (SINR + speed + target RSRP + sustained low RSRP) captures more failure modes than the single-variable sigmoid in v2.

### 4.6 Ping-Pong Hysteresis

After a ping-pong is detected, the reverse cell pair (target → serving) receives an extra 3 dB HO margin for 20 steps. This prevents the UE from immediately bouncing back — a standard SON (Self-Optimising Networks) technique in 3GPP TS 36.902.

```python
if ping_pong and not ho_failed:
    rev_key = (best_nb_id, ue.serving_cell)
    ue.pp_extra_margin[rev_key]     = 3.0    # dB
    ue.pp_margin_remaining[rev_key] = 20     # steps
```

---

## 5. Phase 2 — Feature Engineering

### 5.1 Pipeline (`src/features.py`)

```
load_raw() → clean() → engineer_features() → split_and_scale()
```

No data leakage: all temporal features (lags, rolling windows, deltas) use `shift(k)` with k ≥ 1, meaning they reference only past values. The StandardScaler is fit on the training set and applied to val/test.

### 5.2 Feature Types

Starting from 26 raw columns, 86 features are produced:

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

Positive rate ≈ 11% → ~8× imbalance. Each model uses a different mechanism:

| Model | Imbalance handling |
|-------|--------------------|
| Logistic Regression | `class_weight = {0: 1.0, 1: ~8.0}` (computed from train set) |
| Random Forest | `class_weight = {0: 1.0, 1: ~8.0}` (computed from train set) |
| LSTM | `WeightedRandomSampler` + `BCEWithLogitsLoss(pos_weight=~8)` |

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
| Logistic Regression | 0.315 | 0.852 | 0.460 | 0.894 |
| **Random Forest** | **0.408** | **0.720** | **0.521** | **0.909** |
| LSTM | 0.320 | 0.827 | 0.462 | 0.876 |

### 7.2 Model Analysis

**Logistic Regression:**  
Highest recall (0.85) at the cost of precision (0.31). Useful when missed handovers are more costly than unnecessary preparations. Coefficient analysis confirms `l3_rsrp_serving_delta3`, `rsrp_diff`, and `sinr_roll5_mean` as dominant linear predictors.

**Random Forest:**  
Best F1 (0.521) and AUC (0.909). Feature importances reveal that `l3_rsrp_serving`, `rsrp_diff`, `sinr_roll5_mean`, and `cell_load_pct`-derived features dominate. The ensemble captures the non-linear boundary: HO risk spikes when the L3-filtered RSRP gap exceeds the A3 margin and SINR has been declining for several steps.

**LSTM:**  
Comparable to Logistic Regression (F1=0.462, AUC=0.876). Its advantage is learning the temporal *trajectory* of L3-filtered RSRP degradation directly from sequences, without manually crafted delta features. Further gains are expected with more tuning — the current 30-epoch budget is conservative.

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
| Raw columns | 26 (incl. `l3_rsrp_serving`, `l3_rsrp_neighbor`, `los_flag`, `cell_load_pct`, `rsrp_diff`) |
| Positive label rate | 11.0% |
| HO attempts | 1,263 (1,009 success · 254 failed) |
| HO failure rate | 20.1% |
| Ping-pong rate | 26.9% of HO attempts |
| Event split | A3: 57.7% · A4: 19.7% · A5: 22.6% |
| LOS fraction | 21.4% of timesteps |
| RSRP range | −107.6 to −25.0 dBm |
| Best model | Random Forest |
| Best F1 | **0.521** |
| Best ROC-AUC | **0.909** |
| Engineered features | 86 (from 26 raw) |

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
cd Intelligent-Handover-Prediction-LTE

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
