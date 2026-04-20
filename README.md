# Intelligent Handover Prediction in LTE Networks using ML

A full machine-learning pipeline that predicts **imminent handover events** (`handover_soon`) in LTE networks from simulated UE radio measurements.

---

## Project Overview

| Item | Detail |
|------|--------|
| Task | Binary time-series classification |
| Label | `handover_soon = 1` if a successful HO occurs within the next 3 steps — see [label note](#label-lookahead-note) |
| Dataset | 27,000 rows · 15 UEs · 1,800 s simulation |
| Simulator | v3 — 3GPP UMa LOS/NLOS, Random Waypoint mobility, A3/A4/A5 events |
| Models | LR · RF · XGBoost · LSTM · GRU · Stacking Ensemble |
| Best F1 | Random Forest (F1 = 0.521) |
| Best AUC | XGBoost (ROC-AUC = 0.912) |
| Champion | Random Forest (score = 0.6×F1 + 0.4×AUC) — see [champion criterion](#champion-model--model-promotion) |
| MLOps | MLflow experiment tracking · DVC pipeline · GitHub Actions CI |

---

## Repository Structure

```
├── simulate.py                   # Phase 1 — LTE network simulation (v3)
├── run_pipeline.py               # Master runner (phases 2–5)
├── predict.py                    # Inference script — score new measurements
├── requirements.txt
├── dvc.yaml                      # DVC pipeline definition (5 stages)
├── .python-version               # Pins Python 3.11
│
├── src/
│   ├── radio_model.py            # 3GPP UMa path loss, shadow/fast fading, SINR, CQI
│   ├── mobility.py               # Random Waypoint UE mobility (pedestrian + vehicle)
│   ├── handover_logic.py         # A3/A4/A5 events, L3 filter, TTT, HO failure, ping-pong
│   ├── features.py               # Phase 2 — feature engineering + train/val/test split
│   ├── models.py                 # Phase 3 — all 6 models + MLflow logging
│   ├── evaluate.py               # Phase 4 — metrics, ROC curves, confusion matrices
│   └── explain.py                # Phase 5 — SHAP explanations (LR, RF, XGBoost)
│
├── scripts/
│   └── promote_best_model.py     # Queries MLflow, copies best model to models/champion/
│
├── data/
│   ├── raw/dataset.csv           # 27k-row simulated dataset
│   └── processed/                # train/val/test splits + meta.json
│
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_dashboard_preview.ipynb
│   ├── 06_simulation_realism.ipynb    # v3 radio model analysis
│   ├── 07_model_impact.ipynb          # model comparison
│   ├── 08_mlflow_experiment_tracking.ipynb  # MLflow run visualisation
│   ├── 09_shap_explanations.ipynb           # deep-dive SHAP analysis
│   └── 10_model_comparison.ipynb            # PR curves, calibration, speed benchmark
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── lstm.pt
│   ├── gru.pt
│   ├── stacking_ensemble.pkl
│   ├── scaler.pkl
│   ├── mlflow_run_ids.json           # Maps model name → MLflow run ID
│   └── champion/
│       ├── stacking_ensemble.pkl     # Copy of the promoted champion
│       └── metadata.json            # Name, AUC, F1, promotion timestamp
│
├── reports/
│   ├── evaluation.txt
│   ├── roc_curves.png
│   ├── confusion_matrices.png
│   └── shap/                        # 9 SHAP plots (bar + beeswarm + waterfall × 3 models)
│
├── docs/                            # Architecture and setup reference
├── app/
│   └── dashboard.py                 # Streamlit dashboard (7 tabs)
├── mlflow.db                        # MLflow SQLite store (auto-created, gitignored)
└── .github/
    └── workflows/
        └── ci.yml                   # GitHub Actions CI
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/SouhailBourhim/Intelligent-Handover-Prediction-LTE.git
cd Intelligent-Handover-Prediction-LTE

# 2. Create environment  (Python 3.11 recommended)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Generate dataset
python simulate.py

# 4. Run full pipeline (features → training → evaluation → SHAP)
python run_pipeline.py

# 5. Promote best model to champion
python scripts/promote_best_model.py

# 6. Launch dashboard
streamlit run app/dashboard.py
```

Run individual phases:

```bash
python run_pipeline.py --phase 2   # feature engineering only
python run_pipeline.py --phase 3   # training only (all 6 models)
python run_pipeline.py --phase 4   # evaluation only
python run_pipeline.py --phase 5   # SHAP explanations only
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

#### Label lookahead note {#label-lookahead-note}

`handover_soon` looks **3 steps ahead** (3 s at 1 s/step).  This is a deliberate tradeoff:
- **Shorter horizon (1–2 s)** — higher achievable precision (less noise), but the prediction arrives too late for the network to pre-configure the target cell.
- **3-step horizon** — gives the scheduler ~3 s of preparation time, which is sufficient for standard X2-based handover signalling.
- **Longer horizon (5–10 s)** — more operationally useful but label noise increases sharply because mobility is unpredictable over many seconds, making the positive class harder to separate and reducing max achievable precision.

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
| `pos_x`, `pos_y` | UE position (m) — **simulation-only ground truth**, not available in real LTE deployments; excluded from engineered features |
| `los_flag` | 1 if serving link is currently LOS — **simulation-only ground truth**, not observable in real networks; excluded from engineered features |
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

Six models are trained in sequence. Each run is logged to MLflow automatically (see [MLflow](#mlflow-experiment-tracking)).

| Model | Key settings |
|-------|-------------|
| Logistic Regression | C=0.5, saga solver, class_weight balanced |
| Random Forest | 300 trees, max_depth=12, class_weight balanced |
| XGBoost | n_estimators=400, max_depth=6, lr=0.05, early stopping (20 rounds) |
| LSTM | 2 layers × 64 hidden, seq_len=10, WeightedRandomSampler |
| GRU | 2 layers × 64 hidden, seq_len=10 — lighter recurrent alternative to LSTM |
| Stacking Ensemble | meta-LR trained on val-set probs from XGBoost + RF + LSTM |

**Stacking details:** the meta-learner receives three probability columns (one per base model) aligned to the same row subset. LSTM/GRU drop the first `SEQ_LEN−1` rows per UE (no full look-back window); `_build_seq_row_indices()` reconstructs those rows so the sklearn models cover the exact same subset when building meta-features.

### Phase 4 — Evaluation (`src/evaluate.py`)

| Model | Precision | Recall | F1 | ROC-AUC | Score† |
|-------|-----------|--------|----|---------|--------|
| Logistic Regression | 0.315 | 0.852 | 0.460 | 0.894 | 0.634 |
| **Random Forest** ★ | 0.408 | 0.720 | **0.521** | 0.909 | 0.677 |
| XGBoost | 0.409 | 0.684 | 0.512 | **0.912** | 0.672 |
| LSTM | 0.319 | 0.832 | 0.461 | 0.887 | 0.631 |
| GRU | 0.318 | 0.843 | 0.462 | 0.882 | 0.630 |
| Stacking Ensemble | **0.517** | 0.322 | 0.397 | 0.912 | 0.603 |

† Score = 0.6 × F1 + 0.4 × AUC — the champion promotion criterion.

★ **Random Forest** is the current **champion** by weighted score (0.677).  It achieves the best F1 of all six models, meaning it balances recall (catching imminent handovers) against false alarms more effectively than any other model.

**Stacking Ensemble footnote:** The ensemble's low F1 (0.397) does not disqualify it — its AUC of 0.912 shows excellent discriminative ability at all thresholds, and its precision of 0.517 is the highest of any model.  However, at the default 0.5 threshold its recall (0.322) is too low to be operationally safe: more than two-thirds of imminent handovers go unpredicted, which would cause connectivity drops.  Lowering the threshold to 0.25–0.30 substantially recovers recall at the cost of more false alarms.  The weighted score criterion reflects this by penalising low F1, so the Stacking Ensemble is not promoted despite its high AUC.

**Random Forest** maximises F1. **XGBoost** and **Stacking** are effectively tied on AUC. **LSTM/GRU** capture raw temporal RSRP trajectories without manual feature engineering.

### Phase 5 — SHAP Explanation (`src/explain.py`)

Computes SHAP values for four models and saves twelve plots to `reports/shap/`:

| Model | Explainer | Why this one |
|-------|-----------|--------------|
| Logistic Regression | `LinearExplainer` | Exact, analytical — instant |
| Random Forest | `TreeExplainer` | Exact tree-path SHAP — fast |
| XGBoost | `TreeExplainer` | Exact tree-path SHAP — fast |
| Stacking Ensemble | `KernelExplainer` | Model-agnostic; features are the 3 base-model output probabilities — shows which base model drives each prediction |

Three plot types per model:

| Plot | File | What it shows |
|------|------|---------------|
| Bar chart | `shap_bar_<model>.png` | Top-20 features ranked by mean \|SHAP\| |
| Beeswarm | `shap_summary_<model>.png` | Per-sample SHAP distribution, coloured by feature value |
| Waterfall | `shap_waterfall_<model>.png` | Single positive prediction decomposed feature-by-feature |

LSTM and GRU are excluded (DeepExplainer would take minutes and produce noisier values than TreeExplainer).

SHAP plots are also logged as artifacts into each model's MLflow run under the `shap/` sub-directory.

### Phase 6 — Dashboard (`app/dashboard.py`)

Interactive Streamlit app with seven tabs:

| Tab | Content |
|-----|---------|
| 📊 Radio KPIs | RSRP, SINR, RSRQ, CQI charts with handover markers |
| 🔁 HO Timeline | Scatter of handover events by UE and target cell |
| ⚠️ Risk | Per-UE risk heatmap + gauge for the selected model |
| 📋 Model Comparison | Metrics table parsed from `reports/evaluation.txt` |
| 🔍 SHAP Explanations | Bar / beeswarm / waterfall plots; on-demand recompute button |
| 🧪 MLflow Runs | Live query of `mlflow.db` — val F1, test AUC, status |
| 🗺️ Mobility Map | UE position tracks + base station markers |

A **champion model banner** at the top reads `models/champion/metadata.json` and displays the promoted model name, AUC, and promotion timestamp.

---

## Inference — `predict.py`

Score new UE measurements against the champion model without running the full pipeline:

```bash
# Show current champion info
python predict.py --info

# Score a CSV file (one row per timestep)
python predict.py --csv path/to/measurements.csv

# Score a single row as JSON
python predict.py --json '{"rsrp_serving": -85, "sinr": 12, ...}'

# Custom decision threshold
python predict.py --csv measurements.csv --threshold 0.35
```

**Example output:**

```
Model: stacking_ensemble  |  AUC=0.9119

    UE     Prob    Decision  (threshold=0.50)
───────────────────────────────────────────
     3   0.7821  ⚠  HANDOVER
     3   0.0431  ✓  ok
     7   0.1204  ✓  ok
```

The script handles all model types automatically:
- **sklearn models** (LR, RF, XGBoost) — direct `predict_proba`
- **PyTorch models** (LSTM, GRU) — sliding window sequences with padding
- **Stacking Ensemble** — transparently loads base models and composes meta-features

Input CSVs only need the raw signal columns; any missing engineered features (lags, rolling stats, deltas) are filled with zero so the script works with minimal inputs.

---

## Notebooks

All 10 notebooks have pre-computed outputs and require the `lte_venv` (or project `.venv`) kernel to re-execute.

| Notebook | Content |
|----------|---------|
| 01 | Data generation walkthrough |
| 02 | Feature engineering deep-dive |
| 03 | Model training step-by-step |
| 04 | Evaluation + ROC/confusion visualisation |
| 05 | Dashboard preview |
| 06 | Simulation realism analysis (v3 radio model) |
| 07 | Side-by-side model impact comparison |
| **08** | **MLflow experiment tracking** — run comparison, radar chart, champion spotlight |
| **09** | **SHAP explanations** — beeswarm, dependency plots, waterfall, temporal SHAP per UE |
| **10** | **Model comparison deep-dive** — PR curves, calibration, threshold sweep, speed benchmark, error analysis |

To re-execute any notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=300 \
  --ExecutePreprocessor.kernel_name=lte_venv \
  notebooks/<notebook>.ipynb
```

---

## MLflow Experiment Tracking

MLflow is an **optional** dependency — the pipeline runs normally without it. When installed, every training run is automatically logged.

MLflow uses a **local SQLite database** by default — no server required, and all UI tabs (including the Overview tab) work correctly:

```
mlflow.db            ← created automatically on first run (gitignored)
```

> **Why not `mlruns/`?** MLflow's file-based backend does not support the Overview tab or several comparison features. SQLite is the recommended single-machine alternative and requires zero extra setup.

### What gets logged

| Phase | What is logged |
|-------|---------------|
| Training (phase 3) | Hyperparameters, validation F1 score, serialised model artifact |
| Evaluation (phase 4) | Test precision, recall, F1, ROC-AUC; `roc_curves.png` and `confusion_matrices.png` as artifacts |

Run IDs are saved to `models/mlflow_run_ids.json` during training so that the evaluation phase can **resume the same run** and append test metrics rather than create a second run per model.

### Viewing runs in the browser

```bash
# From the project root — opens http://localhost:5000
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then navigate to the **lte_handover_prediction** experiment to compare runs side-by-side, inspect logged parameters, and download plot artifacts.

Alternatively, view runs directly in the **Streamlit dashboard** under the 🧪 MLflow Runs tab (no separate server needed).

### Querying runs from Python

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient()

exp = client.get_experiment_by_name("lte_handover_prediction")
runs = client.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["metrics.test_roc_auc DESC"],
)

for r in runs:
    name = r.data.tags.get("mlflow.runName")
    auc  = r.data.metrics.get("test_roc_auc")
    f1   = r.data.metrics.get("test_f1")
    print(f"{name:<25} AUC={auc:.4f}  F1={f1:.4f}")
```

---

## Champion Model & Model Promotion

After evaluation, `scripts/promote_best_model.py` selects the run with the highest **weighted score (0.6 × F1 + 0.4 × AUC)** and copies its artifact to `models/champion/`.

**Why F1 is weighted higher than AUC:**  
In LTE handover prediction, a missed handover (false negative) causes an immediate connectivity drop and potential Radio Link Failure, which requires recovery signalling and degrades user experience.  A false alarm (false positive) merely triggers a preparation phase that the network aborts within milliseconds — a much lower cost.  F1 penalises false negatives more directly than AUC, so it is weighted at 60 % to reflect this cost asymmetry.  Pure AUC optimisation can crown a model that achieves high discriminative ability but sets its decision boundary so conservatively that it misses most real handovers at the default threshold.

```bash
python scripts/promote_best_model.py
```

```
models/champion/
├── stacking_ensemble.pkl   # copy of the winning model
└── metadata.json           # { model_name, run_id, test_roc_auc, test_f1, promoted_at }
```

The dashboard reads `metadata.json` on startup and displays the champion banner. If MLflow is not available, the script falls back to parsing `reports/evaluation.txt`.

---

## DVC Pipeline

[DVC](https://dvc.org) tracks data and model artifacts and defines five pipeline stages in `dvc.yaml`:

| Stage | Command | Inputs | Outputs |
|-------|---------|--------|---------|
| `simulate` | `python simulate.py` | `src/radio_model.py`, `src/mobility.py`, `src/handover_logic.py` | `data/raw/dataset.csv` |
| `features` | `python run_pipeline.py --phase 2` | `src/features.py`, `data/raw/dataset.csv` | `data/processed/`, `models/scaler.pkl` |
| `train` | `python run_pipeline.py --phase 3` | `src/models.py`, `data/processed/` | `models/*.pkl`, `models/*.pt` |
| `evaluate` | `python run_pipeline.py --phase 4` | `src/evaluate.py`, `data/processed/test.csv`, `models/` | `reports/evaluation.txt`, `reports/*.png` |
| `explain` | `python run_pipeline.py --phase 5` | `src/explain.py`, `data/processed/test.csv`, `models/` | `reports/shap/` |

Reproduce the full pipeline:

```bash
pip install dvc
dvc repro
```

DVC detects which stages are stale (inputs changed) and only re-runs those. Run `dvc dag` to visualise the dependency graph.

---

## SHAP Explanation

[SHAP](https://shap.readthedocs.io) assigns each feature a contribution score for every prediction, grounded in cooperative game theory. Three explainer types are used depending on the model:

| Explainer | Models | Speed |
|-----------|--------|-------|
| `TreeExplainer` | XGBoost, Random Forest | Fast — exact tree-path computation |
| `LinearExplainer` | Logistic Regression | Instant — analytical solution |

Plots are saved to `reports/shap/` — twelve files total (three per model):

```
reports/shap/
├── shap_bar_logistic_regression.png
├── shap_bar_random_forest.png
├── shap_bar_xgboost.png
├── shap_bar_stacking_ensemble.png
├── shap_summary_logistic_regression.png
├── shap_summary_random_forest.png
├── shap_summary_xgboost.png
├── shap_summary_stacking_ensemble.png
├── shap_waterfall_logistic_regression.png
├── shap_waterfall_random_forest.png
├── shap_waterfall_xgboost.png
└── shap_waterfall_stacking_ensemble.png
```

**Bar chart** — global importance ranked by mean |SHAP|. The top features across all three models are `rsrp_diff`-related deltas and rolling statistics, confirming that the rate of change in the serving/neighbour RSRP gap is the strongest predictor of an imminent handover.

**Beeswarm** — one dot per test sample, coloured by feature value (red = high, blue = low). A cluster of red dots on the right means *high feature value increases handover risk*.

**Waterfall** — single positive-class prediction decomposed feature-by-feature, starting from the model base value and ending at the final output. Useful for explaining *why* a specific UE was flagged.

---

## CI/CD — GitHub Actions

`.github/workflows/ci.yml` runs on every push and pull request to `main`:

1. **Install dependencies** — `pip install -r requirements.txt && pip install dvc`
2. **Init DVC** — `dvc init --no-scm`
3. **Reproduce pipeline** — `dvc repro` (all five stages: simulate → features → train → evaluate → explain)
4. **Promote champion** — `python scripts/promote_best_model.py`
5. **Upload artifacts** — `reports/` and `models/champion/metadata.json` retained for 30 days

---

## Results

![ROC Curves](reports/roc_curves.png)
![Confusion Matrices](reports/confusion_matrices.png)

---

## Requirements

- Python 3.11 (`.python-version` file included)
- pandas · numpy · scikit-learn · xgboost · torch · streamlit · plotly · seaborn · joblib · statsmodels · mlflow · dvc · shap

See [`requirements.txt`](requirements.txt) for pinned versions.

> **macOS note:** PyTorch and XGBoost both initialise OpenMP threads. On macOS this can cause a segfault when both are loaded in the same process. The pipeline sets `OMP_NUM_THREADS=1` before importing torch to prevent this — no action needed from the user.

---

## Author

**Souhail Bourhim** — LTE network simulation + ML pipeline
