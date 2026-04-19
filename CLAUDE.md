# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Dataset generation (only needed once — output is data/raw/dataset.csv)
python simulate.py

# Full pipeline (phases 2 → 3 → 4 → 5)
python run_pipeline.py

# Individual phases
python run_pipeline.py --phase 2   # feature engineering
python run_pipeline.py --phase 3   # model training
python run_pipeline.py --phase 4   # evaluation
python run_pipeline.py --phase 5   # SHAP explanations

# Promote the best model to models/champion/
python scripts/promote_best_model.py

# Streamlit dashboard
streamlit run app/dashboard.py

# DVC pipeline (CI-safe — skips the heavy simulate step)
dvc repro features train evaluate explain

# Execute a notebook with the project kernel (required — default kernel lacks seaborn/torch)
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=300 \
  --ExecutePreprocessor.kernel_name=lte_venv \
  notebooks/<notebook>.ipynb
```

There is no lint or test suite; the CI runs `dvc repro` end-to-end as the integration test.

## Architecture

### Data flow

```
simulate.py
  └─ data/raw/dataset.csv (27 k rows, 15 UEs × 1800 s)
       └─ src/features.py  →  data/processed/{train,val,test}.csv + scaler.pkl + meta.json
            └─ src/models.py  →  models/{lr,rf,xgb,lstm,gru,stacking}.{pkl,pt} + mlflow_run_ids.json
                 ├─ src/evaluate.py  →  reports/{evaluation.txt,roc_curves.png,confusion_matrices.png}
                 └─ src/explain.py   →  reports/shap/shap_{bar,summary,waterfall}_*.png
                      └─ scripts/promote_best_model.py  →  models/champion/{model} + metadata.json
```

`run_pipeline.py` is the sole entry-point; it calls the `run_*()` function from each `src/` module in order. The DVC pipeline mirrors this exactly and is what the GitHub Actions CI uses.

### Simulation layer (`simulate.py`, `src/radio_model.py`, `src/mobility.py`, `src/handover_logic.py`)

- **`radio_model.py`** — 3GPP UMa LOS/NLOS path loss, Gudmundson AR(1) shadow fading, AR(1) complex Gaussian fast fading, L3 EMA filtering (`alpha=0.5`), SINR/RSRQ/CQI computation.
- **`mobility.py`** — Random Waypoint model with direction persistence (heading blend + noise). `UE` dataclass holds the full per-UE state including per-BS channel state (`shadow`, `ff_i`, `ff_q`, `los`, `l3_rsrp`), handover FSM state (`ttt_counter`, `ho_history`, `ping_pong_*`), and failure counters.
- **`handover_logic.py`** — A3/A4/A5 event classification, velocity-aware TTT (`velocity_ttt(speed_ms)`), multi-factor sigmoid failure probability (`ho_failure_prob`), and ping-pong hysteresis. The main state machine is `evaluate_handover()`.
- **`simulate.py`** — Assembles 4 BSs + 15 UEs (8 pedestrians, 7 vehicles), runs 1800-step simulation loop, emits the raw dataset. The label `handover_soon` is set to 1 if any handover occurs within the next 3 steps for that UE.

### Feature engineering (`src/features.py`)

Key constants at module level:
- `LAG_STEPS = [1, 2, 3]`, `ROLL_WINDOWS = [3, 5]`, `DELTA_STEPS = [1, 3]`
- Rolling stats are **shifted by 1** to avoid look-ahead; lags are backward-only — no data leakage.
- Applied to all 8 radio signals (raw + L3-filtered RSRP/RSRQ, SINR, CQI, neighbor RSRP/RSRQ).
- Split is **temporal** (70/15/15, no shuffling); `StandardScaler` fit on train only.
- `meta.json` records `feat_cols` (the canonical feature list used everywhere downstream).

### Model training (`src/models.py`)

Six models are trained in sequence:
| Model | Type | Key detail |
|---|---|---|
| `logistic_regression` | sklearn | `C=0.5, solver='saga', class_weight='balanced'` |
| `random_forest` | sklearn | `n_estimators=300, max_depth=12`, balanced |
| `xgboost` | XGBoost | `n_estimators=400, lr=0.05`, early stopping on val, **input must be float32** |
| `lstm` | PyTorch | `hidden=64, layers=2, dropout=0.3`, `SEQ_LEN=10`, WeightedRandomSampler |
| `gru` | PyTorch | Mirror of LSTM, GRU cell |
| `stacking_ensemble` | sklearn meta-LR | Meta-features = [XGB probs, RF probs, LSTM probs] |

**Stacking alignment is critical:** LSTM/GRU drop the first `SEQ_LEN-1` rows per UE (insufficient look-back). XGB and RF must be restricted to the same row subset when building meta-features. This row alignment uses `_build_seq_row_indices()` from `models.py` and is replicated in `evaluate.py` and notebook 10.

MLflow logging is wrapped in `try/except`; if the SQLite store (`mlflow.db`) is absent the pipeline still runs. The tracking URI is `sqlite:///mlflow.db` (project root).

### Evaluation & explainability (`src/evaluate.py`, `src/explain.py`)

- `evaluate.py` loads all 6 models and computes precision/recall/F1/ROC-AUC on the held-out test set. It then resumes each MLflow run to append test metrics + plot artifacts.
- `explain.py` runs SHAP only for LR (LinearExplainer + Independent masker), RF, and XGBoost (TreeExplainer). LSTM/GRU are excluded (DeepExplainer is too slow and noisy). SHAP 0.51 changed `TreeExplainer.shap_values()` to return a 3-D array `(n_samples, n_features, n_classes)`; the code handles both the old list format and the new ndarray.

### Champion model promotion (`scripts/promote_best_model.py`)

Queries MLflow for the run with the highest `test_roc_auc`. Falls back to parsing `reports/evaluation.txt` if MLflow is unavailable. Copies the winner to `models/champion/` and writes `metadata.json`.

### Dashboard (`app/dashboard.py`)

Seven-tab Streamlit app. Reads `data/processed/` CSVs, `models/` files, `reports/shap/` PNGs, and `models/champion/metadata.json`. Displays a champion banner, model-comparison table, per-UE KPI trends, handover timeline, real-time risk gauge, SHAP plots, and MLflow run table.

### Notebooks (`notebooks/`)

Notebooks 08–10 are analysis-only (read from already-executed pipeline outputs):
- `08_mlflow_experiment_tracking.ipynb` — connects to `mlflow.db`, visualises all runs
- `09_shap_explanations.ipynb` — stratified SHAP beeswarm, dependency, waterfall, temporal plots
- `10_model_comparison.ipynb` — PR curves, calibration, threshold sweep, speed benchmark, error analysis

All three must be executed with the `lte_venv` kernel (see command above).

## Important implementation notes

- **XGBoost requires float32 inputs** everywhere (training, eval, SHAP). Cast with `.astype(np.float32)`.
- **macOS OpenMP segfault**: set `OMP_NUM_THREADS=1` before importing `torch` when XGBoost is also imported in the same process.
- **`meta.json` is the source of truth** for `feat_cols`. Always load feature names from `data/processed/meta.json["feat_cols"]` rather than hard-coding them.
- **`mlflow.db` and `mlruns/`** are gitignored and regenerated by the pipeline. Do not commit them.
- **Model files (`*.pkl`, `*.pt`)** are gitignored. `data/processed/train.csv`, `val.csv`, `test.csv` are also gitignored; only `meta.json` and `data/raw/dataset.csv` are retained via DVC.
- All pipeline configuration (signal ranges, TTT, fading params, feature lag windows, model hyperparameters) is in **module-level dataclasses and constants** in each `src/` file — there is no external YAML config.
