# Architecture

## System overview

```
simulate.py  ──►  run_pipeline.py (phases 2–5)  ──►  app/dashboard.py
                       │                                    │
              ┌────────┴─────────┐                 🔮 Live Prediction tab
        src/features.py     src/models.py          (generate_test_dataset.py
              │                  │                  or predict.py --csv)
        data/processed/    models/*.pkl / *.pt
              │                  │
        src/evaluate.py ◄────────┘
              │
        reports/ + MLflow (mlflow.db)
              │
        src/explain.py  ──►  reports/shap/
              │
        scripts/promote_best_model.py  ──►  models/champion/
```

## Module responsibilities

### Simulation layer (`src/` + standalone scripts)

| Module | Responsibility |
|--------|---------------|
| `radio_model.py` | 3GPP UMa LOS/NLOS path loss, Gudmundson AR(1) shadow fading, AR(1) complex Gaussian fast fading, L3 EMA filter, SINR/RSRQ/CQI computation |
| `mobility.py` | Random Waypoint model with direction persistence; `UE` dataclass holds full per-UE state (position, channel, HO FSM) |
| `handover_logic.py` | A3/A4/A5 event classification, velocity-aware TTT, multi-factor HO failure sigmoid, ping-pong hysteresis |
| `simulate.py` | Assembles the full 15-UE / 1800-step training dataset (`data/raw/dataset.csv`) |
| `generate_test_dataset.py` | Standalone script — generates smaller CSVs for dashboard live-prediction and `predict.py` testing; 4 scenario presets (`default`, `vehicle`, `stable`, `cell_edge`); pre-generated samples in `data/test_scenarios/` |

### Pipeline layer

| Module | Phase | Responsibility |
|--------|-------|---------------|
| `src/features.py` | 2 | Load + clean raw CSV, engineer lag/rolling/delta features per UE, 70/15/15 temporal split, StandardScaler |
| `src/models.py` | 3 | Train LR, RF, XGBoost, LSTM, GRU, Stacking Ensemble; log hyperparams + val F1 to MLflow |
| `src/evaluate.py` | 4 | Load all 6 models, score test set, log metrics + plots to MLflow, write `reports/evaluation.txt` |
| `src/explain.py` | 5 | SHAP values for LR (LinearExplainer), RF, XGBoost (TreeExplainer); bar, beeswarm, waterfall plots |

### MLOps layer

| Component | Role |
|-----------|------|
| `mlflow.db` | SQLite MLflow tracking store — experiment runs, metrics, hyperparams, artifact paths |
| `dvc.yaml` | Five-stage reproducible pipeline: simulate → features → train → evaluate → explain |
| `.github/workflows/ci.yml` | GitHub Actions — runs `dvc repro` end-to-end on every push to `main` |
| `scripts/promote_best_model.py` | Queries MLflow for highest `test_roc_auc`, copies winner to `models/champion/` |
| `predict.py` | CLI inference entry point — loads champion model, handles sklearn/PyTorch/stacking; also powers the dashboard Live Prediction tab |
| `app/dashboard.py` | Eight-tab Streamlit app — KPI charts, HO timeline, risk gauge, 🔮 Live Prediction, model comparison, SHAP, MLflow, mobility map |

## Key design decisions

**Stacking row alignment** — LSTM/GRU drop the first `SEQ_LEN−1` rows per UE (no look-back history). The helper `_build_seq_row_indices()` returns the exact row indices that were scored by LSTM so XGBoost and RF can be restricted to the same subset when constructing stacking meta-features. This alignment is applied consistently in `models.py`, `evaluate.py`, and `predict.py`.

**L3 filter separation** — Raw RSRP and L3-filtered RSRP are both included as features. The handover state machine uses the L3 value; the ML models learn from both, allowing them to detect the difference between instantaneous dips and sustained degradation.

**MLflow optional** — All MLflow calls in `models.py` and `evaluate.py` are wrapped in `try/except`. The pipeline runs and produces correct outputs even if MLflow is not installed.

**XGBoost float32** — All inputs to XGBoost are explicitly cast to `float32` to maintain compatibility across XGBoost 2.x and 3.x.
