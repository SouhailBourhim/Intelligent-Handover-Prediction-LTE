# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Full pipeline (8 DVC stages: simulate → features → train → evaluate → explain
#   → promote → generalize → closed_loop), ~7 min on CPU; reruns only stale stages
dvc repro
dvc status

# Same steps by hand
python simulate.py                       # data/raw/dataset.csv
python run_pipeline.py                   # phases 2–5 (or --phase 2|3|4|5)
python scripts/promote_best_model.py     # models/champion/ (selection on validation)
python -m src.generalization             # reports/generalization.json
python -m src.closed_loop                # reports/closed_loop.json (~4 min)

# Tests (36, ~10 s) — CI runs these before the pipeline
pytest -q

# Inference on raw measurements (any row order)
python predict.py --csv data/test_scenarios/default_ues5_steps300_seed42.csv
python predict.py --info

# Scenario CSVs (default, vehicle, stable, cell_edge); samples in data/test_scenarios/
python generate_test_dataset.py --scenario vehicle --ues 8

# Figures and the French course PDF
python scripts/make_figures.py           # docs/img/ (reads reports/*.json)
python scripts/make_course_figures.py    # docs/cours/img/
python scripts/build_course_pdf.py       # needs Edge or Chrome

streamlit run app/dashboard.py

# Notebooks: run nbconvert as a module of the project venv so the kernel resolves
python -m ipykernel install --sys-prefix --name lte_venv   # once, inside the venv
python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 --ExecutePreprocessor.kernel_name=lte_venv \
  notebooks/<notebook>.ipynb
```

On Windows set `PYTHONIOENCODING=utf-8`; several scripts print box-drawing characters.

## Architecture

### Data flow

```
src/simulator.py (via simulate.py)
  └─ data/raw/dataset.csv (54 k rows, 15 UEs × 3600 s, 27 columns)
       └─ src/features.py  →  data/processed/{train,val,test}.csv + models/scaler.pkl + meta.json
            └─ src/models.py  →  models/*.pkl|*.pt (7 models incl. rule baseline)
                                + models/thresholds.json (validation thresholds & metrics)
                 ├─ src/evaluate.py (+ operational.py) → reports/evaluation.{txt,json},
                 │                                        operational.json, roc/pr/cm PNGs
                 ├─ src/explain.py   →  reports/shap/
                 └─ scripts/promote_best_model.py → models/champion/{model files, metadata.json}
                      ├─ src/generalization.py → reports/generalization.json
                      └─ src/closed_loop.py    → reports/closed_loop.json
```

### Simulation (`src/simulator.py`, `radio_model.py`, `mobility.py`, `handover_logic.py`)

- **`simulator.py`** owns the step loop shared by `simulate.py`, `generate_test_dataset.py`, generalization and the closed loop. Each step: phase 1 *measure* (move, channel, L3, SINR/RSRQ/CQI) for all UEs, then optional `ttt_policy(t, recs) -> {ue_id: ttt}`, then phase 2 *decide* (A3 state machine, RLF countdown, RRC re-establishment). It uses **two RNG streams**: `rng` (mobility + channel) and `ho_rng` (failure draws), so policies never change trajectories (tested). Records carry internal `_a3_gap`/`_in_rlf` keys for policies; `run()` strips them. `make_simulator(ue_factory, seed, policy)` builds both streams from one seed. `add_handover_soon_label` lives here.
- **`radio_model.py`**:
  - RSRP is **per resource element** (`RadioConfig.rs_power_dbm` = 46 − 10·log10(600)), reported via `report_rsrp` in [−140, −44] dBm; the simulator uses `compute_rsrp(..., report=False)` internally and clamps only when writing rows.
  - Noise per RE, so SINR equals the wideband ratio (tested).
  - LOS is spatially consistent (`update_los_latent` / `los_from_latent`, 50 m).
  - Fast fading is power-averaged over 1 + Δd/(λ/2) samples. In `update_fast_fading`, `ff_i` holds the last averaged dB value.
- **`mobility.py`**: `UE.speed` is the cruise speed of the current leg (redrawn from `speed_range` at each waypoint); `UE.current_speed` is the distance actually travelled / dt (0 while paused) and is what TTT, failure and the `ue_speed` column use.
- **`handover_logic.py`**:
  - Thresholds are on the per-RE scale (A4 −83, A5 −96/−88, low RSRP −113, target ramp −98→−133).
  - `evaluate_handover(..., ttt_override=None)`; A3 is the only trigger, and `event_type` is a descriptive tag.
  - Only executed handovers enter `ho_history` (ping-pong detection).

### Features (`src/features.py`)

- `SIGNAL_COLS` (8) × {value, lags 1–3, rolling mean/std 3 & 5 (shifted by 1), deltas 1 & 3} + `CONTEXT_COLS` (`ue_speed`, `rsrp_diff`, `cell_load_pct`) = **83 features** (`SCALE_COLS`). `pos_x`, `pos_y`, `los_flag` are raw-only, never features.
- **`build_features(df, drop_warmup=True)`** is the single offline builder: groups by UE, sorts by timestamp, **keeps the input index**, adds `history_steps`/`warmup` (first `WARMUP_STEPS`=3 rows per UE). `engineer_features` = `build_features(...).reset_index(drop=True)`.
- **`OnlineFeatureBuffer.push(rec)`** reproduces the same features in streaming mode (parity test).
- Temporal 70/15/15 split; scaler fit on train only; `meta.json["feature_cols"]` is the source of truth.

### Models (`src/models.py`)

Seven models: LR, RF, XGBoost, LSTM, GRU, Stacking, and `RuleA3Gap` (a picklable sklearn-like class over scaled features).

- **Imbalance:** one correction per model (class_weight / scale_pos_weight / sampler only for LSTM/GRU, never also `pos_weight`).
- **XGBoost:** early stopping on validation `aucpr`.
- **LSTM/GRU:** checkpoint on validation PR-AUC. **Sequence scoring must go through `score_sequence_model(net, df, feat_cols)`**, which builds windows per UE and returns a Series on `df.index`.
- **Stacking:** class-balanced meta-LR. `train_stacking` returns `(model, oof_val_f1, oof_series)`, and its validation metrics are out-of-fold.
- **After training:** every model is scored on the validation rows with a full `SEQ_LEN` history. `tune_threshold` / `val_metrics` write `models/thresholds.json` and log to MLflow.
- `seed_everything()` makes runs bit-for-bit reproducible.

MLflow (SQLite `mlflow.db`) calls are wrapped in try/except, and run IDs are recorded before artifact logging. sklearn models are logged through `_mlf_log_sklearn`: MLflow 3 needs `serialization_format="cloudpickle"`, because skops rejects tree objects.

### Evaluation (`src/evaluate.py`, `src/operational.py`)

- **`score_all_models(df, feat_cols)`** returns probabilities of all 7 models on `df.index`.
- **Evaluation rows:** every model is evaluated on the same test rows (`_build_seq_row_indices`) at its validation threshold, with PR-AUC included.
- **`operational.py`:**
  - `event_metrics` computes event recall@3/5 s, lead time and **false-alarm time share** (time-based on purpose; an episode count rewards permanent alarms);
  - `operating_curve`;
  - `recall_at_false_alarm_budget`, with `FA_BUDGET` = 5 % in `evaluate.py`.

### Promotion (`scripts/promote_best_model.py`)

Selects on **validation** (`0.6·val_f1_tuned + 0.4·val_roc_auc` from `thresholds.json`). The rule is not eligible; test metrics come from `reports/evaluation.json` for reporting only. Writes `metadata.json` with `display_name`, `threshold`, val/test metrics and `run_id`. A stacking champion copies its base models too.

### Serving (`src/serving.py`)

The single inference path: `load_model(name)`, `load_champion()`, `predict_scaled(model, scaled_df, feat_cols)` and `score_measurements(raw_df, model)`. The last returns `prob`/`warmup`/`alarm` on the raw index. It's used by `predict.py`, the dashboard (`score_live`, risk tab) and `generalization.py`. Never attach scores by position.

### Experiments

- **`closed_loop.py`:** policies baseline / always_short / always_long / predictive / random_gate on seeds 101/202/303 × 1800 s.
  - The predictive `hi` is the champion's validation threshold; `lo` is the 5th percentile of its validation probabilities on positives.
  - The random gate matches the predictive override rates.
  - Non-sklearn champions fall back to XGBoost in the loop.
- **`generalization.py`:** champion vs rule on the 4 scenario presets and unseen seeds 1001–1003.

### Dashboard (`app/dashboard.py`)

Nine tabs, including **Proactive HO** (closed loop + generalization JSON). The model comparison reads `reports/evaluation.json` and `operational.json`, and live prediction uses `score_live` (joins by index).

## Important implementation notes

- **XGBoost requires float32 inputs** everywhere (`.astype(np.float32)`).
- **macOS OpenMP:** set `OMP_NUM_THREADS=1` before importing torch when XGBoost is also loaded.
- **Git-tracked vs generated:**
  - `reports/*.json`, `models/thresholds.json`, `models/champion/metadata.json` and `dvc.lock` are git-tracked (`cache: false` outputs);
  - CSVs, `*.pkl`/`*.pt`, PNG reports, `reports/shap/` and `mlflow.db` are regenerated and ignored.
  - There is no DVC remote: data is reproducible from the seeded simulator.
- **Commits:** do not add Claude co-author / attribution lines (author's preference).
- All configuration lives in module-level dataclasses/constants (`RadioConfig`, `HOConfig`, `MobilityConfig`, feature constants, model hyperparameters); there is no YAML config.
- `docs/cours/` is a French course built on this project. Keep its numbers in sync with `reports/*.json` when results change.
