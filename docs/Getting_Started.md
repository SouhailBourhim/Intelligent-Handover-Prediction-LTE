# Getting Started

## Requirements

- Python 3.11 (`.python-version` file included)
- ~2 GB disk space (models + MLflow artifacts)
- No GPU required — LSTM/GRU train on CPU in under 5 minutes

## Installation

```bash
git clone https://github.com/SouhailBourhim/Intelligent-Handover-Prediction-LTE.git
cd Intelligent-Handover-Prediction-LTE

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the full pipeline

```bash
# Generate the 27k-row LTE dataset (takes ~30 s)
python simulate.py

# Feature engineering → train all 6 models → evaluate → SHAP explanations
python run_pipeline.py

# Promote the best model to models/champion/
python scripts/promote_best_model.py
```

Run a single phase without re-running the others:

```bash
python run_pipeline.py --phase 2   # features
python run_pipeline.py --phase 3   # training
python run_pipeline.py --phase 4   # evaluation
python run_pipeline.py --phase 5   # SHAP
```

## Launch the dashboard

```bash
streamlit run app/dashboard.py
# Opens http://localhost:8501
```

## Generate test datasets

`generate_test_dataset.py` creates small CSVs you can upload straight to the dashboard's **🔮 Live Prediction** tab or pass to `predict.py --csv`:

```bash
# Default (mixed scenario, 5 UEs, 300 steps) → data/test_scenarios/default_ues5_steps300_seed42.csv
python generate_test_dataset.py

# Vehicle scenario — all fast UEs (8–20 m/s), higher HO rate
python generate_test_dataset.py --scenario vehicle --ues 8 --steps 500

# Cell-edge scenario — UEs near cell boundaries, worst-case RSRP
python generate_test_dataset.py --scenario cell_edge --ues 6 --steps 400

# Stable scenario — slow pedestrians near BSs, low HO rate
python generate_test_dataset.py --scenario stable --ues 3 --steps 200

# Custom output path
python generate_test_dataset.py --scenario default --output my_test.csv
```

Four pre-generated sample files are already committed to `data/test_scenarios/` — you can use them immediately without running the script.

## Score new measurements

```bash
# Show current champion
python predict.py --info

# Score a CSV file (or use a pre-generated scenario file)
python predict.py --csv data/test_scenarios/vehicle_ues4_steps200_seed42.csv

# Score a single row as JSON (only raw signal columns needed)
python predict.py --json '{"rsrp_serving": -92, "sinr": 6.5, "ue_speed": 14}'
```

## View MLflow experiment

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Opens http://localhost:5000  →  navigate to "lte_handover_prediction"
```

## DVC pipeline

If you have DVC installed, the pipeline is fully reproducible:

```bash
pip install dvc
dvc repro          # runs only stale stages
dvc dag            # visualise the dependency graph
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: seaborn` when running notebooks | Use the project kernel: `--ExecutePreprocessor.kernel_name=lte_venv` |
| Segfault on macOS when XGBoost and PyTorch are both imported | Already handled — `OMP_NUM_THREADS=1` is set in `src/models.py` before torch import |
| `Experiment not found` in notebook 08 | Run `python run_pipeline.py` phases 3 + 4 at least once to populate `mlflow.db` |
| `models/champion/metadata.json not found` in `predict.py` | Run `python scripts/promote_best_model.py` after the pipeline |
| DVC `features` stage fails with missing dataset | Run `python simulate.py` first, or run `dvc repro` which includes the `simulate` stage |
| Dashboard Live Prediction tab shows "No scaler found" | Run the full pipeline (`python run_pipeline.py`) at least once so `models/scaler.pkl` exists |
| Uploaded CSV columns not recognised in Live Prediction | Use a file from `data/test_scenarios/` or generate one with `generate_test_dataset.py` to ensure correct column names |
