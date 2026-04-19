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

## Score new measurements

```bash
# Show current champion
python predict.py --info

# Score a CSV file
python predict.py --csv path/to/measurements.csv

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
