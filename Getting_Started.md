# Getting Started

## Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/SouhailBourhim/Intelligent-Handover-Prediction-LTE.git
   cd Intelligent-Handover-Prediction-LTE
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Python Environment Setup
- It's recommended to use a virtual environment for Python.
- Set up a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
  ```

## Quickstart Guide

### Phase 1: Data Ingestion
Run the following command to ingest data:
```bash
python data_ingestion.py
```

### Phase 2: Data Preprocessing
Run the following command to preprocess the data:
```bash
python data_preprocessing.py
```

### Phase 3: Feature Engineering
To engineer features, execute:
```bash
python feature_engineering.py
```

### Phase 4: Model Training
Train the model using:
```bash
python model_training.py
```

### Phase 5: Model Evaluation
Evaluate the model with:
```bash
python model_evaluation.py
```

### Phase 6: Model Deployment
Deploy the model by running:
```bash
python model_deployment.py
```

## MLflow Setup
1. Install MLflow:
   ```bash
   pip install mlflow
   ```
2. Start the MLflow server:
   ```bash
   mlflow ui
   ```
3. Navigate to `http://127.0.0.1:5000` to view the dashboard.

## Troubleshooting
- If you encounter issues, ensure all dependencies are correctly installed and compatible with your Python version.
- For common errors, refer to the [FAQs](https://github.com/SouhailBourhim/Intelligent-Handover-Prediction-LTE/wiki/FAQs) section.
