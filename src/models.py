"""
Phase 3 — Modeling

Six model tiers:
  1. Logistic Regression    (linear baseline)
  2. Random Forest          (tree-based ensemble)
  3. XGBoost                (gradient boosting, early stopping)
  4. LSTM                   (sequence model — captures temporal dynamics)
  5. GRU                    (sequence model — lighter recurrent alternative)
  6. Stacking Ensemble      (meta-LR on val-set probs from XGB + RF + LSTM)

All models predict handover_soon (binary classification).
Class imbalance handled via scale_pos_weight (XGB) and class_weight (LR/RF).
LSTM/GRU build fixed-length sequences per UE, sorted by timestamp.

MLflow tracking is wrapped in try/except — the pipeline runs without it.
"""

import json
import math
import os
import numpy as np
import pandas as pd
import joblib

# Set OMP_NUM_THREADS before importing torch to prevent a segfault caused by
# PyTorch and XGBoost both trying to initialize OpenMP on macOS.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.nn as nn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

# ── MLflow (optional) ──────────────────────────────────────────────────────────
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"

LABEL   = "handover_soon"
SEQ_LEN = 10   # LSTM/GRU look-back window (steps)

MLF_EXP_NAME    = "lte_handover_prediction"
MLF_TRACKING    = str(ROOT / "mlruns")
MLF_RUN_IDS     = MODELS_DIR / "mlflow_run_ids.json"


# ── MLflow helpers ─────────────────────────────────────────────────────────────

def _mlf_setup():
    """Configure MLflow experiment once. No-op if unavailable."""
    if not MLFLOW_AVAILABLE:
        return
    try:
        mlflow.set_tracking_uri(MLF_TRACKING)
        mlflow.set_experiment(MLF_EXP_NAME)
    except Exception:
        pass


def _mlf_save_run_id(name: str, run_id: str):
    """Persist run_id so evaluate.py can resume the same run."""
    try:
        data: dict = {}
        if MLF_RUN_IDS.exists():
            with open(MLF_RUN_IDS) as f:
                data = json.load(f)
        data[name] = run_id
        with open(MLF_RUN_IDS, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_splits(feat_cols: list[str] | None = None):
    """Load train/val/test and derive feature columns from meta if not given."""
    meta_path = PROCESSED_DIR / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    feat_cols = feat_cols or meta["feature_cols"]

    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val   = pd.read_csv(PROCESSED_DIR / "val.csv")
    test  = pd.read_csv(PROCESSED_DIR / "test.csv")

    feat_cols = [c for c in feat_cols if c in train.columns]
    return train, val, test, feat_cols, meta


def xy(df: pd.DataFrame, feat_cols: list[str]):
    X = df[feat_cols].fillna(0).values
    y = df[LABEL].values
    return X, y


def pos_weight(y_train: np.ndarray) -> float:
    """Scale factor for imbalanced binary labels (neg/pos ratio)."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return n_neg / max(n_pos, 1)


def _build_seq_row_indices(df: pd.DataFrame, seq_len: int = SEQ_LEN) -> list[int]:
    """
    Return the original DataFrame index of the *last* row of every valid
    sequence window built by SequenceDataset.  Used to align LSTM/GRU
    outputs with sklearn rows when constructing stacking meta-features.
    """
    indices = []
    for _, grp in df.sort_values("timestamp").groupby("ue_id"):
        grp_idxs = list(grp.index)
        for i in range(seq_len - 1, len(grp_idxs)):
            indices.append(grp_idxs[i])
    return indices


# ── Model 1: Logistic Regression ───────────────────────────────────────────────

def train_logistic(X_train, y_train):
    w = pos_weight(y_train)
    model = LogisticRegression(
        max_iter=2000,
        C=0.5,
        class_weight={0: 1.0, 1: w},
        solver="saga",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


# ── Model 2: Random Forest ─────────────────────────────────────────────────────

def train_random_forest(X_train, y_train):
    w = pos_weight(y_train)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight={0: 1.0, 1: w},
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ── Model 3: XGBoost ───────────────────────────────────────────────────────────

def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Gradient boosted trees with:
      • scale_pos_weight  — handles class imbalance
      • early_stopping    — stops after 20 rounds without val improvement
    XGBoost 3.x prefers float32 input; cast explicitly to avoid segfaults.
    """
    # Cast to float32 — XGBoost 3.x crashes on float64 with large datasets
    X_tr = np.asarray(X_train, dtype=np.float32)
    X_vl = np.asarray(X_val,   dtype=np.float32)
    y_tr = np.asarray(y_train, dtype=np.int32)
    y_vl = np.asarray(y_val,   dtype=np.int32)

    spw = pos_weight(y_tr)
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        verbose=False,
    )
    print(f"    XGBoost best iteration: {model.best_iteration}")
    return model


# ── Sequence models (shared dataset) ──────────────────────────────────────────

class SequenceDataset(torch.utils.data.Dataset):
    """
    Builds fixed-length sequences [t-SEQ_LEN+1 … t] per UE, in time order.
    Only rows where a full look-back window exists are included.
    """
    def __init__(self, df: pd.DataFrame, feat_cols: list[str], seq_len: int = SEQ_LEN):
        self.sequences: list[np.ndarray] = []
        self.labels:    list[int]        = []

        for _, grp in df.sort_values("timestamp").groupby("ue_id"):
            X = grp[feat_cols].fillna(0).values.astype(np.float32)
            y = grp[LABEL].values.astype(np.float32)
            for i in range(seq_len - 1, len(X)):
                self.sequences.append(X[i - seq_len + 1 : i + 1])
                self.labels.append(y[i])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx],    dtype=torch.float32),
        )


# ── Model 4: LSTM ──────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.classifier(h_n[-1]).squeeze(-1)


# ── Model 5: GRU ───────────────────────────────────────────────────────────────

class GRUClassifier(nn.Module):
    """GRU mirror of LSTMClassifier — same hyper-parameters, lighter state."""
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        return self.classifier(h_n[-1]).squeeze(-1)


def _train_sequence_model(
    model:      nn.Module,
    train_df:   pd.DataFrame,
    val_df:     pd.DataFrame,
    feat_cols:  list[str],
    epochs:     int   = 30,
    batch_size: int   = 256,
    lr:         float = 1e-3,
    patience:   int   = 5,
) -> nn.Module:
    """Shared training loop for LSTM and GRU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SequenceDataset(train_df, feat_cols)
    val_ds   = SequenceDataset(val_df,   feat_cols)

    labels_arr = np.array(train_ds.labels)
    n_pos = labels_arr.sum()
    n_neg = len(labels_arr) - n_pos
    weights = np.where(labels_arr == 1, n_neg / max(n_pos, 1), 1.0)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float32),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )

    model = model.to(device)
    pw = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    best_f1    = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb.to(device))
                probs  = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend((probs > 0.5).astype(int))
                all_labels.extend(yb.numpy().astype(int))

        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        scheduler.step(val_f1)

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d} | val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stop at epoch {epoch} (best val F1={best_f1:.4f})")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_f1


def train_lstm(train_df, val_df, feat_cols, epochs=30, batch_size=256,
               lr=1e-3, patience=5):
    model = LSTMClassifier(input_size=len(feat_cols))
    model, best_f1 = _train_sequence_model(
        model, train_df, val_df, feat_cols,
        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
    )
    return model, best_f1


def train_gru(train_df, val_df, feat_cols, epochs=30, batch_size=256,
              lr=1e-3, patience=5):
    model = GRUClassifier(input_size=len(feat_cols))
    model, best_f1 = _train_sequence_model(
        model, train_df, val_df, feat_cols,
        epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
    )
    return model, best_f1


# ── Model 6: Stacking Ensemble ─────────────────────────────────────────────────

def _get_seq_probs(seq_model: nn.Module, df: pd.DataFrame,
                   feat_cols: list[str]) -> tuple[np.ndarray, list[int]]:
    """Return (probs, aligned_row_indices) for a trained sequence model."""
    device = torch.device("cpu")
    seq_model = seq_model.to(device).eval()

    ds     = SequenceDataset(df, feat_cols, seq_len=SEQ_LEN)
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)

    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            probs.extend(torch.sigmoid(seq_model(xb)).numpy())

    row_indices = _build_seq_row_indices(df, seq_len=SEQ_LEN)
    return np.array(probs), row_indices


def train_stacking(
    val_df:     pd.DataFrame,
    feat_cols:  list[str],
    xgb_model,
    rf_model,
    lstm_model: nn.Module,
) -> LogisticRegression:
    """
    Train a meta Logistic Regression on val-set probabilities from
    XGBoost, Random Forest, and LSTM.

    LSTM covers only rows with a full SEQ_LEN look-back window; the
    sklearn models are restricted to the same subset for alignment.
    """
    lstm_probs, row_indices = _get_seq_probs(lstm_model, val_df, feat_cols)

    # Subset val_df to the rows the LSTM actually covered
    val_sub  = val_df.loc[row_indices].copy()
    X_val_sk = val_sub[feat_cols].fillna(0).values
    y_val    = val_sub[LABEL].values

    xgb_probs = xgb_model.predict_proba(np.asarray(X_val_sk, dtype=np.float32))[:, 1]
    rf_probs  = rf_model.predict_proba(X_val_sk)[:, 1]

    meta_X = np.column_stack([xgb_probs, rf_probs, lstm_probs])

    meta_lr = LogisticRegression(C=1.0, max_iter=500, random_state=42)
    meta_lr.fit(meta_X, y_val)

    val_f1 = f1_score(y_val, meta_lr.predict(meta_X), zero_division=0)
    print(f"    Stacking meta-LR val F1: {val_f1:.4f}")
    return meta_lr, val_f1


# ── Save / Load helpers ────────────────────────────────────────────────────────

def save_sklearn(model, name: str):
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"  Saved {name} → {path}")


def save_torch(model: nn.Module, name: str):
    path = MODELS_DIR / f"{name}.pt"
    torch.save(model.state_dict(), path)
    print(f"  Saved {name} → {path}")


# Keep backwards-compatible aliases
def save_lstm(model: LSTMClassifier, name: str = "lstm"):
    save_torch(model, name)


def load_lstm(feat_cols: list[str], name: str = "lstm") -> LSTMClassifier:
    model = LSTMClassifier(input_size=len(feat_cols))
    model.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt", map_location="cpu"))
    model.eval()
    return model


def load_gru(feat_cols: list[str], name: str = "gru") -> GRUClassifier:
    model = GRUClassifier(input_size=len(feat_cols))
    model.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt", map_location="cpu"))
    model.eval()
    return model


# ── Train all ──────────────────────────────────────────────────────────────────

def run_training():
    print("Phase 3 — Model Training")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test, feat_cols, meta = load_splits()

    X_train, y_train = xy(train, feat_cols)
    X_val,   y_val   = xy(val,   feat_cols)

    print(f"  Features: {len(feat_cols)} | Train: {len(X_train)} | Val: {len(X_val)}")

    _mlf_setup()
    run_id_map: dict[str, str] = {}

    # ── 1. Logistic Regression ─────────────────────────────────────────────────
    print("\n  [1/6] Logistic Regression...")
    lr_model  = train_logistic(X_train, y_train)
    lr_val_f1 = f1_score(y_val, lr_model.predict(X_val), zero_division=0)
    print(f"    val F1: {lr_val_f1:.4f}")
    save_sklearn(lr_model, "logistic_regression")

    try:
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name="logistic_regression") as run:
                mlflow.log_params({"C": 0.5, "solver": "saga",
                                   "class_weight": "balanced"})
                mlflow.log_metric("val_f1", lr_val_f1)
                mlflow.sklearn.log_model(lr_model, "model")
                run_id_map["Logistic Regression"] = run.info.run_id
    except Exception:
        pass

    # ── 2. Random Forest ───────────────────────────────────────────────────────
    print("\n  [2/6] Random Forest...")
    rf_model  = train_random_forest(X_train, y_train)
    rf_val_f1 = f1_score(y_val, rf_model.predict(X_val), zero_division=0)
    print(f"    val F1: {rf_val_f1:.4f}")
    save_sklearn(rf_model, "random_forest")

    try:
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name="random_forest") as run:
                mlflow.log_params({"n_estimators": 300, "max_depth": 12,
                                   "min_samples_leaf": 5})
                mlflow.log_metric("val_f1", rf_val_f1)
                mlflow.sklearn.log_model(rf_model, "model")
                run_id_map["Random Forest"] = run.info.run_id
    except Exception:
        pass

    # ── 3. XGBoost ─────────────────────────────────────────────────────────────
    print("\n  [3/6] XGBoost...")
    xgb_model  = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_val_f1 = f1_score(y_val, xgb_model.predict(X_val), zero_division=0)
    print(f"    val F1: {xgb_val_f1:.4f}")
    save_sklearn(xgb_model, "xgboost")

    try:
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name="xgboost") as run:
                mlflow.log_params({
                    "n_estimators": 400, "max_depth": 6,
                    "learning_rate": 0.05, "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "best_iteration": int(xgb_model.best_iteration),
                })
                mlflow.log_metric("val_f1", xgb_val_f1)
                mlflow.sklearn.log_model(xgb_model, "model")
                run_id_map["XGBoost"] = run.info.run_id
    except Exception:
        pass

    # ── 4. LSTM ────────────────────────────────────────────────────────────────
    print("\n  [4/6] LSTM...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    device: {device}")
    lstm_model, lstm_val_f1 = train_lstm(train, val, feat_cols)
    save_torch(lstm_model, "lstm")

    try:
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name="lstm") as run:
                mlflow.log_params({
                    "seq_len": SEQ_LEN, "hidden_size": 64,
                    "num_layers": 2, "dropout": 0.3,
                    "epochs": 30, "batch_size": 256, "lr": 1e-3,
                })
                mlflow.log_metric("val_f1", lstm_val_f1)
                run_id_map["LSTM"] = run.info.run_id
    except Exception:
        pass

    # ── 5. GRU ─────────────────────────────────────────────────────────────────
    print("\n  [5/6] GRU...")
    gru_model, gru_val_f1 = train_gru(train, val, feat_cols)
    save_torch(gru_model, "gru")

    try:
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name="gru") as run:
                mlflow.log_params({
                    "seq_len": SEQ_LEN, "hidden_size": 64,
                    "num_layers": 2, "dropout": 0.3,
                    "epochs": 30, "batch_size": 256, "lr": 1e-3,
                })
                mlflow.log_metric("val_f1", gru_val_f1)
                run_id_map["GRU"] = run.info.run_id
    except Exception:
        pass

    # ── 6. Stacking Ensemble ───────────────────────────────────────────────────
    print("\n  [6/6] Stacking Ensemble...")
    stacking_model, stack_val_f1 = train_stacking(
        val, feat_cols, xgb_model, rf_model, lstm_model
    )
    save_sklearn(stacking_model, "stacking_ensemble")

    try:
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name="stacking_ensemble") as run:
                mlflow.log_params({
                    "base_models": "xgboost,random_forest,lstm",
                    "meta_model": "logistic_regression",
                    "meta_C": 1.0,
                })
                mlflow.log_metric("val_f1", stack_val_f1)
                mlflow.sklearn.log_model(stacking_model, "model")
                run_id_map["Stacking Ensemble"] = run.info.run_id
    except Exception:
        pass

    # Persist run IDs for evaluate.py
    _mlf_save_run_id("__all__", "")  # ensure file exists
    for name, rid in run_id_map.items():
        _mlf_save_run_id(name, rid)

    print("\nPhase 3 complete.\n")
    return lr_model, rf_model, xgb_model, lstm_model, gru_model, stacking_model, feat_cols


if __name__ == "__main__":
    run_training()
