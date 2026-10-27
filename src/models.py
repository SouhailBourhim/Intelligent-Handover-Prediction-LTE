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
Class imbalance handled via scale_pos_weight (XGB), class_weight (LR/RF and the
stacking meta-model) and a WeightedRandomSampler (LSTM/GRU, no extra loss
weight).  LSTM/GRU build fixed-length sequences per UE, sorted by timestamp.

After training, every model (plus the rule-based A3 baseline) is scored on the
validation rows that have a full SEQ_LEN history.  Its F1-optimal threshold
and validation metrics go to models/thresholds.json and MLflow; the champion
is selected from these validation metrics, never from the test set.

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
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
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
MLF_TRACKING    = f"sqlite:///{ROOT / 'mlflow.db'}"
MLF_RUN_IDS     = MODELS_DIR / "mlflow_run_ids.json"
THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"
SEED            = 42

# Display name ↔ file stem of every model the pipeline produces
MODEL_FILES = {
    "Logistic Regression": "logistic_regression",
    "Random Forest":       "random_forest",
    "XGBoost":             "xgboost",
    "LSTM":                "lstm",
    "GRU":                 "gru",
    "Stacking Ensemble":   "stacking_ensemble",
    "Rule (A3 gap)":       "rule_a3_gap",
}


def seed_everything(seed: int = SEED):
    """Make numpy and PyTorch training reproducible."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


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


def tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Decision threshold in [0.05, 0.95] that maximises F1."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    f1s  = [f1_score(y_true, y_prob >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def val_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    thr = tune_threshold(y_true, y_prob)
    return {
        "threshold":    thr,
        "val_f1_tuned": round(f1_score(y_true, y_prob >= thr, zero_division=0), 4),
        "val_f1_0.5":   round(f1_score(y_true, y_prob >= 0.5, zero_division=0), 4),
        "val_roc_auc":  round(roc_auc_score(y_true, y_prob), 4),
        "val_pr_auc":   round(average_precision_score(y_true, y_prob), 4),
    }


class RuleA3Gap:
    """
    Rule-based baseline with no learning: the probability of an imminent
    handover is a logistic function of the L3 neighbour − serving gap, i.e.
    the quantity the A3 trigger itself compares with its 3 dB offset.

    It takes the same scaled feature matrix as the ML models and undoes the
    scaling of the two L3 columns, so it plugs into every evaluation path.
    """

    def __init__(self, scaler, feat_cols: list[str], offset_db: float = 3.0,
                 scale_db: float = 2.0):
        self.i_srv = feat_cols.index("l3_rsrp_serving")
        self.i_nb  = feat_cols.index("l3_rsrp_neighbor")
        self.mean  = scaler.mean_[[self.i_srv, self.i_nb]]
        self.std   = scaler.scale_[[self.i_srv, self.i_nb]]
        self.offset_db, self.scale_db = offset_db, scale_db

    def predict_proba(self, X) -> np.ndarray:
        X   = np.asarray(X, dtype=float)
        srv = X[:, self.i_srv] * self.std[0] + self.mean[0]
        nb  = X[:, self.i_nb]  * self.std[1] + self.mean[1]
        p   = 1.0 / (1.0 + np.exp(-((nb - srv) - self.offset_db) / self.scale_db))
        return np.column_stack([1.0 - p, p])


# ── Model 1: Logistic Regression ───────────────────────────────────────────────

def train_logistic(X_train, y_train):
    w = pos_weight(y_train)
    # class_weight rescales the per-sample loss: minority class (handover)
    # gets w× more weight so the linear boundary isn't pushed to always
    # predict the majority (no-handover) class.
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
    # class_weight propagates into the impurity calculation for each split,
    # giving positive (handover) samples more influence on tree structure
    # without oversampling them in memory.
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
      • scale_pos_weight  — handles class imbalance (neg/pos ratio)
      • early_stopping    — stops after 30 rounds without val improvement
    XGBoost 3.x prefers float32 input; cast explicitly to avoid segfaults.
    """
    # Cast to float32 — XGBoost 3.x crashes on float64 with large datasets
    X_tr = np.asarray(X_train, dtype=np.float32)
    X_vl = np.asarray(X_val,   dtype=np.float32)
    y_tr = np.asarray(y_train, dtype=np.int32)
    y_vl = np.asarray(y_val,   dtype=np.int32)

    # scale_pos_weight is XGBoost's equivalent of class_weight: it multiplies
    # the gradient of positive samples by neg/pos, steering the boosting
    # toward fewer missed handovers rather than overall accuracy.
    spw = pos_weight(y_tr)
    model = XGBClassifier(
        n_estimators=1500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="aucpr",          # early-stop on PR-AUC (rare positives)
        early_stopping_rounds=30,
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

    seed_everything()
    labels_arr = np.array(train_ds.labels)
    n_pos = labels_arr.sum()
    n_neg = len(labels_arr) - n_pos
    weights = np.where(labels_arr == 1, n_neg / max(n_pos, 1), 1.0)
    # The WeightedRandomSampler balances mini-batches (≈ 50 % positives), the
    # PyTorch counterpart of class_weight.  The loss is NOT re-weighted on top:
    # doing both would weight positives (n_neg/n_pos)² ≈ 150× and push every
    # probability up (v3 LSTM/GRU precision ≈ 0.33 came from exactly that).
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float32),
        num_samples=len(train_ds),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    best_ap    = 0.0
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
        all_probs, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                all_probs.extend(torch.sigmoid(model(xb.to(device))).cpu().numpy())
                all_labels.extend(yb.numpy().astype(int))

        # Checkpoints are selected on validation PR-AUC (threshold-free): with
        # balanced mini-batches the outputs are not calibrated to the true 6 %
        # prior, so F1 at a fixed 0.5 cut-off is a noisy selection signal.
        val_ap = average_precision_score(all_labels, all_probs)
        scheduler.step(val_ap)

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d} | val PR-AUC: {val_ap:.4f}")

        if val_ap > best_ap:
            best_ap    = val_ap
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stop at epoch {epoch} (best val PR-AUC={best_ap:.4f})")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_ap


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


def score_sequence_model(net: nn.Module, df: pd.DataFrame, feat_cols: list[str],
                         seq_len: int = SEQ_LEN) -> pd.Series:
    """
    One probability per row of `df`, returned as a Series on df's index.

    Windows are built **per UE** in timestamp order, so a window never mixes
    two UEs.  The first seq_len−1 rows of each UE (no full history yet) are
    left-padded with that UE's own first row.  Row order of `df` is irrelevant.
    """
    net = net.to("cpu").eval()
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for _, grp in df.groupby("ue_id", sort=False):
        grp = grp.sort_values("timestamp")
        X   = grp[feat_cols].fillna(0).values.astype(np.float32)
        Xp  = np.vstack([np.repeat(X[:1], seq_len - 1, axis=0), X])
        win = np.lib.stride_tricks.sliding_window_view(Xp, seq_len, axis=0)
        win = np.ascontiguousarray(win.transpose(0, 2, 1))    # (n, seq_len, n_feat)
        with torch.no_grad():
            p = torch.sigmoid(net(torch.from_numpy(win))).numpy()
        out.loc[grp.index] = p
    return out


# ── Model 6: Stacking Ensemble ─────────────────────────────────────────────────

def _get_seq_probs(seq_model: nn.Module, df: pd.DataFrame,
                   feat_cols: list[str]) -> tuple[np.ndarray, list[int]]:
    """Return (probs, row_indices) on the rows that have a full SEQ_LEN window."""
    row_indices = _build_seq_row_indices(df, seq_len=SEQ_LEN)
    probs = score_sequence_model(seq_model, df, feat_cols).loc[row_indices].values
    return probs, row_indices


def stacking_meta_features(df: pd.DataFrame, feat_cols: list[str],
                           xgb_model, rf_model, lstm_probs: np.ndarray) -> np.ndarray:
    """[p_xgb, p_rf, p_lstm] for the rows of df (already aligned with lstm_probs)."""
    X = df[feat_cols].fillna(0).values
    return np.column_stack([
        xgb_model.predict_proba(X.astype(np.float32))[:, 1],
        rf_model.predict_proba(X)[:, 1],
        lstm_probs,
    ])


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
    val_sub = val_df.loc[row_indices]
    meta_X  = stacking_meta_features(val_sub, feat_cols, xgb_model, rf_model, lstm_probs)
    y_val   = val_sub[LABEL].values

    # class_weight='balanced': at a ~7 % positive rate an unweighted meta-model
    # puts its 0.5 boundary far too high (v3 recall was 0.40).
    meta_lr = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced",
                                 random_state=SEED)

    # The meta-model is fit on validation, so its *validation* score must come
    # from out-of-fold predictions (5 contiguous time blocks) to be comparable
    # with the other models when the champion is selected.
    oof = cross_val_predict(meta_lr, meta_X, y_val, cv=KFold(5, shuffle=False),
                            method="predict_proba")[:, 1]
    meta_lr.fit(meta_X, y_val)

    val_f1 = f1_score(y_val, oof >= 0.5, zero_division=0)
    print(f"    Stacking meta-LR out-of-fold val F1: {val_f1:.4f}")
    return meta_lr, val_f1, pd.Series(oof, index=row_indices)


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
    seed_everything()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test, feat_cols, meta = load_splits()

    X_train, y_train = xy(train, feat_cols)
    X_val,   y_val   = xy(val,   feat_cols)

    print(f"  Features: {len(feat_cols)} | Train: {len(X_train)} | Val: {len(X_val)}")

    _mlf_setup()
    run_id_map: dict[str, str] = {}

    # ── 1. Logistic Regression ─────────────────────────────────────────────────
    print("\n  [1/7] Logistic Regression...")
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
    print("\n  [2/7] Random Forest...")
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
    print("\n  [3/7] XGBoost...")
    xgb_model  = train_xgboost(X_train, y_train, X_val, y_val)
    xgb_val_f1 = f1_score(y_val, xgb_model.predict(X_val), zero_division=0)
    print(f"    val F1: {xgb_val_f1:.4f}")
    save_sklearn(xgb_model, "xgboost")

    try:
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name="xgboost") as run:
                mlflow.log_params({
                    "n_estimators": 1500, "max_depth": 6,
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
    print("\n  [4/7] LSTM...")
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
                mlflow.log_metric("val_pr_auc_checkpoint", lstm_val_f1)
                run_id_map["LSTM"] = run.info.run_id
    except Exception:
        pass

    # ── 5. GRU ─────────────────────────────────────────────────────────────────
    print("\n  [5/7] GRU...")
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
                mlflow.log_metric("val_pr_auc_checkpoint", gru_val_f1)
                run_id_map["GRU"] = run.info.run_id
    except Exception:
        pass

    # ── 6. Stacking Ensemble ───────────────────────────────────────────────────
    print("\n  [6/7] Stacking Ensemble...")
    stacking_model, stack_val_f1, stack_oof = train_stacking(
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
                    "meta_class_weight": "balanced",
                })
                mlflow.log_metric("val_f1", stack_val_f1)
                mlflow.sklearn.log_model(stacking_model, "model")
                run_id_map["Stacking Ensemble"] = run.info.run_id
    except Exception:
        pass

    # ── 7. Rule baseline (no learning) ─────────────────────────────────────────
    print("\n  [7/7] Rule baseline (A3 gap)...")
    scaler    = joblib.load(MODELS_DIR / "scaler.pkl")
    rule      = RuleA3Gap(scaler, feat_cols)
    save_sklearn(rule, "rule_a3_gap")
    try:
        if MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name="rule_a3_gap") as run:
                mlflow.log_params({"offset_db": rule.offset_db, "scale_db": rule.scale_db})
                run_id_map["Rule (A3 gap)"] = run.info.run_id
    except Exception:
        pass

    # ── Validation scoring: thresholds + selection metrics ─────────────────────
    # All models are scored on the same validation rows (full SEQ_LEN history).
    rows  = _build_seq_row_indices(val, seq_len=SEQ_LEN)
    v     = val.loc[rows]
    X_v   = v[feat_cols].fillna(0).values
    y_v   = v[LABEL].values
    lstm_v = score_sequence_model(lstm_model, val, feat_cols).loc[rows].values
    val_probs = {
        "Logistic Regression": lr_model.predict_proba(X_v)[:, 1],
        "Random Forest":       rf_model.predict_proba(X_v)[:, 1],
        "XGBoost":             xgb_model.predict_proba(X_v.astype(np.float32))[:, 1],
        "LSTM":                lstm_v,
        "GRU":                 score_sequence_model(gru_model, val, feat_cols).loc[rows].values,
        "Stacking Ensemble":   stack_oof.loc[rows].values,     # out-of-fold
        "Rule (A3 gap)":       rule.predict_proba(X_v)[:, 1],
    }
    summary = {name: val_metrics(y_v, p) for name, p in val_probs.items()}
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n  Validation (same rows for every model):")
    for name, m in summary.items():
        print(f"    {name:<20} thr={m['threshold']:.2f}  F1={m['val_f1_tuned']:.4f}  "
              f"AUC={m['val_roc_auc']:.4f}  PR-AUC={m['val_pr_auc']:.4f}")
        try:
            if MLFLOW_AVAILABLE and name in run_id_map:
                with mlflow.start_run(run_id=run_id_map[name]):
                    mlflow.log_metrics(m)
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
