"""
Phase 3 — Modeling

Three model tiers:
  1. Logistic Regression  (linear baseline)
  2. XGBoost              (tree-based strong baseline)
  3. LSTM                 (sequence model — captures temporal dynamics)

All models predict handover_soon (binary classification).
Class imbalance handled via scale_pos_weight (XGB) and class_weight (LR).
LSTM builds fixed-length sequences per UE, sorted by timestamp.
"""

import json
import math
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"

LABEL = "handover_soon"
SEQ_LEN = 10  # LSTM look-back window (steps)


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

    # Only keep feature cols that survived the pipeline
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


# ── Model 3: LSTM ──────────────────────────────────────────────────────────────

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


class SequenceDataset(torch.utils.data.Dataset):
    """
    Builds fixed-length sequences [t-SEQ_LEN+1 … t] per UE, in time order.
    Only rows where a full look-back window exists are included.
    """
    def __init__(self, df: pd.DataFrame, feat_cols: list[str], seq_len: int = SEQ_LEN):
        self.sequences: list[np.ndarray] = []
        self.labels: list[int] = []

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


def train_lstm(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    feat_cols: list[str],
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
) -> LSTMClassifier:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  LSTM device: {device}")

    train_ds = SequenceDataset(train_df, feat_cols)
    val_ds   = SequenceDataset(val_df,   feat_cols)

    # Weighted sampler to address class imbalance
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

    model = LSTMClassifier(input_size=len(feat_cols)).to(device)

    # Pos weight for BCEWithLogitsLoss
    pw = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )

    best_f1 = 0.0
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

        # Validate
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
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stop at epoch {epoch} (best val F1={best_f1:.4f})")
                break

    if best_state:
        model.load_state_dict(best_state)

    return model


# ── Save / Load helpers ────────────────────────────────────────────────────────

def save_sklearn(model, name: str):
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"  Saved {name} → {path}")


def save_lstm(model: LSTMClassifier, name: str = "lstm"):
    path = MODELS_DIR / f"{name}.pt"
    torch.save(model.state_dict(), path)
    print(f"  Saved LSTM → {path}")


def load_lstm(feat_cols: list[str], name: str = "lstm") -> LSTMClassifier:
    model = LSTMClassifier(input_size=len(feat_cols))
    model.load_state_dict(torch.load(MODELS_DIR / f"{name}.pt", map_location="cpu"))
    model.eval()
    return model


# ── Train all ──────────────────────────────────────────────────────────────────

def run_training():
    print("Phase 3 — Model Training")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test, feat_cols, _ = load_splits()

    X_train, y_train = xy(train, feat_cols)
    X_val,   y_val   = xy(val,   feat_cols)

    print(f"  Features: {len(feat_cols)} | Train: {len(X_train)} | Val: {len(X_val)}")

    print("\n  [1/3] Logistic Regression...")
    lr_model = train_logistic(X_train, y_train)
    save_sklearn(lr_model, "logistic_regression")

    print("\n  [2/3] Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    save_sklearn(rf_model, "random_forest")

    print("\n  [3/3] LSTM...")
    lstm_model = train_lstm(train, val, feat_cols)
    save_lstm(lstm_model)

    print("\nPhase 3 complete.\n")
    return lr_model, rf_model, lstm_model, feat_cols


if __name__ == "__main__":
    run_training()
