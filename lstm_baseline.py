"""
lstm_baseline.py
================
Publication-clean LSTM baseline for comparison against the learned DBN.

Fixes included:
  - Chronological train/test split
  - Feature scaler fit only on training data
  - Target scaler fit only on training data
  - No future leakage
  - Reproducible random seeds
  - LSTM regression on scaled throughput
  - MAE on original throughput scale
  - Discretized accuracy using train-fitted bins
  - Persistence baseline for sanity check

Usage:
    python lstm_baseline.py
"""

import ast
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.metrics import mean_absolute_error


# ============================================================
# CONFIG — must match DBN sweep
# ============================================================

CSV_PATH = "share/metrics/dbn_wide_20260310_160246_seed161312.csv"

TARGETS = ["throughput_1", "throughput_2", "throughput_3"]

MODELING_GRANULARITY_SEC = 30

TRAIN_FRAC = 0.8

SEQUENCE_LENGTH = 5

N_BINS = 4

EPOCHS = 100

HIDDEN_SIZE = 64

LEARNING_RATE = 1e-3

EXCLUDE_OTHER_THROUGHPUTS = True

RANDOM_STATE = 0


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# LOAD + CLEAN + AGGREGATE
# ============================================================

def load_and_clean(path):
    df = pd.read_csv(path)

    if "s_config" in df.columns:
        def parse(val):
            try:
                d = ast.literal_eval(val)
                return d if isinstance(d, dict) else {}
            except Exception:
                return {}

        cfg = df["s_config"].apply(parse).apply(pd.Series)
        cfg = cfg.rename(columns=lambda c: f"s_config_{c}")

        df = pd.concat([df.drop(columns=["s_config"]), cfg], axis=1)

    for col in list(df.columns):
        if "time" in col.lower():
            df.drop(columns=[col], inplace=True)

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    df = df.select_dtypes(include=[np.number])
    df = df.dropna().reset_index(drop=True)

    return df


def aggregate(df):
    group_ids = np.arange(len(df)) // MODELING_GRANULARITY_SEC
    return df.groupby(group_ids).mean().reset_index(drop=True)


# ============================================================
# MODEL
# ============================================================

class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.linear(last_hidden)


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_feature_cols(df, target):
    return [
        c for c in df.columns
        if c != target
        and not (
            EXCLUDE_OTHER_THROUGHPUTS
            and c.startswith("throughput_")
        )
    ]


# ============================================================
# SEQUENCE PREPARATION WITHOUT LEAKAGE
# ============================================================

def make_sequences(X, y, seq_len):
    X_seqs = []
    y_seqs = []

    for i in range(seq_len, len(X)):
        X_seqs.append(X[i - seq_len:i])
        y_seqs.append(y[i])

    return (
        np.array(X_seqs, dtype=np.float32),
        np.array(y_seqs, dtype=np.float32)
    )


def prepare_data(df, target):
    feature_cols = get_feature_cols(df, target)

    X_raw = df[feature_cols].to_numpy(dtype=float)
    y_raw = df[target].to_numpy(dtype=float)

    split_raw = int(TRAIN_FRAC * len(df))

    if split_raw <= SEQUENCE_LENGTH:
        raise ValueError(
            "Training set is too small for the chosen SEQUENCE_LENGTH."
        )

    X_train_raw = X_raw[:split_raw]
    X_test_raw = X_raw[split_raw - SEQUENCE_LENGTH:]

    y_train_raw = y_raw[:split_raw]
    y_test_raw = y_raw[split_raw - SEQUENCE_LENGTH:]

    # Fit scalers only on training data
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaler.fit(X_train_raw)
    y_scaler.fit(y_train_raw.reshape(-1, 1))

    X_train_scaled = X_scaler.transform(X_train_raw)
    X_test_scaled = X_scaler.transform(X_test_raw)

    y_train_scaled = y_scaler.transform(
        y_train_raw.reshape(-1, 1)
    ).flatten()

    y_test_scaled = y_scaler.transform(
        y_test_raw.reshape(-1, 1)
    ).flatten()

    X_train_seq, y_train_seq = make_sequences(
        X_train_scaled,
        y_train_scaled,
        SEQUENCE_LENGTH
    )

    X_test_seq, y_test_seq_scaled = make_sequences(
        X_test_scaled,
        y_test_scaled,
        SEQUENCE_LENGTH
    )

    # Original-scale y for evaluation
    _, y_test_seq_raw = make_sequences(
        X_test_raw,
        y_test_raw,
        SEQUENCE_LENGTH
    )

    _, y_train_seq_raw = make_sequences(
        X_train_raw,
        y_train_raw,
        SEQUENCE_LENGTH
    )

    return {
        "X_train": X_train_seq,
        "y_train_scaled": y_train_seq,
        "y_train_raw": y_train_seq_raw,
        "X_test": X_test_seq,
        "y_test_scaled": y_test_seq_scaled,
        "y_test_raw": y_test_seq_raw,
        "y_scaler": y_scaler,
        "feature_cols": feature_cols,
        "split_raw": split_raw,
        "y_raw": y_raw,
    }


# ============================================================
# BASELINES
# ============================================================

def persistence_baseline(df, target):
    """
    Predict y(t) = y(t-1) on the same test region.
    """
    y = df[target].to_numpy(dtype=float)

    split_raw = int(TRAIN_FRAC * len(df))

    y_test = y[split_raw:]
    y_pred = y[split_raw - 1:-1]

    mae = float(mean_absolute_error(y_test, y_pred))

    kbd = KBinsDiscretizer(
        n_bins=N_BINS,
        encode="ordinal",
        strategy="uniform"
    )

    kbd.fit(y[:split_raw].reshape(-1, 1))

    y_test_disc = kbd.transform(
        y_test.reshape(-1, 1)
    ).astype(int).flatten()

    y_pred_disc = kbd.transform(
        y_pred.reshape(-1, 1)
    ).astype(int).flatten()

    accuracy = float(np.mean(y_pred_disc == y_test_disc))

    return mae, accuracy


# ============================================================
# TRAIN + EVALUATE LSTM
# ============================================================

def run_lstm(df, target):
    data = prepare_data(df, target)

    X_train = torch.tensor(data["X_train"], dtype=torch.float32)
    y_train = torch.tensor(
        data["y_train_scaled"],
        dtype=torch.float32
    ).unsqueeze(1)

    X_test = torch.tensor(data["X_test"], dtype=torch.float32)

    y_test_raw = data["y_test_raw"]
    y_scaler = data["y_scaler"]

    model = LSTMPredictor(
        input_size=X_train.shape[2],
        hidden_size=HIDDEN_SIZE,
        output_size=1
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.MSELoss()

    model.train()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        pred = model(X_train)

        loss = criterion(pred, y_train)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

    model.eval()

    with torch.no_grad():
        y_pred_scaled = model(X_test).squeeze().numpy()

    y_pred_raw = y_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).flatten()

    mae = float(mean_absolute_error(y_test_raw, y_pred_raw))

    # Discretized accuracy
    split_raw = data["split_raw"]
    y_raw = data["y_raw"]

    kbd = KBinsDiscretizer(
        n_bins=N_BINS,
        encode="ordinal",
        strategy="uniform"
    )

    kbd.fit(y_raw[:split_raw].reshape(-1, 1))

    y_test_disc = kbd.transform(
        y_test_raw.reshape(-1, 1)
    ).astype(int).flatten()

    y_pred_disc = kbd.transform(
        y_pred_raw.reshape(-1, 1)
    ).astype(int).flatten()

    accuracy = float(np.mean(y_pred_disc == y_test_disc))

    diagnostics = {
        "y_test_min": float(np.min(y_test_raw)),
        "y_test_max": float(np.max(y_test_raw)),
        "y_pred_min": float(np.min(y_pred_raw)),
        "y_pred_max": float(np.max(y_pred_raw)),
        "final_train_loss": float(loss.item()),
        "n_train_sequences": len(data["X_train"]),
        "n_test_sequences": len(data["X_test"]),
        "n_features": len(data["feature_cols"]),
    }

    return mae, accuracy, diagnostics


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(RANDOM_STATE)

    print("=" * 60)
    print("LSTM BASELINE")
    print("=" * 60)

    raw = load_and_clean(CSV_PATH)
    raw = aggregate(raw)

    print(f"Rows after aggregation: {len(raw)}")
    print(
        f"Sequence length: {SEQUENCE_LENGTH} steps = "
        f"{SEQUENCE_LENGTH * MODELING_GRANULARITY_SEC} seconds"
    )
    print(f"Train fraction: {TRAIN_FRAC}")
    print(f"Epochs: {EPOCHS}")
    print(f"Hidden size: {HIDDEN_SIZE}")
    print()

    results = []

    for target in TARGETS:
        if target not in raw.columns:
            print(f"[SKIP] {target} not found in data")
            print()
            continue

        print(f"Target: {target}")
        print("-" * 40)

        lstm_mae, lstm_acc, diagnostics = run_lstm(raw, target)
        pers_mae, pers_acc = persistence_baseline(raw, target)

        print(f"  LSTM MAE:              {lstm_mae:.3f}")
        print(f"  LSTM accuracy:         {lstm_acc:.3f} (n_bins={N_BINS})")
        print()
        print(f"  Persistence MAE:       {pers_mae:.3f}")
        print(f"  Persistence accuracy:  {pers_acc:.3f} (n_bins={N_BINS})")
        print()
        print("  Diagnostics:")
        print(f"    Train sequences:     {diagnostics['n_train_sequences']}")
        print(f"    Test sequences:      {diagnostics['n_test_sequences']}")
        print(f"    Features:            {diagnostics['n_features']}")
        print(f"    Final train loss:    {diagnostics['final_train_loss']:.6f}")
        print(
            f"    y_test range:        "
            f"{diagnostics['y_test_min']:.3f} to {diagnostics['y_test_max']:.3f}"
        )
        print(
            f"    y_pred range:        "
            f"{diagnostics['y_pred_min']:.3f} to {diagnostics['y_pred_max']:.3f}"
        )
        print()

        results.append({
            "target": target,
            "lstm_mae": lstm_mae,
            "lstm_accuracy": lstm_acc,
            "persistence_mae": pers_mae,
            "persistence_accuracy": pers_acc,
            **diagnostics,
        })

    if results:
        out = pd.DataFrame(results)
        out.to_csv("lstm_baseline_results.csv", index=False)

        print("=" * 60)
        print("Results saved to lstm_baseline_results.csv")
        print("=" * 60)


if __name__ == "__main__":
    main()