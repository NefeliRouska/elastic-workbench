import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination

from full_dynamic_bn_learn_final import load_and_clean, build_dbn

CSV_PATH = "prom_all_metrics_wide.csv"
TARGET = "throughput"
TRAIN_FRAC = 0.8
MAX_BINS = 5
EPS = 1e-12

SUBSET_EVIDENCE = [
    "throughput",
    "avg_p_latency",
    "cores",
    "data_quality",
    "container_cpu_usage_seconds_total",
    "container_memory_usage_bytes",
    "container_spec_cpu_quota",
]


# ----------------------------
# Train-fitted discretizer
# ----------------------------
def fit_discretizer_fd(train_df: pd.DataFrame, max_bins=5):
    """
    Fit bin edges on TRAIN only.
    Returns dict col -> bin_edges (np.array) for numeric cols to discretize.
    """
    bins = {}
    for c in train_df.columns:
        if not pd.api.types.is_numeric_dtype(train_df[c]):
            continue

        col = train_df[c].dropna()
        if col.nunique() <= max_bins:
            # Treat as already discrete-ish: do NOT bin it (keeps values as states).
            # But then test must be mapped to nearest train value later.
            bins[c] = None
            continue

        vmin, vmax = float(col.min()), float(col.max())
        if vmin == vmax:
            bins[c] = None
            continue

        n = len(col)
        q75, q25 = np.percentile(col, [75, 25])
        iqr = q75 - q25

        if iqr == 0:
            k = min(max_bins, int(col.nunique()))
            k = max(2, k)
        else:
            bw = 2 * iqr / (n ** (1 / 3))
            if bw <= 0:
                k = min(max_bins, int(col.nunique()))
                k = max(2, k)
            else:
                k = int(np.ceil((vmax - vmin) / bw))
                k = max(2, min(k, max_bins))

        edges = np.linspace(vmin, vmax, k + 1)
        bins[c] = edges

    return bins


def transform_with_bins(df: pd.DataFrame, bins: dict):
    """
    Apply TRAIN-fitted bins to any df.
    - If bins[c] is an array: pd.cut using those edges -> integer labels 0..k-1
      and clip out-of-range to nearest bin via include_lowest + clipping.
    - If bins[c] is None: map values to nearest value among TRAIN unique states.
    """
    out = df.copy()

    for c, edges in bins.items():
        if c not in out.columns:
            continue

        if not pd.api.types.is_numeric_dtype(out[c]):
            continue

        col = out[c]

        if edges is None:
            # Map to nearest observed TRAIN state (to avoid unseen states in TEST).
            # This makes state space consistent between train and test.
            train_states = np.array(sorted(pd.unique(col.dropna())))
            # WARNING: we need TRAIN states, not df states. So we store them elsewhere.
            # We'll overwrite in the wrapper below.
            continue

        k = len(edges) - 1
        # Cut; out-of-range becomes NaN, so we clip manually using digitize
        # to guarantee an integer bin for every value.
        x = col.to_numpy(dtype=float)
        idx = np.digitize(x, edges[1:-1], right=False)  # yields 0..k-1
        idx = np.clip(idx, 0, k - 1)
        out[c] = idx.astype(int)

    return out


def transform_with_bins_and_state_maps(train_df: pd.DataFrame, test_df: pd.DataFrame, bins: dict):
    """
    Handle the edges=None case properly: for low-unique columns we map TEST to nearest TRAIN state.
    """
    train_out = train_df.copy()
    test_out = test_df.copy()

    for c, edges in bins.items():
        if c not in train_df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(train_df[c]):
            continue

        if edges is not None:
            # numeric binned case
            k = len(edges) - 1
            for out_df, src_df in [(train_out, train_df), (test_out, test_df)]:
                x = src_df[c].to_numpy(dtype=float)
                idx = np.digitize(x, edges[1:-1], right=False)
                idx = np.clip(idx, 0, k - 1)
                out_df[c] = idx.astype(int)
        else:
            # small-unique case: build mapping from TRAIN unique states
            states = np.array(sorted(pd.unique(train_df[c].dropna())))
            if len(states) == 0:
                continue

            def nearest_state(v):
                # if exact match, keep
                if v in states:
                    return v
                # otherwise map to nearest numeric state
                return states[np.argmin(np.abs(states - v))]

            train_out[c] = train_df[c].apply(lambda v: nearest_state(v) if pd.notna(v) else v)
            test_out[c] = test_df[c].apply(lambda v: nearest_state(v) if pd.notna(v) else v)

            # finally, remap those states to compact ints 0..(m-1)
            state_to_int = {s: i for i, s in enumerate(states)}
            train_out[c] = train_out[c].map(state_to_int).astype(int)
            test_out[c] = test_out[c].map(state_to_int).astype(int)

    return train_out, test_out


# ----------------------------
# Evaluation
# ----------------------------
def get_factor(q, var_name):
    return q[var_name] if isinstance(q, dict) else q


def eval_baseline_persistence(df_test):
    n = len(df_test) - 1
    correct = sum(df_test.loc[t, TARGET] == df_test.loc[t + 1, TARGET] for t in range(n))
    return {"model": "baseline(persist)", "n_eval": n, "accuracy": correct / n, "mean_p_true": None, "mean_nll": None}


def eval_dbn(infer, df_test, evidence_vars, label):
    model = infer.model
    var_t1 = f"{TARGET}_t1"

    # target states from trained model
    allowed = None
    sn = getattr(model, "state_names", {})
    if sn and var_t1 in sn and sn[var_t1] is not None:
        allowed = list(sn[var_t1])

    n = len(df_test) - 1
    correct = 0
    p_trues = []
    nlls = []

    for t in range(n):
        ev = {f"{v}_t": int(df_test.loc[t, v]) for v in evidence_vars if v in df_test.columns}
        true_next = int(df_test.loc[t + 1, TARGET])

        # index is true_next because we forced consistent discretization (0..k-1)
        true_idx = true_next

        q = infer.query([var_t1], evidence=ev, show_progress=False)
        factor = get_factor(q, var_t1)
        probs = np.asarray(factor.values, dtype=float).reshape(-1)
        probs = probs / probs.sum()

        pred_idx = int(np.argmax(probs))
        if pred_idx == true_idx:
            correct += 1

        p_true = float(probs[true_idx])
        p_trues.append(p_true)
        nlls.append(-np.log(max(p_true, EPS)))

    return {
        "model": label,
        "n_eval": n,
        "accuracy": correct / n,
        "mean_p_true": float(np.mean(p_trues)),
        "mean_nll": float(np.mean(nlls)),
    }


def main():
    # 1) Load RAW (do not discretize yet)
    df_raw = load_and_clean(CSV_PATH).reset_index(drop=True)

    if TARGET not in df_raw.columns:
        raise ValueError(f"TARGET '{TARGET}' not found.")

    # 2) Time-based split on RAW
    n = len(df_raw)
    split = int(np.floor(TRAIN_FRAC * n))
    split = min(max(split, 2), n - 2)

    raw_train = df_raw.iloc[:split].copy()
    raw_test = df_raw.iloc[split:].copy().reset_index(drop=True)

    # 3) Fit discretizer on TRAIN only, apply to train+test consistently
    bins = fit_discretizer_fd(raw_train, max_bins=MAX_BINS)
    df_train, df_test = transform_with_bins_and_state_maps(raw_train, raw_test, bins)

    # Drop any remaining NaNs just in case
    df_train = df_train.dropna().reset_index(drop=True)
    df_test = df_test.dropna().reset_index(drop=True)

    print("\n=== STEP 5 FIXED: train/test with consistent discretization ===")
    print(f"Raw total rows: {n}")
    print(f"Train rows: {len(df_train)}  | Test rows: {len(df_test)}")

    # 4) Train DBN on TRAIN
    model_2s, _dbn = build_dbn(df_train, method="normal")
    infer = VariableElimination(model_2s)

    # 5) Evaluate on TEST
    ev_subset = [v for v in SUBSET_EVIDENCE if v in df_test.columns]
    ev_full = list(df_test.columns)

    results = []
    results.append(eval_baseline_persistence(df_test))
    results.append(eval_dbn(infer, df_test, [TARGET], "DBN(throughput_only)"))
    results.append(eval_dbn(infer, df_test, ev_subset, "DBN(subset_evidence)"))
    results.append(eval_dbn(infer, df_test, ev_full, "DBN(full_state)"))

    out = pd.DataFrame(results)
    print("\nTrain/Test results on TEST set:")
    with pd.option_context("display.width", 200):
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
