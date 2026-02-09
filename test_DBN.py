import numpy as np
import pandas as pd
from pgmpy.inference import VariableElimination

from full_dynamic_bn_learn_final import load_and_clean, prepare_normal, build_dbn

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "prom_all_metrics_wide.csv"
TARGET = "throughput"

# How many transitions to evaluate in the "quick" tests (Steps 2–4)
MAX_TRANSITIONS_QUICK = 300

# Train/test split for Step 5 (time-based: earlier rows train, later rows test)
TRAIN_FRAC = 0.8

# Numerical stability
EPS = 1e-12

# Evidence set for the "subset" tests (edit as you like)
SUBSET_EVIDENCE = [
    "throughput",
    "avg_p_latency",
    "cores",
    "data_quality",
    "container_cpu_usage_seconds_total",
    "container_memory_usage_bytes",
    "container_spec_cpu_quota",
]

# ============================================================
# Helpers for pgmpy quirks (state labels + query return types)
# ============================================================

def get_factor(query_result, var_name):
    """pgmpy version compatibility: query returns dict or DiscreteFactor."""
    return query_result[var_name] if isinstance(query_result, dict) else query_result


def coerce_to_state_label(model, var, value):
    """
    Convert `value` into a state label that actually exists in model.state_names[var].

    Handles cases where model expects states like "0"..."4" but df has ints 0..4,
    or other mild type mismatches.
    """
    sn = getattr(model, "state_names", None)
    if not sn or var not in sn or sn[var] is None:
        return value

    allowed = list(sn[var])

    # Common candidate representations
    candidates = [value]

    try:
        candidates.append(str(value))
    except Exception:
        pass

    try:
        candidates.append(str(int(value)))
    except Exception:
        pass

    try:
        candidates.append(int(value))
    except Exception:
        pass

    # Pick the first candidate that is a valid state label
    for c in candidates:
        if c in allowed:
            return c

    # If nothing matches, return original (pgmpy will error; but now it's informative)
    return value



def label_to_index(model, var, label):
    """
    Map a state label to an index in the probability vector.
    If model has explicit state_names, use those; else assume label is already index.
    """
    sn = getattr(model, "state_names", {})
    if sn and var in sn and sn[var] is not None:
        allowed = list(sn[var])
        return allowed.index(label), allowed
    return int(label), None


# ============================================================
# Core evaluators
# ============================================================

def smoke_test(model_2s, infer, df, t=50):
    """
    Step 1: One timepoint sanity check.
    Evidence: TARGET_t
    Query: P(TARGET_t1 | TARGET_t)
    Print distribution, argmax, p_true.
    """
    t = min(t, len(df) - 2)

    var_t = f"{TARGET}_t"
    var_t1 = f"{TARGET}_t1"

    x_t = df.loc[t, TARGET]
    true_next = df.loc[t + 1, TARGET]

    ev_val = coerce_to_state_label(model_2s, var_t, x_t)
    evidence = {var_t: ev_val}

    q = infer.query([var_t1], evidence=evidence, show_progress=False)
    factor = get_factor(q, var_t1)

    probs = np.asarray(factor.values, dtype=float).reshape(-1)
    probs = probs / probs.sum()

    true_label = coerce_to_state_label(model_2s, var_t1, true_next)
    true_idx, allowed = label_to_index(model_2s, var_t1, true_label)

    pred_idx = int(np.argmax(probs))
    pred_label = allowed[pred_idx] if allowed is not None else pred_idx

    print("\n=== STEP 1: SMOKE TEST (single transition) ===")
    print(f"t = {t}")
    print(f"Evidence: {var_t} = {ev_val}")
    print(f"True next: {var_t1} = {true_label}")
    print("\nPredicted distribution:", probs)
    print(f"\nArgmax prediction: {pred_label}  (prob={float(probs[pred_idx]):.4f})")
    print(f"Probability of true next: {float(probs[true_idx]):.4f}")


def evaluate_dbn_one_step(infer, df, evidence_vars, max_transitions=None):
    """
    Evaluate DBN one-step predictions for TARGET(t+1) given evidence at time t.

    evidence_vars are df column names (without suffix). We'll convert to *_t internally.

    Metrics:
      - accuracy: argmax match rate
      - mean_p_true: average probability assigned to true next state
      - mean_nll: average -log(p_true)
    """
    model = infer.model
    var_t1 = f"{TARGET}_t1"

    n_trans = len(df) - 1
    if max_transitions is not None:
        n_trans = min(n_trans, max_transitions)
    if n_trans <= 1:
        return {"n_eval": 0, "accuracy": np.nan, "mean_p_true": np.nan, "mean_nll": np.nan}

    correct = 0
    p_trues = []
    nlls = []

    for t in range(n_trans):
        # Build evidence dict for time t
        ev = {}
        for v in evidence_vars:
            if v not in df.columns:
                continue
            ev_name = f"{v}_t"
            ev[ev_name] = coerce_to_state_label(model, ev_name, df.loc[t, v])

        # True next label/index
        true_next = df.loc[t + 1, TARGET]
        true_label = coerce_to_state_label(model, var_t1, true_next)
        try:
            true_idx, _allowed = label_to_index(model, var_t1, true_label)
        except ValueError:
            # If state label isn't recognized, skip this transition
            continue

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

    n_eval = len(p_trues)
    if n_eval == 0:
        return {"n_eval": 0, "accuracy": np.nan, "mean_p_true": np.nan, "mean_nll": np.nan}

    return {
        "n_eval": n_eval,
        "accuracy": correct / n_eval,
        "mean_p_true": float(np.mean(p_trues)),
        "mean_nll": float(np.mean(nlls)),
    }


def evaluate_baseline_persistence(df, max_transitions=None):
    """
    Baseline: predict TARGET(t+1) = TARGET(t)
    Only produces accuracy (no probabilities => no NLL).
    """
    n_trans = len(df) - 1
    if max_transitions is not None:
        n_trans = min(n_trans, max_transitions)
    if n_trans <= 1:
        return {"n_eval": 0, "accuracy": np.nan}

    correct = 0
    for t in range(n_trans):
        if df.loc[t, TARGET] == df.loc[t + 1, TARGET]:
            correct += 1

    return {"n_eval": n_trans, "accuracy": correct / n_trans}


def print_table(rows, title):
    out = pd.DataFrame(rows)
    print(f"\n{title}")
    with pd.option_context("display.width", 200):
        print(out.to_string(index=False))


# ============================================================
# Main pipeline: run all steps
# ============================================================

def main():
    # ------------------------------------------------------------
    # Load + prep data
    # ------------------------------------------------------------
    df_raw = load_and_clean(CSV_PATH)
    df = prepare_normal(df_raw).reset_index(drop=True)

    if TARGET not in df.columns:
        raise ValueError(f"TARGET '{TARGET}' not found in dataframe columns.")

    # ------------------------------------------------------------
    # Build on ALL data for Steps 1–4 (quick diagnostics)
    # ------------------------------------------------------------
    model_2s_all, _dbn_all = build_dbn(df, method="normal")
    infer_all = VariableElimination(model_2s_all)

    # STEP 1: smoke test
    smoke_test(model_2s_all, infer_all, df, t=50)

    # STEP 2: throughput-only across many transitions
    print("\n=== STEP 2: QUICK EVAL (throughput-only evidence) ===")
    r2 = evaluate_dbn_one_step(infer_all, df, evidence_vars=[TARGET], max_transitions=MAX_TRANSITIONS_QUICK)
    print(r2)

    # STEP 3: evidence comparison
    print("\n=== STEP 3: QUICK EVIDENCE COMPARISON (same data) ===")
    ev_subset = [v for v in SUBSET_EVIDENCE if v in df.columns]
    ev_full = list(df.columns)

    rows3 = []
    for name, ev in [
        ("DBN: throughput only", [TARGET]),
        ("DBN: subset evidence", ev_subset),
        ("DBN: full state", ev_full),
    ]:
        res = evaluate_dbn_one_step(infer_all, df, evidence_vars=ev, max_transitions=MAX_TRANSITIONS_QUICK)
        rows3.append({
            "model": name,
            "n_eval": res["n_eval"],
            "accuracy": res["accuracy"],
            "mean_p_true": res["mean_p_true"],
            "mean_nll": res["mean_nll"],
            "n_evidence_vars": len(ev),
        })
    print_table(rows3, "Evidence comparison (quick, same data)")

    # STEP 4: baseline vs DBN (quick, same data)
    print("\n=== STEP 4: QUICK BASELINE COMPARISON (same data) ===")
    base = evaluate_baseline_persistence(df, max_transitions=MAX_TRANSITIONS_QUICK)

    rows4 = [{
        "model": "Baseline: persistence (predict same bin)",
        "n_eval": base["n_eval"],
        "accuracy": base["accuracy"],
        "mean_p_true": None,
        "mean_nll": None,
        "n_evidence_vars": 1,
    }]

    for name, ev in [
        ("DBN: throughput only", [TARGET]),
        ("DBN: subset evidence", ev_subset),
        ("DBN: full state", ev_full),
    ]:
        res = evaluate_dbn_one_step(infer_all, df, evidence_vars=ev, max_transitions=MAX_TRANSITIONS_QUICK)
        rows4.append({
            "model": name,
            "n_eval": res["n_eval"],
            "accuracy": res["accuracy"],
            "mean_p_true": res["mean_p_true"],
            "mean_nll": res["mean_nll"],
            "n_evidence_vars": len(ev),
        })

    print_table(rows4, "Baseline vs DBN (quick, same data)")

    # ------------------------------------------------------------
    # STEP 5: 80/20 train/test split (time-based) — real test
    # ------------------------------------------------------------
    print("\n=== STEP 5: TRAIN/TEST (80/20 time split) + BASELINE + EVIDENCE SETS ===")

    n = len(df)
    split = int(np.floor(TRAIN_FRAC * n))
    split = min(max(split, 2), n - 2)

    df_train = df.iloc[:split].copy()
    df_test = df.iloc[split:].copy().reset_index(drop=True)

    # Train model ONLY on train
    model_2s_train, _dbn_train = build_dbn(df_train, method="normal")
    infer_train = VariableElimination(model_2s_train)

    # Evaluate on test transitions
    base_test = evaluate_baseline_persistence(df_test, max_transitions=None)

    rows5 = [{
        "model": "Baseline: persistence (predict same bin)",
        "dataset": "TEST",
        "n_eval": base_test["n_eval"],
        "accuracy": base_test["accuracy"],
        "mean_p_true": None,
        "mean_nll": None,
        "n_evidence_vars": 1,
    }]

    for name, ev in [
        ("DBN: throughput only", [TARGET]),
        ("DBN: subset evidence", ev_subset),
        ("DBN: full state", ev_full),
    ]:
        res = evaluate_dbn_one_step(infer_train, df_test, evidence_vars=ev, max_transitions=None)
        rows5.append({
            "model": name,
            "dataset": "TEST",
            "n_eval": res["n_eval"],
            "accuracy": res["accuracy"],
            "mean_p_true": res["mean_p_true"],
            "mean_nll": res["mean_nll"],
            "n_evidence_vars": len(ev),
        })

    print(f"\nTrain rows: {len(df_train)} | Test rows: {len(df_test)}")
    print_table(rows5, "Train/Test results (this is the main result to trust)")

    print("\nDone.")


if __name__ == "__main__":
    main()




"""
import pandas as pd
import numpy as np
from pgmpy.inference import DBNInference

from full_dynamic_bn_learn_final import build_dbn, load_and_clean, prepare_normal

# -------------------------------------------------
# LOAD + PREPARE DATA
# -------------------------------------------------
df_raw = load_and_clean("prom_all_metrics_wide.csv")
df_ready = prepare_normal(df_raw)

# IMPORTANT: use this for evidence
df = df_ready.reset_index(drop=True)

# -------------------------------------------------
# BUILD DBN
# -------------------------------------------------
model_2s, dbn = build_dbn(df_ready, method="normal")

# -------------------------------------------------
# CREATE DBN INFERENCE OBJECT
# -------------------------------------------------
dbn_infer = DBNInference(dbn)

# -------------------------------------------------
# PICK A TIME INDEX (NOT LAST ROW!)
# -------------------------------------------------
t = 50

# -------------------------------------------------
# COMMON EVIDENCE (REAL OBSERVED STATE AT TIME t)
# -------------------------------------------------
evidence_common = {
    ('throughput', 0): int(df.loc[t, 'throughput']),
    ('container_cpu_usage_seconds_total', 0): int(df.loc[t, 'container_cpu_usage_seconds_total']),
    ('container_memory_usage_bytes', 0): int(df.loc[t, 'container_memory_usage_bytes']),
    ('cores', 0): int(df.loc[t, 'cores']),
    ('data_quality', 0): int(df.loc[t, 'data_quality']),
}


# -------------------------------------------------
# QUERY 1: Throughput next step
# -------------------------------------------------
q1 = dbn_infer.query(
    variables=[('throughput', 1)],
    evidence={('throughput', 0): evidence_common[('throughput', 0)]}
)
print("\nQ1: P(throughput(t+1) | throughput(t))")
print(q1)

# -------------------------------------------------
# QUERY 2: Latency from throughput
# -------------------------------------------------
q2 = dbn_infer.query(
    variables=[('avg_p_latency', 1)],
    evidence={('throughput', 0): evidence_common[('throughput', 0)]}
)
print("\nQ2: P(latency(t+1) | throughput(t))")
print(q2)

# -------------------------------------------------
# QUERY 3: Latency from CPU usage
# -------------------------------------------------
q3 = dbn_infer.query(
    variables=[('avg_p_latency', 1)],
    evidence={('container_cpu_usage_seconds_total', 0):
              evidence_common[('container_cpu_usage_seconds_total', 0)]}
)
print("\nQ3: P(latency(t+1) | CPU usage(t))")
print(q3)

# -------------------------------------------------
# QUERY 4: Throughput from cores
# -------------------------------------------------
q4 = dbn_infer.query(
    variables=[('throughput', 1)],
    evidence={('cores', 0): evidence_common[('cores', 0)]}
)
print("\nQ4: P(throughput(t+1) | cores(t))")
print(q4)

# -------------------------------------------------
# QUERY 5: Memory failures from memory usage
# -------------------------------------------------
q5 = dbn_infer.query(
    variables=[('container_memory_failures_total', 1)],
    evidence={('container_memory_usage_bytes', 0):
              evidence_common[('container_memory_usage_bytes', 0)]}
)
print("\nQ5: P(memory failures(t+1) | memory usage(t))")
print(q5)

# -------------------------------------------------
# QUERY 6: CPU throttling from usage + quota
# -------------------------------------------------
q6 = dbn_infer.query(
    variables=[('container_cpu_cfs_throttled_periods_total', 1)],
    evidence={
        ('container_cpu_usage_seconds_total', 0):
            evidence_common[('container_cpu_usage_seconds_total', 0)],
        ('container_spec_cpu_quota', 0):
            int(df.loc[t, 'container_spec_cpu_quota'])
    }
)
print("\nQ6: P(CPU throttling(t+1) | usage, quota(t))")
print(q6)

# -------------------------------------------------
# QUERY 7: Full system → throughput
# -------------------------------------------------
q7 = dbn_infer.query(
    variables=[('throughput', 1)],
    evidence=evidence_common
)
print("\nQ7: P(throughput(t+1) | full system state(t))")
print(q7)

# -------------------------------------------------
# QUERY 8: MULTI-STEP prediction (DBN advantage)
# -------------------------------------------------
q8 = dbn_infer.query(
    variables=[('throughput', 3)],
    evidence={('throughput', 0): evidence_common[('throughput', 0)]}
)
print("\nQ8: P(throughput(t+3) | throughput(t))")
print(q8)

# -------------------------------------------------
# QUERY 9: Prediction uncertainty (entropy)
# -------------------------------------------------
entropy = q1[('throughput', 1)].entropy()
print("\nQ9: Prediction entropy for throughput(t+1)")
print("Entropy =", entropy)

# -------------------------------------------------
# QUERY 10: Probability of TRUE observed next state
# -------------------------------------------------
true_next = int(df.loc[t + 1, 'throughput'])
prob_true = q1[('throughput', 1)].values[true_next]

print("\nQ10: Probability assigned to true throughput(t+1)")
print("True state =", true_next)
print("Predicted probability =", prob_true)
"""

