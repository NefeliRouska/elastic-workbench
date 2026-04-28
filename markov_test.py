"""
markov_test.py
==============
Standalone diagnostic script to test whether your system
approximately satisfies a first-order Markov assumption.

Run this ONCE before the DBN sweep to justify your DBN design.

What it does:
  - Loads and cleans your data the same way the sweep does
  - Aggregates rows to match the DBN modeling granularity
  - Builds two comparable feature sets using the SAME prediction rows:
      First order:  features at time t
      Second order: features at time t AND t-1
  - Trains a Random Forest classifier on each
  - Compares time-series cross-validated accuracy

Interpretation:
  - Second-order improvement < 2%  -> first-order assumption is adequate
  - Improvement 2-5%              -> weak memory effect, acknowledge as limitation
  - Improvement > 5%              -> substantial memory beyond one step

Important:
  This is an empirical diagnostic, not a formal proof of the Markov property.

Usage:
    python markov_test.py
"""

import ast
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import cross_val_score, TimeSeriesSplit


# ============================================================
# CONFIG
# ============================================================

# Single CSV
CSV_PATH = "share/metrics/dbn_wide_20260310_160246_seed161312.csv"

# Multiple CSVs — uncomment when you have all four workload files
# CSV_PATHS = [
#     "share/metrics/dbn_wide_static_seed161312.csv",
#     "share/metrics/dbn_wide_periodic_seed161312.csv",
#     "share/metrics/dbn_wide_oial_seed161312.csv",
#     "share/metrics/dbn_wide_unpredictable_seed161312.csv",
# ]

TARGETS = ["throughput_1", "throughput_2", "throughput_3"]

MODELING_GRANULARITY_SEC = 30

N_BINS_TO_TEST = [3, 4, 5]

CV_FOLDS = 5

EXCLUDE_OTHER_THROUGHPUTS = True

RANDOM_STATE = 0


# ============================================================
# LOAD + CLEAN
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

    # Drop time columns
    for col in list(df.columns):
        if "time" in col.lower():
            df.drop(columns=[col], inplace=True)

    # Convert possible numeric strings
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
# FEATURE BUILDERS
# ============================================================

def get_predictor_cols(df, target):
    return [
        c for c in df.columns
        if c != target
        and not (
            EXCLUDE_OTHER_THROUGHPUTS
            and c.startswith("throughput_")
        )
    ]


def first_order_features(df, target):
    """
    Fair first-order model:

    X = features at time t
    y = target at time t+1

    Uses t = 1 ... n-2 so that it matches the second-order model.
    """
    cols = get_predictor_cols(df, target)

    X = df[cols].iloc[1:-1].reset_index(drop=True)
    y = df[target].iloc[2:].reset_index(drop=True)

    return X, y


def second_order_features(df, target):
    """
    Second-order model:

    X = features at time t AND features at time t-1
    y = target at time t+1

    Uses the same prediction rows as first_order_features().
    """
    cols = get_predictor_cols(df, target)

    X_t = df[cols].iloc[1:-1].reset_index(drop=True)
    X_tm1 = df[cols].iloc[:-2].reset_index(drop=True)
    X_tm1.columns = [f"{c}_lag1" for c in X_tm1.columns]

    X = pd.concat([X_t, X_tm1], axis=1)
    y = df[target].iloc[2:].reset_index(drop=True)

    return X, y


# ============================================================
# DISCRETIZE TARGET
# ============================================================

def discretize(y, n_bins):
    kbd = KBinsDiscretizer(
        n_bins=n_bins,
        encode="ordinal",
        strategy="uniform"
    )

    y_disc = kbd.fit_transform(
        y.to_numpy().reshape(-1, 1)
    ).astype(int).flatten()

    return y_disc


# ============================================================
# MAIN TEST
# ============================================================

def run_markov_test(df, target, n_bins):
    X1, y1 = first_order_features(df, target)
    X2, y2 = second_order_features(df, target)

    if len(X1) != len(X2):
        raise ValueError("First-order and second-order feature sets are not aligned.")

    if not y1.equals(y2):
        raise ValueError("First-order and second-order targets are not aligned.")

    y_disc = discretize(y1, n_bins)

    min_class_count = pd.Series(y_disc).value_counts().min()

    if min_class_count < 2:
        raise ValueError(
            f"Not enough samples in at least one class for target={target}, "
            f"n_bins={n_bins}. Try fewer bins."
        )

    cv = TimeSeriesSplit(n_splits=CV_FOLDS)

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    score1 = float(np.mean(cross_val_score(
        clf,
        X1,
        y_disc,
        cv=cv,
        scoring="accuracy"
    )))

    score2 = float(np.mean(cross_val_score(
        clf,
        X2,
        y_disc,
        cv=cv,
        scoring="accuracy"
    )))

    improvement = score2 - score1

    return score1, score2, improvement


def interpret(improvement):
    if improvement < 0.02:
        return "HOLDS — second-order features do not meaningfully improve accuracy; first-order DBN is adequate"
    elif improvement < 0.05:
        return "BORDERLINE — second-order features improve accuracy by 2-5%; weak memory effect, acknowledge in paper"
    else:
        return "VIOLATED — second-order features improve accuracy by > 5%; memory beyond one step is substantial"


def main():
    print("=" * 60)
    print("MARKOV PROPERTY DIAGNOSTIC")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    # Single CSV
    raw = load_and_clean(CSV_PATH)

    # Multiple CSVs — uncomment this block and comment the line above
    # dfs = []
    # for path in CSV_PATHS:
    #     df = load_and_clean(path)
    #     dfs.append(df)
    # raw = pd.concat(dfs, ignore_index=True)

    raw = aggregate(raw)

    print(f"Rows after aggregation: {len(raw)}")
    print(f"Granularity: {MODELING_GRANULARITY_SEC} seconds per row")
    print(f"CV method: TimeSeriesSplit with {CV_FOLDS} folds")
    print()

    if len(raw) < CV_FOLDS + 3:
        raise ValueError(
            "Not enough rows after aggregation for the chosen number of CV folds. "
            "Reduce CV_FOLDS or use a smaller aggregation window."
        )

    results = []

    for target in TARGETS:
        if target not in raw.columns:
            print(f"[SKIP] {target} not found in data")
            print()
            continue

        print(f"Target: {target}")
        print("-" * 40)

        for n_bins in N_BINS_TO_TEST:
            try:
                score1, score2, improvement = run_markov_test(raw, target, n_bins)
                verdict = interpret(improvement)

                print(f"  n_bins={n_bins}")
                print(f"    First order  X(t):        {score1:.3f}")
                print(f"    Second order X(t), X(t-1): {score2:.3f}")
                print(f"    Improvement:              {improvement:+.3f}")
                print(f"    Verdict: {verdict}")
                print()

                results.append({
                    "target": target,
                    "n_bins": n_bins,
                    "score_first_order": score1,
                    "score_second_order": score2,
                    "improvement": improvement,
                    "verdict": verdict,
                })

            except Exception as e:
                print(f"  n_bins={n_bins}")
                print(f"    [ERROR] {e}")
                print()

    if len(results) == 0:
        print("No valid results were produced.")
        return

    out = pd.DataFrame(results)
    out.to_csv("markov_test_results.csv", index=False)

    print("=" * 60)
    print("Results saved to markov_test_results.csv")
    print()

    print("OVERALL SUMMARY")
    print("-" * 40)

    avg_improvement = float(out["improvement"].mean())
    max_improvement = float(out["improvement"].max())

    print(f"Average second-order improvement: {avg_improvement:+.3f}")
    print(f"Maximum second-order improvement: {max_improvement:+.3f}")
    print(f"Overall verdict based on average: {interpret(avg_improvement)}")


if __name__ == "__main__":
    main()