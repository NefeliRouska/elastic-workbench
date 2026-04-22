"""
markov_test.py
==============
Standalone diagnostic script to test whether your system
satisfies the first-order Markov assumption.

Run this ONCE before the sweep to justify your DBN design.

What it does:
  - Loads and cleans your data the same way the sweep does
  - Builds two sets of features:
      First order:  features at time t only
      Second order: features at time t AND t-1
  - Trains a Random Forest classifier on each
  - Compares cross-validated accuracy

Interpretation:
  - Difference < 2-3%  -> Markov assumption holds, DBN justified
  - Difference > 2-3%  -> System has memory beyond one step,
                          acknowledge as limitation in paper

Usage:
    python markov_test.py
"""

import ast
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import cross_val_score


# ============================================================
# CONFIG
# ============================================================

# Single CSV (currently active)
CSV_PATH = "share/metrics/dbn_wide_20260310_160246_seed161312.csv"

# Multiple CSVs — uncomment when you have all four workload files
# CSV_PATHS = [
#     "share/metrics/dbn_wide_static_seed161312.csv",
#     "share/metrics/dbn_wide_periodic_seed161312.csv",
#     "share/metrics/dbn_wide_oial_seed161312.csv",
#     "share/metrics/dbn_wide_unpredictable_seed161312.csv",
# ]

TARGETS                   = ["throughput_1", "throughput_2", "throughput_3"]
MODELING_GRANULARITY_SEC  = 30   # must match sweep script
N_BINS_TO_TEST            = [3, 4, 5]  # test robustness across bin counts
CV_FOLDS                  = 5
EXCLUDE_OTHER_THROUGHPUTS = True


# ============================================================
# LOAD + CLEAN  (identical to sweep script)
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
        df  = pd.concat([df.drop(columns=["s_config"]), cfg], axis=1)

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
    X = features at t
    y = target at t+1
    """
    cols = get_predictor_cols(df, target)
    X    = df[cols].iloc[:-1].reset_index(drop=True)
    y    = df[target].iloc[1:].reset_index(drop=True)
    return X, y


def second_order_features(df, target):
    """
    X = features at t AND features at t-1
    y = target at t+1
    Requires at least 3 rows.
    """
    cols   = get_predictor_cols(df, target)
    X_t    = df[cols].iloc[1:-1].reset_index(drop=True)
    X_tm1  = df[cols].iloc[:-2].reset_index(drop=True)
    X_tm1.columns = [f"{c}_lag1" for c in X_tm1.columns]
    X      = pd.concat([X_t, X_tm1], axis=1)
    y      = df[target].iloc[2:].reset_index(drop=True)
    return X, y


# ============================================================
# DISCRETIZE TARGET
# ============================================================
def discretize(y, n_bins):
    kbd = KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy="uniform"
    )
    return kbd.fit_transform(
        y.to_numpy().reshape(-1, 1)
    ).astype(int).flatten()


# ============================================================
# MAIN TEST
# ============================================================
def run_markov_test(df, target, n_bins):
    X1, y1 = first_order_features(df, target)
    X2, y2 = second_order_features(df, target)

    y1_disc = discretize(y1, n_bins)
    y2_disc = discretize(y2, n_bins)

    clf = RandomForestClassifier(
        n_estimators=100, random_state=0, n_jobs=-1
    )

    score1 = float(np.mean(cross_val_score(
        clf, X1, y1_disc, cv=CV_FOLDS, scoring="accuracy"
    )))
    score2 = float(np.mean(cross_val_score(
        clf, X2, y2_disc, cv=CV_FOLDS, scoring="accuracy"
    )))

    return score1, score2, score2 - score1


def interpret(diff):
    if abs(diff) < 0.02:
        return "HOLDS — difference < 2%, first-order Markov assumption is justified"
    elif abs(diff) < 0.05:
        return "BORDERLINE — difference 2-5%, weak memory effect, acknowledge in paper"
    else:
        return "VIOLATED — difference > 5%, system has significant memory beyond one step"


def main():
    print("=" * 60)
    print("MARKOV PROPERTY TEST")
    print("=" * 60)

    # Load data
    # To use multiple CSVs uncomment the block below
    # and comment out the single CSV load:
    # --------------------------------------------------
    # dfs = []
    # for path in CSV_PATHS:
    #     df = load_and_clean(path)
    #     dfs.append(df)
    # raw = pd.concat(dfs, ignore_index=True)
    # --------------------------------------------------
    raw = load_and_clean(CSV_PATH)
    raw = aggregate(raw)

    print(f"Rows after aggregation: {len(raw)}")
    print(f"Granularity: {MODELING_GRANULARITY_SEC} seconds per row")
    print()

    results = []

    for target in TARGETS:
        if target not in raw.columns:
            print(f"[SKIP] {target} not found in data\n")
            continue

        print(f"Target: {target}")
        print("-" * 40)

        for n_bins in N_BINS_TO_TEST:
            score1, score2, diff = run_markov_test(raw, target, n_bins)
            verdict = interpret(diff)

            print(f"  n_bins={n_bins}")
            print(f"    First order  (t only):    {score1:.3f}")
            print(f"    Second order (t + t-1):   {score2:.3f}")
            print(f"    Difference:               {diff:+.3f}")
            print(f"    Verdict: {verdict}")
            print()

            results.append({
                "target":       target,
                "n_bins":       n_bins,
                "score_first":  score1,
                "score_second": score2,
                "difference":   diff,
                "verdict":      verdict,
            })

    # Save results
    out = pd.DataFrame(results)
    out.to_csv("markov_test_results.csv", index=False)
    print("=" * 60)
    print("Results saved to markov_test_results.csv")
    print()

    # Overall summary
    print("OVERALL SUMMARY")
    print("-" * 40)
    avg_diff = float(out["difference"].mean())
    print(f"Average difference across all targets and bin counts: {avg_diff:+.3f}")
    print(f"Overall verdict: {interpret(avg_diff)}")
    print()
    print("Use this result to justify (or caveat) the first-order")
    print("Markov assumption in your paper.")


if __name__ == "__main__":
    main()