"""
control_variable_check.py

Checks whether changes in control variables (cores_*, data_quality_*)
are followed by changes in throughput_3 more often than baseline.

If control changes predict throughput changes, that's a real
early-warning signal persistence structurally cannot use -- persistence
only ever looks at throughput_3's own past value.

Usage:
    python control_variable_check.py --csv path/to/dbn_wide_*.csv
"""

import argparse
import numpy as np
import pandas as pd

TARGET = "throughput_3"
MODELING_GRANULARITY_SEC = 30


def load_and_aggregate(csv_path):
    df = pd.read_csv(csv_path)
    for col in list(df.columns):
        if "time" in col.lower():
            df.drop(columns=[col], inplace=True)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    df = df.select_dtypes(include=[np.number]).dropna().reset_index(drop=True)

    group_ids = np.arange(len(df)) // MODELING_GRANULARITY_SEC
    return df.groupby(group_ids).mean().reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    raw = load_and_aggregate(args.csv)

    control_cols = [c for c in raw.columns
                    if c.lower().startswith(("cores_", "data_quality_"))]

    if not control_cols:
        print("No cores_* or data_quality_* columns found in this CSV.")
        return

    print(f"[DATA] {len(raw)} rows after aggregation")
    print(f"[CONTROL COLUMNS FOUND] {control_cols}\n")

    # "acted" = did ANY control variable change value from the
    # previous row to this one
    control_diff = raw[control_cols].diff().abs().sum(axis=1)
    acted = (control_diff > 1e-9)

    # discretize throughput_3 into 4 bins, same as the main pipeline,
    # purely so "did it change" means "did it cross a bin boundary",
    # not "did it move by 0.0001"
    target_bins = pd.qcut(raw[TARGET], 4, labels=False, duplicates="drop")
    changes_next = (target_bins.shift(-1) != target_bins)

    # drop the last row (no "next" to compare against) and align
    df_check = pd.DataFrame({
        "acted": acted,
        "changes_next": changes_next,
    }).iloc[:-1]

    summary = df_check.groupby("acted")["changes_next"].agg(["mean", "count"])
    summary.index = summary.index.map({False: "no control change", True: "control changed"})

    print("=== RESULT ===")
    print(summary.rename(columns={"mean": "P(throughput changes next step)", "count": "n_rows"}))

    rate_no_action = summary.loc["no control change", "mean"]
    rate_action = summary.loc["control changed", "mean"] if "control changed" in summary.index else float("nan")

    print()
    if np.isnan(rate_action):
        print("No rows with a control change found -- cannot compare.")
    else:
        ratio = rate_action / rate_no_action if rate_no_action > 0 else float("inf")
        print(f"Baseline change rate (no control action): {rate_no_action:.1%}")
        print(f"Change rate right after a control action:  {rate_action:.1%}")
        print(f"Ratio: {ratio:.2f}x")
        print()
        if ratio > 2:
            print("STRONG signal: control changes are followed by throughput "
                  "changes much more often than baseline. This is a real, "
                  "usable early-warning signal persistence cannot access.")
        elif ratio > 1.3:
            print("MODEST signal: some elevated change rate after control "
                  "actions, worth investigating further but not dramatic.")
        else:
            print("NO signal: control changes are not followed by "
                  "throughput changes any more often than baseline. "
                  "This path does not look promising.")

    # Also check per individual control column, in case one dominates
    print("\n=== PER-COLUMN BREAKDOWN ===")
    for col in control_cols:
        col_diff = raw[col].diff().abs()
        col_acted = (col_diff > 1e-9).iloc[:-1]
        if col_acted.sum() == 0:
            print(f"{col}: never changes in this data, skipping")
            continue
        rate = changes_next.iloc[:-1][col_acted].mean()
        n = col_acted.sum()
        print(f"{col}: change rate after action = {rate:.1%} (n={n} actions) "
              f"vs baseline {rate_no_action:.1%}")


if __name__ == "__main__":
    main()