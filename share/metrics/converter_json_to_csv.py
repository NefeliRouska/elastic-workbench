import pandas as pd
import json
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Prometheus dump to wide DBN CSV"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to raw Prometheus dump CSV"
    )
    return parser.parse_args()

args = parse_args()

IN_PATH  = Path(args.csv)
OUT_WIDE = IN_PATH.parent / IN_PATH.name.replace("prom_dump_all_", "dbn_wide_")
OUT_MAP  = IN_PATH.parent / IN_PATH.name.replace("prom_dump_all_", "dbn_variable_mapping_")

#IN_PATH  = Path("share/metrics/prom_dump_all_20260310_160246_seed161312.csv")
#OUT_WIDE = Path("share/metrics/dbn_wide_20260310_160246_seed161312.csv")
#OUT_MAP  = Path("share/metrics/dbn_variable_mapping_20260310_160246_seed161312.csv")

# These labels often change but are not helpful for stable identity.
# Keep container_id / operation / device etc.
IGNORE_KEYS = {"__name__"}

def signature(metric_name: str, labels: dict) -> str:
    """
    Build a deterministic identity signature for a Prometheus time series:
    metric_name + sorted(label_key=label_value) after removing IGNORE_KEYS.
    """
    kept = {k: str(v) for k, v in labels.items() if k not in IGNORE_KEYS}
    items = sorted(kept.items())
    return metric_name + "|" + "|".join(f"{k}={v}" for k, v in items)

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Cannot find {IN_PATH.resolve()}")

    df = pd.read_csv(IN_PATH)

    required = {"metric_name", "timestamp", "value", "labels_json"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    has_run_tag = "run_tag" in df.columns

    # Parse labels JSON
    df["labels_dict"] = df["labels_json"].apply(json.loads)

    # Build series signature per row
    df["sig"] = df.apply(lambda r: signature(r["metric_name"], r["labels_dict"]), axis=1)

    # Assign per-metric indices based on sorted unique signatures (deterministic)
    df["series_idx"] = -1
    mapping_rows = []

    for metric, sub in df.groupby("metric_name", sort=True):
        sigs = sorted(sub["sig"].unique())
        sig_to_idx = {s: i + 1 for i, s in enumerate(sigs)}
        df.loc[sub.index, "series_idx"] = sub["sig"].map(sig_to_idx)

        # store one example labels_json for each signature
        for s in sigs:
            ex_row = sub[sub["sig"] == s].iloc[0]
            mapping_rows.append({
                "variable": f"{metric}_{sig_to_idx[s]}",
                "metric_name": metric,
                "series_idx": sig_to_idx[s],
                "signature": s,
                "labels_json": ex_row["labels_json"],
            })

    df["variable"] = df["metric_name"] + "_" + df["series_idx"].astype(int).astype(str)

    # Make sure timestamp/value are numeric
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Build wide table
    index_cols = ["timestamp"]
    if has_run_tag:
        index_cols = ["run_tag", "timestamp"]

    wide = df.pivot_table(
        index=index_cols,
        columns="variable",
        values="value",
        aggfunc="mean",   # if duplicates at same time exist
    ).sort_index()

    # Flatten columns from Index to plain strings
    wide.columns = [str(c) for c in wide.columns]

    # Save outputs
    OUT_WIDE.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUT_WIDE, index=True)

    pd.DataFrame(mapping_rows).to_csv(OUT_MAP, index=False)

    print("Wrote wide DBN table:", OUT_WIDE)
    print("Wrote variable mapping:", OUT_MAP)
    print(f"Wide shape: {wide.shape[0]} rows × {wide.shape[1]} columns")
    if wide.shape[1] > 2000:
        print("Note: you have many series/columns. DBNs may need dimensionality reduction/aggregation later.")

if __name__ == "__main__":
    main()
