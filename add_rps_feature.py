#!/usr/bin/env python3
"""
add_rps_feature.py

Reconstructs the experiment RPS signal for each 30-second window in a wide-format
DBN dataset and appends it as a new column `rps`.

RPS logic is a direct port from the experiment orchestrator (experiment_orchestrator.py).
All constants must match those in the orchestrator exactly.

Usage:
    python add_rps_feature.py \
        --csv dbn_wide_20260313_192652_seed783122.csv \
        --pattern periodic \
        --out dbn_wide_20260313_192652_seed783122_with_rps.csv

Patterns:
    periodic        Cosine-based, period=3000s, range [150,350], quantised to 60s steps.
                    Starts at peak (350 RPS) at elapsed=0. Seed not used.
    oial            200 RPS baseline; spike to 600 RPS at [40min, 43min); back to 200.
    unpredictable   random.Random(seed*100000 + slot).randint(100,500), new slot every 600s.
"""

import argparse
import random
import sys
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# Constants — keep in sync with experiment_orchestrator.py
# ============================================================

# Periodic
PERIODIC_LOW              = 150
PERIODIC_HIGH             = 350
PERIODIC_SINE_PERIOD_SEC  = 3000   # 50 minutes
PERIODIC_UPDATE_SEC       = 60     # 1 minute
PERIODIC_START_AT_PEAK    = True   # True -> cosine (starts at 350)

# Once-in-a-lifetime
OIAL_BASE       = 200
OIAL_SPIKE      = 600
OIAL_BEFORE_SEC = 2400   # 40 min before spike
OIAL_SPIKE_SEC  = 180    # 3 min spike

# Unpredictable
UNPREDICTABLE_MIN      = 100
UNPREDICTABLE_MAX      = 500
UNPREDICTABLE_STEP_SEC = 600   # 10 minutes


# ============================================================
# RPS signal reconstruction (ported from orchestrator)
# ============================================================

def rps_periodic(elapsed_seconds: np.ndarray) -> np.ndarray:
    mean      = (PERIODIC_LOW + PERIODIC_HIGH) / 2.0    # 250
    amplitude = (PERIODIC_HIGH - PERIODIC_LOW) / 2.0    # 100
    sampled_time = (elapsed_seconds // PERIODIC_UPDATE_SEC) * PERIODIC_UPDATE_SEC
    angle        = 2.0 * np.pi * sampled_time / PERIODIC_SINE_PERIOD_SEC
    rps          = mean + amplitude * np.cos(angle)
    rps_int = np.round(rps).astype(int)
    return np.clip(rps_int, PERIODIC_LOW, PERIODIC_HIGH)


def rps_oial(elapsed_seconds: np.ndarray) -> np.ndarray:
    spike_start = float(OIAL_BEFORE_SEC)
    spike_end   = float(OIAL_BEFORE_SEC + OIAL_SPIKE_SEC)
    return np.where(
        (elapsed_seconds >= spike_start) & (elapsed_seconds < spike_end),
        float(OIAL_SPIKE),
        float(OIAL_BASE),
    )


def rps_unpredictable(elapsed_seconds: np.ndarray, seed: int) -> np.ndarray:
    slots        = (elapsed_seconds // UNPREDICTABLE_STEP_SEC).astype(int)
    unique_slots = np.unique(slots)
    slot_to_rps  = {}
    for s in unique_slots:
        local_rng = random.Random(seed * 100000 + int(s))
        slot_to_rps[int(s)] = local_rng.randint(UNPREDICTABLE_MIN, UNPREDICTABLE_MAX)
    return np.array([slot_to_rps[int(s)] for s in slots])


# ============================================================
# Timestamp helpers
# ============================================================

CANDIDATE_TIMESTAMP_COLS = ["timestamp", "time", "Timestamp", "Time", "datetime"]


def find_timestamp_col(df: pd.DataFrame) -> str:
    for name in CANDIDATE_TIMESTAMP_COLS:
        if name in df.columns:
            return name
    for col in df.columns:
        if "time" in col.lower() or "stamp" in col.lower():
            return col
    raise ValueError(
        f"Cannot find a timestamp column. Columns present: {list(df.columns)}"
    )


def to_elapsed_seconds(series: pd.Series) -> np.ndarray:
    """Convert a timestamp series to elapsed seconds from experiment start.
    Tries numeric (Unix epoch) first, falls back to datetime string parsing.
    """
    # Try numeric first — timestamps here are Unix epoch floats.
    # pd.to_datetime() must NOT be tried first: it silently mis-parses
    # epoch floats as nanosecond-scale datetimes, giving elapsed ~ 0.
    vals = pd.to_numeric(series, errors="coerce").values.astype(float)
    if not np.isnan(vals).all():
        return vals - vals.min()

    # Fallback for human-readable datetime strings
    ts      = pd.to_datetime(series)
    elapsed = (ts - ts.min()).dt.total_seconds().values
    return elapsed.astype(float)


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Add reconstructed RPS feature to a wide-format DBN CSV."
    )
    p.add_argument("--csv",     required=True,
                   help="Path to input wide CSV file.")
    p.add_argument("--pattern", required=True,
                   choices=["periodic", "oial", "unpredictable"],
                   help="Workload pattern: periodic | oial | unpredictable")
    p.add_argument("--seed",    type=int, default=783122,
                   help="Random seed used by the experiment orchestrator (default: 783122).")
    p.add_argument("--out",     default=None,
                   help="Output path. Defaults to <stem>_with_rps.csv alongside input.")
    return p.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] Input CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else csv_path.with_name(
        csv_path.stem + "_with_rps" + csv_path.suffix
    )

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape}")

    ts_col = find_timestamp_col(df)
    print(f"  Timestamp column : '{ts_col}'")

    if "run_tag" in df.columns:
        print(f"  run_tag column   : present (ignored for RPS reconstruction)")

    elapsed = to_elapsed_seconds(df[ts_col])
    print(f"  Elapsed range    : {elapsed.min():.1f}s – {elapsed.max():.1f}s  "
          f"({elapsed.max() / 3600:.2f} h)")

    if args.pattern == "periodic":
        rps_values = rps_periodic(elapsed)
    elif args.pattern == "oial":
        rps_values = rps_oial(elapsed)
    elif args.pattern == "unpredictable":
        rps_values = rps_unpredictable(elapsed, args.seed)
    else:
        raise ValueError(f"Unknown pattern: {args.pattern}")

    print(f"  RPS range        : [{rps_values.min()}, {rps_values.max()}]")
    print(f"  Unique RPS values: {len(np.unique(rps_values))}")

    # Insert rps immediately after timestamp column
    insert_pos = df.columns.get_loc(ts_col) + 1
    df.insert(insert_pos, "rps", rps_values)

    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()