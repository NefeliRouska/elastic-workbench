import time
import random
from itertools import product
from pathlib import Path
import requests

# ====== CONFIG ======
BASE = "http://localhost:8080"

ENDPOINTS = {
    "cores":   f"{BASE}/resource_scaling",
    "quality": f"{BASE}/quality_scaling",
}

CORES = list(range(1, 9))                 # 1..8
QUALITIES = list(range(100, 1001, 100))   # 100..1000 step 100

RANDOM_SEED = 42          # change if you want a different (but reproducible) order
STABILIZE_SEC = 10        # wait after applying config
MEASURE_SEC   = 60        # measurement window

INPUT_CSV  = Path("share/metrics/metrics.csv")
OUTPUT_CSV = Path("share/metrics/metrics_nominal_changes_randomized.csv")
# =====================

def set_cores(n: int):
    r = requests.put(ENDPOINTS["cores"], params={"cores": n}, timeout=5)
    r.raise_for_status()

def set_quality(q: int):
    r = requests.put(ENDPOINTS["quality"], params={"data_quality": q}, timeout=5)
    r.raise_for_status()

def now_key19():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def slice_csv_append(in_path: Path, out_path: Path, start_key: str, end_key: str, write_header: bool):
    """Append lines from in_path whose timestamp (first 19 chars) ∈ [start_key, end_key]."""
    kept = 0
    with in_path.open("r", newline="") as fin, out_path.open("a", newline="") as fout:
        header = fin.readline()
        if write_header:
            fout.write(header)
        for line in fin:
            if len(line) < 19:
                continue
            key = line[:19]
            if start_key <= key <= end_key:
                fout.write(line)
                kept += 1
    return kept

def main():
    combos = list(product(CORES, QUALITIES)) #all possible combinations 
    print(combos)
    rng = random.Random(RANDOM_SEED)  # local RNG so other randomness in your system won't affect it
    rng.shuffle(combos)
    print(combos)
    print(f"Total configurations: {len(combos)}")
    print(f"Randomized order with seed={RANDOM_SEED}")

    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()  # start clean

    first_window = True
    for i, (cores, quality) in enumerate(combos, start=1):
        tag = f"cores-{cores}_dq-{quality}"
        print(f"\n[{i}/{len(combos)}] === Running randomized nominal change: {tag} ===")

        # Apply runtime settings
        set_cores(cores)
        set_quality(quality)

        # Wait, measure, and slice
        print(f"Stabilizing for {STABILIZE_SEC}s…")
        time.sleep(STABILIZE_SEC)
        start_key = now_key19()
        print("Start:", start_key)

        time.sleep(MEASURE_SEC)
        end_key = now_key19()
        print("End:  ", end_key)

        kept = slice_csv_append(INPUT_CSV, OUTPUT_CSV, start_key, end_key, write_header=first_window)
        first_window = False
        print(f"Appended {kept} rows to → {OUTPUT_CSV}")

    print(f"\nAll done. Final combined file → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
