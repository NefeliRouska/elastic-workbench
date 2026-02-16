import time
import random
import csv
import json
from itertools import product
from pathlib import Path
import requests

# ====== CONFIG ======
BASE = "http://localhost:8080"
PROM = "http://localhost:9090"   # <-- your Prometheus

ENDPOINTS = {
    "cores":   f"{BASE}/resource_scaling",
    "quality": f"{BASE}/quality_scaling",
}

CORES = list(range(1, 9))
QUALITIES = list(range(100, 1001, 100))

RANDOM_SEED = 42
STABILIZE_SEC = 10
MEASURE_SEC   = 60

STEP = "5s"  # keep close to your scrape interval; increase if you have lots of series

OUTPUT_CSV = Path("share/metrics/prom_dump_all.csv")
# =====================

def set_cores(n: int):
    r = requests.put(ENDPOINTS["cores"], params={"cores": n}, timeout=5)
    r.raise_for_status()

def set_quality(q: int):
    r = requests.put(ENDPOINTS["quality"], params={"data_quality": q}, timeout=5)
    r.raise_for_status()

def prom_metric_names(timeout=30):
    r = requests.get(f"{PROM}/api/v1/label/__name__/values", timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if j.get("status") != "success":
        raise RuntimeError(j)
    return j["data"]

def prom_query_range(query: str, start: float, end: float, step: str, timeout=60):
    r = requests.get(
        f"{PROM}/api/v1/query_range",
        params={"query": query, "start": start, "end": end, "step": step},
        timeout=timeout,
    )
    r.raise_for_status()
    j = r.json()
    if j.get("status") != "success":
        raise RuntimeError(j)
    return j["data"]["result"]

def append_results_csv(path: Path, run_tag: str, metric_name: str, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not path.exists()) or (path.stat().st_size == 0)

    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["run_tag", "metric_name", "timestamp", "value", "labels_json"])

        for series in result:
            labels = series.get("metric", {})
            values = series.get("values", [])
            for ts, val in values:
                w.writerow([run_tag, metric_name, float(ts), val, json.dumps(labels, sort_keys=True)])

def main():
    combos = list(product(CORES, QUALITIES))
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(combos)

    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()

    # Snapshot the list of metric names ONCE (or do it per window if metrics appear/disappear)
    all_metric_names = prom_metric_names()
    # Optional: filter out internal names if you want, but you said "all", so keep everything.

    #print(f"Prometheus metric names: {len(all_metric_names)}")
    #print(f"Prometheus metric names: {all_metric_names}")

    for i, (cores, quality) in enumerate(combos, start=1):
        run_tag = f"cores-{cores}_dq-{quality}"
        print(f"\n[{i}/{len(combos)}] {run_tag}")

        set_cores(cores)
        set_quality(quality)

        print(f"Stabilizing {STABILIZE_SEC}s…")
        time.sleep(STABILIZE_SEC)

        t_start = time.time()
        time.sleep(MEASURE_SEC)
        t_end = time.time()

        # Query ALL metric names for this window
        for mn in all_metric_names:
            # Query expression: just the metric name
            try:
                result = prom_query_range(mn, t_start, t_end, STEP, timeout=60)
                if result:
                    append_results_csv(OUTPUT_CSV, run_tag, mn, result)
            except requests.exceptions.Timeout:
                # Too heavy metric; skip or log
                print(f"  [timeout] {mn}")
            except RuntimeError as e:
                # Prometheus may reject queries that exceed max samples
                print(f"  [error] {mn}: {e}")

    print("\nDone:", OUTPUT_CSV)

if __name__ == "__main__":
    main()
