import time
import random
import csv
import json
from itertools import product
from pathlib import Path
import requests


# ================================
#            CONFIG
# ================================

PROM = "http://localhost:9090"  # Prometheus

# 3 services (edit these)
SERVICES = {
    "svc1": "http://localhost:8080",
    "svc2": "http://localhost:8081",
    "svc3": "http://localhost:8082",
}

# IMPORTANT: this is the fixed *route order* for the quality constraint:
# quality is allowed to stay the same or decrease as you move along this list.
ROUTE = ["svc1", "svc2", "svc3"]

ENDPOINTS = {
    "cores":   "/resource_scaling",   # PUT ?cores=N
    "quality": "/quality_scaling",    # PUT ?data_quality=Q
}

# Core constraints
TOTAL_CORES_BUDGET = 8
MIN_CORES_PER_SERVICE = 1   # set to 0 if cores=0 is valid in your system
MAX_CORES_PER_SERVICE = 8

# Quality domain (values you allow each service to draw)
QUALITIES = list(range(100, 1001, 100))

# Instead of (core_triples x global_quality), we do repeats per core triple,
# because quality is now sampled as a *triple* (q1,q2,q3) each run.
REPEATS_PER_CORE_TRIPLE = 10

# Experiment timing
RANDOM_SEED = 42
STABILIZE_SEC = 10
MEASURE_SEC   = 60

# Prometheus query step
STEP = "5s"  # keep close to your scrape interval

OUTPUT_CSV = Path("share/metrics/prom_dump_all.csv")


# ================================
#          HTTP HELPERS
# ================================

def _url(base: str, path: str) -> str:
    return f"{base}{path}"


def set_cores(service_base: str, n: int):
    """Set CPU cores for one service."""
    r = requests.put(_url(service_base, ENDPOINTS["cores"]), params={"cores": n}, timeout=5)
    r.raise_for_status()


def set_quality(service_base: str, q: int):
    """Set data quality for one service."""
    r = requests.put(_url(service_base, ENDPOINTS["quality"]), params={"data_quality": q}, timeout=5)
    r.raise_for_status()


# ================================
#       PROMETHEUS HELPERS
# ================================

def prom_metric_names(timeout=30):
    """Get all metric names currently known to Prometheus."""
    r = requests.get(f"{PROM}/api/v1/label/__name__/values", timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if j.get("status") != "success":
        raise RuntimeError(j)
    return j["data"]


def prom_query_range(query: str, start: float, end: float, step: str, timeout=60):
    """Range query for a metric name/expression."""
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


# ================================
#            CSV I/O
# ================================

def append_results_csv(path: Path, run_tag: str, metric_name: str, result):
    """
    Append Prometheus result to CSV.
    Each row: run_tag, metric_name, timestamp, value, labels_json
    """
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


# ================================
#   CORE / QUALITY SAMPLING LOGIC
# ================================

def generate_valid_core_triples():
    """
    Generate ALL valid (c1,c2,c3) such that:
      - MIN_CORES_PER_SERVICE <= ci <= MAX_CORES_PER_SERVICE
      - c1 + c2 + c3 <= TOTAL_CORES_BUDGET
    This guarantees we never violate the total core constraint.
    """
    triples = []
    for c1 in range(MIN_CORES_PER_SERVICE, MAX_CORES_PER_SERVICE + 1):
        for c2 in range(MIN_CORES_PER_SERVICE, MAX_CORES_PER_SERVICE + 1):
            for c3 in range(MIN_CORES_PER_SERVICE, MAX_CORES_PER_SERVICE + 1):
                if (c1 + c2 + c3) <= TOTAL_CORES_BUDGET:
                    triples.append((c1, c2, c3))
    return triples


def randomize_core_assignment(rng: random.Random, pattern: tuple[int, int, int], service_names: list[str]):
    """
    Avoid correlation: randomly permute which service gets which entry of pattern.
    Example pattern=(4,2,1) could become svc2=4, svc1=2, svc3=1, etc.
    """
    shuffled = service_names[:]  # copy
    rng.shuffle(shuffled)
    return {svc: c for svc, c in zip(shuffled, pattern)}


def sample_monotone_qualities(rng: random.Random, qualities: list[int]):
    """
    Your requested rule along the route:
      - svc1 gets q1 randomly
      - svc2 draws q2_raw randomly, but final q2 = min(q1, q2_raw)
      - svc3 draws q3_raw randomly, but final q3 = min(q2, q3_raw)

    This guarantees: q1 >= q2 >= q3 (quality can stay the same or decrease downstream).
    """
    q1 = rng.choice(qualities)

    q2_raw = rng.choice(qualities)
    q2 = min(q1, q2_raw)

    q3_raw = rng.choice(qualities)
    q3 = min(q2, q3_raw)

    return q1, q2, q3


# ================================
#              MAIN
# ================================

def main():
    # Basic config safety checks
    if set(ROUTE) != set(SERVICES.keys()):
        raise ValueError("ROUTE must contain exactly the same service names as SERVICES (just ordered).")
    if len(ROUTE) != 3:
        raise ValueError("This script is written for exactly 3 services (svc1->svc2->svc3).")

    rng = random.Random(RANDOM_SEED)
    service_names = list(SERVICES.keys())

    # Build full list of valid core triples, and randomize their order
    core_triples = generate_valid_core_triples()
    rng.shuffle(core_triples)

    # Fresh output
    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()

    # Snapshot metric names once
    all_metric_names = prom_metric_names()

    total_runs = len(core_triples) * REPEATS_PER_CORE_TRIPLE

    run_idx = 0
    for pattern in core_triples:
        for rep in range(REPEATS_PER_CORE_TRIPLE):
            run_idx += 1

            # 1) Assign cores to services with shuffle (anti-correlation)
            cores_map = randomize_core_assignment(rng, pattern, service_names)

            # Total-core safety check (should never fail if generator is correct)
            if sum(cores_map.values()) > TOTAL_CORES_BUDGET:
                raise RuntimeError("BUG: total core budget violated.")

            # 2) Sample route-constrained quality triple (svc1 >= svc2 >= svc3)
            q1, q2, q3 = sample_monotone_qualities(rng, QUALITIES)
            quality_map = {ROUTE[0]: q1, ROUTE[1]: q2, ROUTE[2]: q3}

            # Another safety check for the monotone property
            if not (quality_map[ROUTE[0]] >= quality_map[ROUTE[1]] >= quality_map[ROUTE[2]]):
                raise RuntimeError("BUG: quality is not monotone non-increasing along route.")

            # Tag includes exact per-service cores + per-service quality
            cores_tag = ",".join(f"{svc}={cores_map[svc]}" for svc in sorted(service_names))
            dq_tag    = ",".join(f"{svc}={quality_map[svc]}" for svc in ROUTE)
            run_tag = f"cores[{cores_tag}]_dq[{dq_tag}]_rep-{rep}"

            print(f"\n[{run_idx}/{total_runs}] {run_tag} (total_cores={sum(cores_map.values())})")

            # 3) Apply settings to ALL services
            # Apply cores (all services, shuffled mapping)
            for svc in service_names:
                set_cores(SERVICES[svc], cores_map[svc])

            # Apply quality (all services, route-constrained mapping)
            for svc in ROUTE:
                set_quality(SERVICES[svc], quality_map[svc])

            # 4) Stabilize, then measure window
            print(f"Stabilizing {STABILIZE_SEC}s…")
            time.sleep(STABILIZE_SEC)

            t_start = time.time()
            time.sleep(MEASURE_SEC)
            t_end = time.time()

            # 5) Query ALL Prometheus metric names for this window
            for mn in all_metric_names:
                try:
                    result = prom_query_range(mn, t_start, t_end, STEP, timeout=60)
                    if result:
                        append_results_csv(OUTPUT_CSV, run_tag, mn, result)
                except requests.exceptions.Timeout:
                    print(f"  [timeout] {mn}")
                except RuntimeError as e:
                    print(f"  [error] {mn}: {e}")

    print("\nDone:", OUTPUT_CSV)


if __name__ == "__main__":
    main()