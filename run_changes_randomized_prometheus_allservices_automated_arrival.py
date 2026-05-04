import time
import random
import csv
import json
from datetime import datetime
from pathlib import Path
import subprocess
import requests
import math

# ================================
#            CONFIG
# ================================

PROM = "http://localhost:9090"  # Prometheus

# 3 services
SERVICES = {
    "svc1": "http://localhost:8080",
    "svc2": "http://localhost:8081",
    "svc3": "http://localhost:8082",
}

# Fixed route order for monotone quality constraint
ROUTE = ["svc1", "svc2", "svc3"]

ENDPOINTS = {
    "cores": "/resource_scaling",   # PUT ?cores=N
    "quality": "/quality_scaling",  # PUT ?data_quality=Q
    "rps": "/change_rps",           # PUT ?client_id=...&rps=...
}

# Core constraints
TOTAL_CORES_BUDGET = 8
MIN_CORES_PER_SERVICE = 1
MAX_CORES_PER_SERVICE = 8

# Quality domain
QUALITIES = list(range(100, 1001, 100))

# Repeats per core triple
REPEATS_PER_CORE_TRIPLE = 10

# Experiment timing
STABILIZE_SEC = 30
MEASURE_SEC = 360  # 6 minutes

# Prometheus query step
STEP = "1s"

# Seeds file
SEEDS_FILE = Path("share/metrics/random_seeds.txt")

# Output directory
OUTPUT_DIR = Path("share/metrics")

# Docker compose restart config
DOCKER_COMPOSE_FILE = "docker-compose_chain.yml"
RESTART_BETWEEN_SEEDS = True
POST_RESTART_SLEEP_SEC = 10
SERVICE_READY_TIMEOUT_SEC = 120
SERVICE_READY_CHECK_INTERVAL_SEC = 2

# ================================
#       WORKLOAD PATTERN CONFIG
# ================================

# This must match the client name used by svc1 in docker-compose.
SOURCE_CLIENT_ID = "buffer"

# Global experiment start, used so workload continues across all runs.
GLOBAL_START_TIME = time.time()

# Patterns to include in the experiment
WORKLOAD_PATTERNS = [
    "unpredictable"
]
# WORKLOAD_PATTERNS = [
#     "static",
#     "periodic",
#     "once_in_a_lifetime",
#     "unpredictable",
# ]

# Static workload
STATIC_RPS = 250

# Periodic workload:
# - one full sine cycle every 50 minutes
# - new sampled workload value every 1 minute
# - sampled value is held constant until the next update
PERIODIC_LOW = 150
PERIODIC_HIGH = 350
PERIODIC_SINE_PERIOD_SEC = 3000   # 50 minutes = full cycle
PERIODIC_UPDATE_SEC = 60          # 1 minute between workload updates
PERIODIC_START_AT_PEAK = True     # True -> starts at 350, False -> starts at 250

# Once-in-a-lifetime workload:
# one single spike during the measurement window
OIAL_BASE = 200
OIAL_SPIKE = 600
OIAL_BEFORE_SEC = 2400      # 40 min baseline before spike
OIAL_SPIKE_SEC = 180        # 3 min spike

# Unpredictable workload: new random value every 10 minutes
UNPREDICTABLE_MIN = 100
UNPREDICTABLE_MAX = 500
UNPREDICTABLE_STEP_SEC = 600  # 10 minutes

# How often the controller checks whether the workload should change.
# Keep this smaller than PERIODIC_UPDATE_SEC so we do not miss update boundaries.
WORKLOAD_CONTROL_INTERVAL_SEC = 20

# ================================
#          HTTP HELPERS
# ================================

def _url(base: str, path: str) -> str:
    return f"{base}{path}"


def set_cores(service_base: str, n: int):
    """Set CPU cores for one service."""
    r = requests.put(
        _url(service_base, ENDPOINTS["cores"]),
        params={"cores": n},
        timeout=5
    )
    r.raise_for_status()


def set_quality(service_base: str, q: int):
    """Set data quality for one service."""
    r = requests.put(
        _url(service_base, ENDPOINTS["quality"]),
        params={"data_quality": q},
        timeout=5
    )
    r.raise_for_status()


def set_rps(service_base: str, client_id: str, rps: int):
    """Set arrival rate for one client of one service."""
    r = requests.put(
        _url(service_base, ENDPOINTS["rps"]),
        params={"client_id": client_id, "rps": rps},
        timeout=5
    )
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
                w.writerow([
                    run_tag,
                    metric_name,
                    float(ts),
                    val,
                    json.dumps(labels, sort_keys=True)
                ])


# ================================
#   CORE / QUALITY SAMPLING LOGIC
# ================================

def generate_valid_core_triples():
    """
    Generate ALL valid (c1,c2,c3) such that:
      - MIN_CORES_PER_SERVICE <= ci <= MAX_CORES_PER_SERVICE
      - c1 + c2 + c3 <= TOTAL_CORES_BUDGET
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
    Randomly permute which service gets which core value,
    to avoid correlation between service identity and core count.
    """
    shuffled = service_names[:]
    rng.shuffle(shuffled)
    return {svc: c for svc, c in zip(shuffled, pattern)}


def sample_monotone_qualities(rng: random.Random, qualities: list[int]):
    """
    Enforce route constraint:
      svc1 >= svc2 >= svc3
    """
    q1 = rng.choice(qualities)

    q2_raw = rng.choice(qualities)
    q2 = min(q1, q2_raw)

    q3_raw = rng.choice(qualities)
    q3 = min(q2, q3_raw)

    return q1, q2, q3


# ================================
#      SEED / OUTPUT HELPERS
# ================================

def load_seeds(path: Path):
    """
    Read integer random seeds from a text file.
    One seed per line. Blank lines and lines starting with # are ignored.
    """
    if not path.exists():
        raise FileNotFoundError(f"Seeds file not found: {path}")

    seeds = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            seeds.append(int(line))

    if not seeds:
        raise ValueError(f"No valid seeds found in: {path}")

    return seeds


def make_output_csv(seed: int) -> Path:
    """
    Create output filename that includes experiment time and random seed.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"prom_dump_all_{timestamp}_seed{seed}.csv"


# ================================
#      DOCKER / READINESS HELPERS
# ================================

def wait_for_services(timeout=SERVICE_READY_TIMEOUT_SEC, check_interval=SERVICE_READY_CHECK_INTERVAL_SEC):
    """
    Wait until all service endpoints are reachable after container restart.
    """
    print("Waiting for services to become ready...")

    deadline = time.time() + timeout

    while time.time() < deadline:
        all_ready = True

        for svc, base_url in SERVICES.items():
            try:
                probe_url = _url(base_url, ENDPOINTS["cores"])
                r = requests.get(probe_url, timeout=3)

                # We only care that the service is alive and answering.
                if r.status_code not in (200, 400, 405, 422):
                    all_ready = False
                    print(f"  {svc} not ready yet: status {r.status_code} at {probe_url}")
                    break

            except requests.RequestException as e:
                all_ready = False
                print(f"  {svc} not reachable yet: {e}")
                break

        if all_ready:
            print("All services are reachable.")
            return

        time.sleep(check_interval)

    raise TimeoutError("Services did not become ready in time after container restart.")


def restart_containers():
    """
    Restart docker compose services between seed experiments.
    """
    print("\nRestarting containers...")

    subprocess.run(
        ["docker", "compose", "-f", DOCKER_COMPOSE_FILE, "down"],
        check=True
    )

    subprocess.run(
        ["docker", "compose", "-f", DOCKER_COMPOSE_FILE, "up", "-d"],
        check=True
    )

    print(f"Containers restarted. Waiting {POST_RESTART_SLEEP_SEC}s before readiness checks...")
    time.sleep(POST_RESTART_SLEEP_SEC)

    wait_for_services()


# ================================
#       WORKLOAD PATTERN LOGIC
# ================================

def workload_rps(pattern_name: str, elapsed_sec: float, seed: int):
    """
    Return the source arrival rate for the current elapsed time.
    The source is always svc1.

    For periodic:
      - one full sine cycle every PERIODIC_SINE_PERIOD_SEC
      - sampled every PERIODIC_UPDATE_SEC
      - held constant between updates
    """
    if pattern_name == "static":
        return STATIC_RPS

    elif pattern_name == "periodic":
        mean = (PERIODIC_LOW + PERIODIC_HIGH) / 2
        amplitude = (PERIODIC_HIGH - PERIODIC_LOW) / 2

        # Quantize time so the workload changes only every PERIODIC_UPDATE_SEC.
        sample_index = int(elapsed_sec // PERIODIC_UPDATE_SEC)
        sampled_time = sample_index * PERIODIC_UPDATE_SEC

        angle = 2 * math.pi * sampled_time / PERIODIC_SINE_PERIOD_SEC

        if PERIODIC_START_AT_PEAK:
            # Starts at max value (e.g. 350) when sampled_time = 0
            rps = mean + amplitude * math.cos(angle)
        else:
            # Starts at midpoint (e.g. 250) when sampled_time = 0
            rps = mean + amplitude * math.sin(angle)

        # Clamp and round for safety
        rps_int = int(round(rps))
        return max(PERIODIC_LOW, min(PERIODIC_HIGH, rps_int))

    elif pattern_name == "once_in_a_lifetime":
        if OIAL_BEFORE_SEC <= elapsed_sec < (OIAL_BEFORE_SEC + OIAL_SPIKE_SEC):
            return OIAL_SPIKE
        return OIAL_BASE

    elif pattern_name == "unpredictable":
        slot = int(elapsed_sec // UNPREDICTABLE_STEP_SEC)
        local_rng = random.Random(seed * 100000 + slot)
        return local_rng.randint(UNPREDICTABLE_MIN, UNPREDICTABLE_MAX)

    else:
        raise ValueError(f"Unknown workload pattern: {pattern_name}")


def run_workload_pattern(pattern_name: str, measure_sec: int, seed: int, control_interval_sec: int = WORKLOAD_CONTROL_INTERVAL_SEC):
    """
    Apply the selected workload pattern to svc1 during the full measurement window.
    Only send /change_rps when the value actually changes.

    The workload is driven by GLOBAL_START_TIME so it continues smoothly
    across all experiment runs.
    """
    run_start = time.time()
    last_rps = None

    while True:
        now = time.time()
        run_elapsed = now - run_start
        if run_elapsed >= measure_sec:
            break

        global_elapsed = now - GLOBAL_START_TIME
        rps = workload_rps(pattern_name, global_elapsed, seed)

        if rps != last_rps:
            print(f"  Updating RPS -> {rps} (pattern={pattern_name}, global_elapsed={global_elapsed:.1f}s)")
            set_rps(SERVICES["svc1"], SOURCE_CLIENT_ID, rps)
            last_rps = rps

        remaining = measure_sec - run_elapsed
        sleep_for = min(control_interval_sec, max(0.0, remaining))
        if sleep_for > 0:
            time.sleep(sleep_for)


# ================================
#        RUN ONE EXPERIMENT
# ================================

def run_experiment(seed: int):
    # Basic config safety checks
    if set(ROUTE) != set(SERVICES.keys()):
        raise ValueError("ROUTE must contain exactly the same service names as SERVICES (just ordered).")
    if len(ROUTE) != 3:
        raise ValueError("This script is written for exactly 3 services (svc1->svc2->svc3).")

    rng = random.Random(seed)
    service_names = list(SERVICES.keys())

    # Build output file for this seed
    output_csv = make_output_csv(seed)

    # Build full list of valid core triples, and randomize their order
    core_triples = generate_valid_core_triples()
    rng.shuffle(core_triples)

    # Fresh output
    if output_csv.exists():
        output_csv.unlink()

    # Snapshot metric names once
    all_metric_names = prom_metric_names()

    total_runs = len(core_triples) * REPEATS_PER_CORE_TRIPLE * len(WORKLOAD_PATTERNS)

    run_idx = 0
    for core_pattern in core_triples:
        for rep in range(REPEATS_PER_CORE_TRIPLE):
            for workload_pattern in WORKLOAD_PATTERNS:
                run_idx += 1

                # 1) Assign cores to services with shuffle
                cores_map = randomize_core_assignment(rng, core_pattern, service_names)

                # Total-core safety check
                if sum(cores_map.values()) > TOTAL_CORES_BUDGET:
                    raise RuntimeError("BUG: total core budget violated.")

                # 2) Sample route-constrained quality triple (svc1 >= svc2 >= svc3)
                q1, q2, q3 = sample_monotone_qualities(rng, QUALITIES)
                quality_map = {ROUTE[0]: q1, ROUTE[1]: q2, ROUTE[2]: q3}

                # Safety check for monotone property
                if not (quality_map[ROUTE[0]] >= quality_map[ROUTE[1]] >= quality_map[ROUTE[2]]):
                    raise RuntimeError("BUG: quality is not monotone non-increasing along route.")

                # Tag includes exact per-service cores + per-service quality + workload pattern
                cores_tag = ",".join(f"{svc}={cores_map[svc]}" for svc in sorted(service_names))
                dq_tag = ",".join(f"{svc}={quality_map[svc]}" for svc in ROUTE)
                run_tag = f"seed-{seed}_workload[{workload_pattern}]_cores[{cores_tag}]_dq[{dq_tag}]_rep-{rep}"

                print(f"\n[{run_idx}/{total_runs}] {run_tag} (total_cores={sum(cores_map.values())})")

                # 3) Apply settings to ALL services
                for svc in service_names:
                    set_cores(SERVICES[svc], cores_map[svc])

                for svc in ROUTE:
                    set_quality(SERVICES[svc], quality_map[svc])

                # 4) Set initial source RPS based on GLOBAL elapsed time
                initial_elapsed = time.time() - GLOBAL_START_TIME
                initial_rps = workload_rps(workload_pattern, initial_elapsed, seed)
                print(f"  Initial RPS -> {initial_rps} (global_elapsed={initial_elapsed:.1f}s)")
                set_rps(SERVICES["svc1"], SOURCE_CLIENT_ID, initial_rps)

                # 5) Stabilize
                print(f"  Stabilizing {STABILIZE_SEC}s...")
                time.sleep(STABILIZE_SEC)

                # 6) Measure window while workload pattern is applied
                t_start = time.time()
                run_workload_pattern(workload_pattern, MEASURE_SEC, seed)
                t_end = time.time()

                # 7) Query ALL Prometheus metric names for this window
                for mn in all_metric_names:
                    try:
                        result = prom_query_range(mn, t_start, t_end, STEP, timeout=60)
                        if result:
                            append_results_csv(output_csv, run_tag, mn, result)
                    except requests.exceptions.Timeout:
                        print(f"  [timeout] {mn}")
                    except RuntimeError as e:
                        print(f"  [error] {mn}: {e}")

    print(f"\nDone: {output_csv}")


# ================================
#              MAIN
# ================================

def main():
    global GLOBAL_START_TIME
    GLOBAL_START_TIME = time.time()

    seeds = load_seeds(SEEDS_FILE)
    print(f"Loaded seeds: {seeds}")
    print(f"Global experiment start time: {GLOBAL_START_TIME}")

    for seed in seeds:
        print("\n===============================")
        print(f"Starting experiment with seed {seed}")
        print("===============================")

        if RESTART_BETWEEN_SEEDS:
            restart_containers()

        run_experiment(seed)

    print("\nAll seeds completed. Stopping containers...")

    subprocess.run(
        ["docker", "compose", "-f", DOCKER_COMPOSE_FILE, "down"],
        check=True
    )


if __name__ == "__main__":
    main()