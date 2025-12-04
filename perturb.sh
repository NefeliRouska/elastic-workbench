#!/usr/bin/env bash
set -euo pipefail

GATEWAY="${GATEWAY:-http://localhost:8080}"
SLEEP="${SLEEP:-3}"     # seconds between requests (>= cooldown ~2s)
LOG="${LOG:-perturb.log}"

say() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }

put() {
  local path="$1"
  say "PUT ${GATEWAY}${path}"
  # -sS silent but show errors, --fail makes non-2xx exit nonzero
  curl -sS --fail -X PUT "${GATEWAY}${path}" || {
    say "ERROR on ${path}"; return 1; }
  sleep "$SLEEP"
}

# ---------- SCENARIOS ----------

# 1) Ramp cores up/down (resource_scaling)
ramp_cores() {
  local values=(2 3 4 3 2)
  for v in "${values[@]}"; do
    put "/resource_scaling?cores=${v}"
  done
}

# 2) Step the input difficulty (quality_scaling)
ramp_quality() {
  # Typical range per your JSON: 100–1000 (pick what you need)
  local values=(200 500 800 500 200)
  for q in "${values[@]}"; do
    put "/quality_scaling?data_quality=${q}"
  done
}

# 3) Change model size (only services that support it)
bump_model_size() {
  # Example range 1–4 (per your cv-analyzer config)
  local values=(1 2 3 2 1)
  for m in "${values[@]}"; do
    put "/model_scaling?model_size=${m}"
  done
}

# 4) Parallelism (if enabled in your config)
ramp_parallelism() {
  local values=(1 2 4 2 1)
  for p in "${values[@]}"; do
    put "/parallelism_scaling?parallelism=${p}"
  done
}

# 5) Mixed perturbation: cores + data_quality together (queue stress)
stress_mix() {
  local cores=(2 3 3 4 2)
  local qual=(300 600 900 600 300)
  for i in "${!cores[@]}"; do
    put "/resource_scaling?cores=${cores[$i]}"
    put "/quality_scaling?data_quality=${qual[$i]}"
  done
}

# ---------- RUN PLAN ----------
# Choose one or stack them. Comment/uncomment as needed.

say "Starting perturbations (sleep=${SLEEP}s, gateway=${GATEWAY})"

ramp_cores
ramp_quality
# bump_model_size
# ramp_parallelism
# stress_mix

say "Done."

#how to use: 
#cd ~/Documents/elastic-workbench
#chmod +x perturb.sh
# Run with defaults (gateway http://localhost:8080, 3s sleep)
#./perturb.sh
# Customize (e.g., longer delay, different gateway)
#SLEEP=5 GATEWAY=http://127.0.0.1:8080 ./perturb.sh




