#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------
# Export ALL (non-internal) Prometheus metrics to CSV files
# and then merge them into ONE wide CSV:
#   <OUT_DIR>/prom_all_metrics_wide.csv
#
# Usage examples:
#
#   # 1) Last 10 minutes, default step=15s, output to prom_export/
#   ./export_prom_all_metrics.sh
#
#   # 2) Specific window (UTC), step 1s, output to prom_all_window_1/
#   ./export_prom_all_metrics.sh \
#       '2025-11-11T15:16:31Z' \
#       '2025-11-11T15:37:22Z' \
#       '1s' \
#       'prom_all_window_1'
#
#   # 3) Same as (2), but only a specific job or container
#   SELECTOR='{job="iot_service"}' \
#   ./export_prom_all_metrics.sh '2025-11-11T15:16:31Z' '2025-11-11T15:37:22Z' '1s' 'prom_all_window_1'
#
# Args:
#   $1 = START (RFC3339, e.g. 2025-11-11T15:16:31Z)   [optional]
#   $2 = END   (RFC3339, e.g. 2025-11-11T15:37:22Z)   [optional]
#   $3 = STEP  (e.g. 400ms, 1s, 5s, 15s; default 15s) [optional]
#   $4 = OUT_DIR (e.g. prom_all_window_1; default prom_export) [optional]
#
# Env:
#   PROM     = Prometheus base URL (default http://localhost:9090)
#   SELECTOR = optional label selector, e.g. '{job="iot_service"}'
#             or '{container_id="elastic-workbench-linked-service-1"}'
# ----------------------------------------------------------

PROM="${PROM:-http://localhost:9090}"     # Prometheus base URL
STEP="${3:-15s}"                          # sample step
OUT_DIR="${4:-prom_export}"               # output directory
SELECTOR="${SELECTOR:-}"                  # optional label selector

# ---------------- ISO8601 helpers (macOS + Linux) ----------------
iso_now() {
  if date -u +%FT%TZ >/dev/null 2>&1; then
    date -u +%FT%TZ              # works on macOS and GNU
  else
    date -u +"%Y-%m-%dT%H:%M:%SZ"
  fi
}

iso_minus_minutes() {
  local mins="${1:-10}"
  if date -u -v-"${mins}"M +%FT%TZ >/dev/null 2>&1; then
    # macOS
    date -u -v-"${mins}"M +%FT%TZ
  else
    # GNU date
    date -u -d "-${mins} minutes" +%FT%TZ
  fi
}
# -----------------------------------------------------------------

# Start/end default to "last 10 minutes" unless args 1 & 2 provided
START="${1:-$(iso_minus_minutes 10)}"
END="${2:-$(iso_now)}"

mkdir -p "$OUT_DIR"

echo "Exporting ALL (non-internal) metrics from $PROM"
echo "  Window: [$START → $END], step=$STEP"
if [[ -n "$SELECTOR" ]]; then
  echo "  Selector: $SELECTOR"
else
  echo "  Selector: (none, exporting all series for each metric)"
fi

# ----------------------------------------------------------
# 1) Fetch all metric names from Prometheus
# ----------------------------------------------------------
NAMES_FILE="$OUT_DIR/_metric_names.txt"
FILTERED="$OUT_DIR/_metric_names.filtered.txt"

curl -sG "$PROM/api/v1/label/__name__/values" \
  | jq -r '.data[]' > "$NAMES_FILE"

# If you REALLY want literally everything (including Prom internals),
# comment out this grep and set METRIC_LIST="$NAMES_FILE".
grep -Ev '^(go_|promhttp_|prometheus_|process_|scrape_|grpc_|grpcio_|tsdb_|otel_).*' "$NAMES_FILE" \
  > "$FILTERED" || true

METRIC_LIST="$FILTERED"
# To include 100% of metrics including Prometheus internals, use:
# METRIC_LIST="$NAMES_FILE"

# ----------------------------------------------------------
# Helper: build query string for each metric
# - attaches SELECTOR if provided
# - wraps *_total / *_seconds_total in rate() to get time-derivatives
# ----------------------------------------------------------
build_query() {
  local metric="$1"
  local base
  if [[ -n "$SELECTOR" ]]; then
    base="${metric}${SELECTOR}"
  else
    base="${metric}"
  fi

  # Auto-derive rate for common counter suffixes
  if [[ "$metric" =~ (_total$|_seconds_total$) ]]; then
    echo "rate(${base}[1m])"
  else
    echo "${base}"
  fi
}

# ----------------------------------------------------------
# 2) Export each metric (one CSV per time series)
#    Filename: <metric>__<some_labels>.csv
#    Columns: metric,labels,timestamp,value
# ----------------------------------------------------------
while IFS= read -r METRIC; do
  [[ -z "$METRIC" ]] && continue

  QUERY="$(build_query "$METRIC")"
  echo "→ Querying metric: $METRIC"
  echo "    PromQL: $QUERY"

  RESP="$(curl -sG "$PROM/api/v1/query_range" \
            --data-urlencode "query=$QUERY" \
            --data-urlencode "start=$START" \
            --data-urlencode "end=$END" \
            --data-urlencode "step=$STEP")"

  STATUS="$(echo "$RESP" | jq -r '.status // empty')"
  if [[ "$STATUS" != "success" ]]; then
    echo "   (query failed or Prometheus returned non-success, skipping)"
    continue
  fi

  COUNT="$(echo "$RESP" | jq '.data.result | length')"
  if [[ "$COUNT" -eq 0 ]]; then
    echo "   (no series found in interval for $METRIC)"
    continue
  fi

  # For each returned series (distinct label set)
  echo "$RESP" | jq -c '.data.result[]' | while read -r SERIES; do
    LABELS="$(echo "$SERIES" | jq -r '.metric | to_entries | map("\(.key)=\(.value)") | sort | join(";")')"

    # Make a compact, filesystem-safe label suffix for filename
    SAFE_LBL="$(echo "$LABELS" \
      | awk -F';' 'BEGIN{OFS=";"}{keep=""}
                   {for(i=1;i<=NF;i++){
                      # keep some relevant labels if present
                      if($i ~ /^(container=|container_id=|name=|instance=|job=|service_type=)/)
                        keep=(keep?keep";"$i:$i)
                    }}
                   END{print keep}' \
      | tr '/:|*?[]{}()<>, ="' '_' )"

    [[ -z "$SAFE_LBL" ]] && SAFE_LBL="nolabels"

    SAFE_METRIC="$(echo "$METRIC" | tr '/:|*?[]{}()<>, ="' '_' )"
    OUT_CSV="$OUT_DIR/${SAFE_METRIC}__${SAFE_LBL}.csv"

    # Header once per file
    if [[ ! -f "$OUT_CSV" ]]; then
      echo "metric,labels,timestamp,value" > "$OUT_CSV"
    fi

    # Append time series rows
    echo "$SERIES" | jq -r --arg m "$METRIC" --arg lbl "$LABELS" '
      .values[] | [$m, $lbl, (.[0] | todateiso8601), .[1]] | @csv
    ' >> "$OUT_CSV"
  done

done < "$METRIC_LIST"

echo "Per-metric CSVs written to: $OUT_DIR"

# ----------------------------------------------------------
# 3) Merge all per-metric CSVs into ONE wide CSV
#    Minimal naming: one column per metric name.
#    Assumes effectively one time series per metric
#    (after any SELECTOR you used).
# ----------------------------------------------------------

OUT_WIDE="$OUT_DIR/prom_all_metrics_wide.csv"
echo "Merging all series into wide table: $OUT_WIDE"

OUT_DIR_ENV="$OUT_DIR" OUT_WIDE_ENV="$OUT_WIDE" python3 - << 'PY'
import os, csv, collections

out_dir = os.environ["OUT_DIR_ENV"]
out_wide = os.environ["OUT_WIDE_ENV"]

# timestamp -> { metric_name -> value }
rows: dict[str, dict[str, str]] = collections.OrderedDict()
metrics_set: set[str] = set()

for fname in os.listdir(out_dir):
    if not fname.endswith(".csv"):
        continue
    if fname.startswith("_"):
        # skip helper files like _metric_names.txt
        continue

    path = os.path.join(out_dir, fname)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            continue

        # Expect header: metric,labels,timestamp,value
        try:
            metric_idx = header.index("metric")
            ts_idx     = header.index("timestamp")
            val_idx    = header.index("value")
        except ValueError:
            # Unexpected format; skip
            continue

        for row in reader:
            if len(row) <= max(metric_idx, ts_idx, val_idx):
                continue
            metric = row[metric_idx]
            ts     = row[ts_idx]
            val    = row[val_idx]

            metrics_set.add(metric)
            bucket = rows.setdefault(ts, {})
            # If there happen to be multiple series for same metric+ts,
            # we just overwrite; in your "minimal one-service" setup,
            # this shouldn't happen.
            bucket[metric] = val

if not rows:
    raise SystemExit("No data read from per-metric CSVs; nothing to merge.")

metrics = sorted(metrics_set)
timestamps = sorted(rows.keys())

with open(out_wide, "w", newline="") as f:
    writer = csv.writer(f)
    # Header: timestamp, <metric1>, <metric2>, ...
    writer.writerow(["timestamp"] + metrics)
    for ts in timestamps:
        bucket = rows.get(ts, {})
        writer.writerow([ts] + [bucket.get(m, "") for m in metrics])

print(f"[PY] Wrote wide CSV with {len(timestamps)} rows and {len(metrics)} metrics: {out_wide}")
PY

echo "Done."
