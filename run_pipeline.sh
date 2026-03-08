#!/usr/bin/env bash
set -euo pipefail

INPUT=""
FILTERED_CSV=""
NORMALIZED_CSV=""
NORMALIZATION_STATS_CSV=""
HILBERT_CSV=""
LAYOUT_CSV=""
TRAIN_OUT=""
VAL_OUT=""
TEST_OUT=""
FEATURES=""
RADIUS_SQUARED="1.0"
DTYPE="float32"
TRAIN_PCT="0.70"
VAL_PCT="0.15"
TEST_PCT="0.15"
HOUSTON_LAT="29.7604"
HOUSTON_LON="-95.3698"
CHUNKSIZE="200000"
WORKERS="3"
FILL_VALUE="nan"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT="$2"; shift 2 ;;
    --filtered-csv) FILTERED_CSV="$2"; shift 2 ;;
    --normalized-csv) NORMALIZED_CSV="$2"; shift 2 ;;
    --normalization-stats-csv) NORMALIZATION_STATS_CSV="$2"; shift 2 ;;
    --hilbert-csv) HILBERT_CSV="$2"; shift 2 ;;
    --layout-csv) LAYOUT_CSV="$2"; shift 2 ;;
    --train-out) TRAIN_OUT="$2"; shift 2 ;;
    --val-out) VAL_OUT="$2"; shift 2 ;;
    --test-out) TEST_OUT="$2"; shift 2 ;;
    --features) FEATURES="$2"; shift 2 ;;
    --radius-squared) RADIUS_SQUARED="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --train-pct) TRAIN_PCT="$2"; shift 2 ;;
    --val-pct) VAL_PCT="$2"; shift 2 ;;
    --test-pct) TEST_PCT="$2"; shift 2 ;;
    --houston-lat) HOUSTON_LAT="$2"; shift 2 ;;
    --houston-lon) HOUSTON_LON="$2"; shift 2 ;;
    --chunksize) CHUNKSIZE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --fill-value) FILL_VALUE="$2"; shift 2 ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

required=(
  INPUT FILTERED_CSV NORMALIZED_CSV NORMALIZATION_STATS_CSV HILBERT_CSV LAYOUT_CSV TRAIN_OUT VAL_OUT TEST_OUT
)
for var in "${required[@]}"; do
  if [[ -z "${!var}" ]]; then
    echo "Missing required argument for $var" >&2
    exit 1
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

feature_args=()
if [[ -n "$FEATURES" ]]; then
  IFS=',' read -r -a feature_array <<< "$FEATURES"
  feature_args=(--features "${feature_array[@]}")
fi

echo "[1/4] Filtering Houston-radius rows..."
python3 "$SCRIPT_DIR/01_filter_houston_radius.py" \
  --input "$INPUT" \
  --output "$FILTERED_CSV" \
  --radius-squared "$RADIUS_SQUARED" \
  --houston-lat "$HOUSTON_LAT" \
  --houston-lon "$HOUSTON_LON" \
  --chunksize "$CHUNKSIZE" \
  "${feature_args[@]}"

echo "[2/4] Normalizing numeric columns to [0,1]..."
python3 "$SCRIPT_DIR/02_normalize_csv.py" \
  --input "$FILTERED_CSV" \
  --output "$NORMALIZED_CSV" \
  --stats-out "$NORMALIZATION_STATS_CSV" \
  --dtype "$DTYPE" \
  --chunksize "$CHUNKSIZE" \
  "${feature_args[@]}"

echo "[3/4] Building dense Hilbert layout and sorted CSV..."
python3 "$SCRIPT_DIR/03_hilbert_sort_csv.py" \
  --input "$NORMALIZED_CSV" \
  --output "$HILBERT_CSV" \
  --layout-out "$LAYOUT_CSV" \
  "${feature_args[@]}"

echo "[4/4] Splitting timestamps into train/val/test tensors..."
python3 "$SCRIPT_DIR/04_split_to_tensors.py" \
  --input "$HILBERT_CSV" \
  --train-out "$TRAIN_OUT" \
  --val-out "$VAL_OUT" \
  --test-out "$TEST_OUT" \
  --train-pct "$TRAIN_PCT" \
  --val-pct "$VAL_PCT" \
  --test-pct "$TEST_PCT" \
  --dtype "$DTYPE" \
  --workers "$WORKERS" \
  --fill-value "$FILL_VALUE" \
  "${feature_args[@]}"

echo "Pipeline complete."
