#!/usr/bin/env bash
set -euo pipefail

AIR_QUALITY=""
WEATHER=""
TRI_FACILITIES=""
TRI_CHEMICALS=""
ZIP_SHP=""
ROADS_SHP=""
PLACE_SHP=""
OUTPUT_DIR=""

TIME_COL="time"
ZIP_COL="zip"

EXCLUDE_NORMALIZATION="time,zip"
EXCLUDE_VARIANCE="time,zip"
EXCLUDE_PCA="time,zip"

VARIANCE_THRESHOLD=""
PCA_RETAINED_VARIANCE=""

HILBERT_ORDER="8"
ROAD_RADIUS_KM="2.0"
FACILITY_RADIUS_KM="10.0"

TRAIN_FRACTION=""
VAL_FRACTION=""
TEST_FRACTION=""

DIRECTION_COLUMNS=""
AUTO_DETECT_DIRECTION_COLUMNS="1"
KEEP_ORIGINAL_DIRECTION_COLUMNS="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --air-quality) AIR_QUALITY="$2"; shift 2 ;;
    --weather) WEATHER="$2"; shift 2 ;;
    --tri-facilities) TRI_FACILITIES="$2"; shift 2 ;;
    --tri-chemicals) TRI_CHEMICALS="$2"; shift 2 ;;
    --zip-shapefile) ZIP_SHP="$2"; shift 2 ;;
    --roads-shapefile) ROADS_SHP="$2"; shift 2 ;;
    --place-shapefile) PLACE_SHP="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;

    --time-col) TIME_COL="$2"; shift 2 ;;
    --zip-col) ZIP_COL="$2"; shift 2 ;;

    --exclude-normalization) EXCLUDE_NORMALIZATION="$2"; shift 2 ;;
    --exclude-variance) EXCLUDE_VARIANCE="$2"; shift 2 ;;
    --exclude-pca) EXCLUDE_PCA="$2"; shift 2 ;;

    --variance-threshold) VARIANCE_THRESHOLD="$2"; shift 2 ;;
    --pca-retained-variance) PCA_RETAINED_VARIANCE="$2"; shift 2 ;;

    --hilbert-order) HILBERT_ORDER="$2"; shift 2 ;;
    --road-radius-km) ROAD_RADIUS_KM="$2"; shift 2 ;;
    --facility-radius-km) FACILITY_RADIUS_KM="$2"; shift 2 ;;

    --train-fraction) TRAIN_FRACTION="$2"; shift 2 ;;
    --val-fraction) VAL_FRACTION="$2"; shift 2 ;;
    --test-fraction) TEST_FRACTION="$2"; shift 2 ;;

    --direction-columns) DIRECTION_COLUMNS="$2"; shift 2 ;;
    --no-auto-detect-direction-columns) AUTO_DETECT_DIRECTION_COLUMNS="0"; shift 1 ;;
    --keep-original-direction-columns) KEEP_ORIGINAL_DIRECTION_COLUMNS="1"; shift 1 ;;

    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$AIR_QUALITY" || -z "$WEATHER" || -z "$TRI_FACILITIES" || -z "$TRI_CHEMICALS" || -z "$ZIP_SHP" || -z "$ROADS_SHP" || -z "$OUTPUT_DIR" || -z "$VARIANCE_THRESHOLD" || -z "$PCA_RETAINED_VARIANCE" || -z "$TRAIN_FRACTION" || -z "$VAL_FRACTION" || -z "$TEST_FRACTION" ]]; then
  echo "Missing required arguments." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXTRA_ARGS=()
if [[ "$AUTO_DETECT_DIRECTION_COLUMNS" == "0" ]]; then
  EXTRA_ARGS+=(--no-auto-detect-direction-columns)
fi
if [[ "$KEEP_ORIGINAL_DIRECTION_COLUMNS" == "1" ]]; then
  EXTRA_ARGS+=(--keep-original-direction-columns)
fi

python3 "$SCRIPT_DIR/preprocessing-pipeline.py" \
  --air-quality "$AIR_QUALITY" \
  --weather "$WEATHER" \
  --tri-facilities "$TRI_FACILITIES" \
  --tri-chemicals "$TRI_CHEMICALS" \
  --zip-shapefile "$ZIP_SHP" \
  --roads-shapefile "$ROADS_SHP" \
  --place-shapefile "$PLACE_SHP" \
  --output-dir "$OUTPUT_DIR" \
  --time-col "$TIME_COL" \
  --zip-col "$ZIP_COL" \
  --exclude-normalization "$EXCLUDE_NORMALIZATION" \
  --exclude-variance "$EXCLUDE_VARIANCE" \
  --exclude-pca "$EXCLUDE_PCA" \
  --variance-threshold "$VARIANCE_THRESHOLD" \
  --pca-retained-variance "$PCA_RETAINED_VARIANCE" \
  --hilbert-order "$HILBERT_ORDER" \
  --road-radius-km "$ROAD_RADIUS_KM" \
  --facility-radius-km "$FACILITY_RADIUS_KM" \
  --train-fraction "$TRAIN_FRACTION" \
  --val-fraction "$VAL_FRACTION" \
  --test-fraction "$TEST_FRACTION" \
  --direction-columns "$DIRECTION_COLUMNS" \
  "${EXTRA_ARGS[@]}"