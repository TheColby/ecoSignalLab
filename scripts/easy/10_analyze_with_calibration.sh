#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: bash scripts/easy/10_analyze_with_calibration.sh <input.wav> <calibration.yaml|json> [out_dir]"
  exit 1
fi

IN="$1"
CAL="$2"
OUT="${3:-out_calibrated}"
STEM="$(basename "${IN%.*}")"

mkdir -p "$OUT"

esl analyze "$IN" \
  --calibration "$CAL" \
  --out-dir "$OUT" \
  --json "$OUT/${STEM}.json" \
  --csv "$OUT/${STEM}.csv" \
  --parquet "$OUT/${STEM}.parquet" \
  --hdf5 "$OUT/${STEM}.h5" \
  --mat "$OUT/${STEM}.mat" \
  --plot

echo "calibrated analysis outputs written under: $OUT"
