#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: bash scripts/easy/12_calibration_check.sh <tone.wav> <calibration.yaml|json> [out.json]"
  exit 1
fi

TONE="$1"
CAL="$2"
OUT="${3:-out/calibration_check.json}"

mkdir -p "$(dirname "$OUT")"

esl calibrate check \
  --tone "$TONE" \
  --calibration "$CAL" \
  --out "$OUT"

echo "calibration report written to: $OUT"
