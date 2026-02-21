#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/easy/04_compare_kpis.sh <input.wav> [out_dir] [stretch_factor]"
  echo "example: bash scripts/easy/04_compare_kpis.sh input.wav out/kpi_compare 2.0"
  exit 1
fi

IN="$1"
OUT="${2:-out/kpi_compare}"
FACTOR="${3:-2.0}"

python scripts/compare_time_stretch_kpis.py \
  --input "$IN" \
  --out-dir "$OUT" \
  --factor "$FACTOR"

echo "kpi comparison written to: $OUT"
