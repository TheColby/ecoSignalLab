#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/easy/02_analyze_and_plot.sh <input.wav> [out_dir]"
  exit 1
fi

IN="$1"
OUT="${2:-out}"

esl analyze "$IN" --out-dir "$OUT" --json "$OUT/$(basename "${IN%.*}").json" --plot
echo "analysis + plots written under: $OUT"
