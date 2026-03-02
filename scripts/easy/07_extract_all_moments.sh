#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/easy/07_extract_all_moments.sh <input.wav> [out_dir] [window_before_s] [window_after_s]"
  exit 1
fi

IN="$1"
OUT="${2:-out/moments_all}"
WIN_BEFORE="${3:-4}"
WIN_AFTER="${4:-8}"

esl moments extract "$IN" \
  --out "$OUT" \
  --all \
  --rank-metric novelty_curve \
  --window-before "$WIN_BEFORE" \
  --window-after "$WIN_AFTER"

echo "all detected moments written under: $OUT"
