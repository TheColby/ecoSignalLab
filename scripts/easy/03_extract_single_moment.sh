#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/easy/03_extract_single_moment.sh <input.wav> [out_dir]"
  exit 1
fi

IN="$1"
OUT="${2:-out/moments}"

esl moments extract "$IN" \
  --out "$OUT" \
  --single \
  --rank-metric novelty_curve \
  --event-window 8

echo "moments CSV + clip written under: $OUT"
