#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/easy/06_extract_topk_moments.sh <input.wav> [out_dir] [top_k] [event_window_s]"
  exit 1
fi

IN="$1"
OUT="${2:-out/moments_topk}"
TOP_K="${3:-5}"
WINDOW_S="${4:-10}"

esl moments extract "$IN" \
  --out "$OUT" \
  --top-k "$TOP_K" \
  --rank-metric novelty_curve \
  --event-window "$WINDOW_S"

echo "top-$TOP_K moments written under: $OUT"
