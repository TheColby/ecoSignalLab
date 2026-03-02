#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/easy/16_24h_ambisonic_moments.sh <input_24h.wav> [out_dir] [chunk_size] [top_k]"
  exit 1
fi

IN="$1"
OUT="${2:-out/24h_moments}"
CHUNK_SIZE="${3:-2880000}"
TOP_K="${4:-12}"

esl moments extract "$IN" \
  --out "$OUT" \
  --metrics novelty_curve,spectral_change_detection,isolation_forest_score,spl_a_db \
  --rank-metric novelty_curve \
  --chunk-size "$CHUNK_SIZE" \
  --sample-rate 96000 \
  --top-k "$TOP_K" \
  --window-before 5 \
  --window-after 7 \
  --merge-gap 2

echo "24h moments outputs written under: $OUT"
