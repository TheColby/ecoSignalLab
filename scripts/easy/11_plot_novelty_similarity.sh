#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/easy/11_plot_novelty_similarity.sh <results.json> [out_dir] [audio.wav]"
  exit 1
fi

JSON_IN="$1"
OUT="${2:-out/plots_matrices}"
AUDIO="${3:-}"

if [[ -n "$AUDIO" ]]; then
  esl plot "$JSON_IN" \
    --out "$OUT" \
    --interactive \
    --audio "$AUDIO" \
    --similarity-matrix \
    --novelty-matrix
else
  esl plot "$JSON_IN" \
    --out "$OUT" \
    --interactive \
    --similarity-matrix \
    --novelty-matrix
fi

echo "novelty/similarity plots written under: $OUT"
