#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: bash scripts/easy/09_similarity_search.sh <query.wav> <corpus_dir> [out_dir] [top_k]"
  exit 1
fi

QUERY="$1"
CORPUS="$2"
OUT="${3:-out/similarity}"
TOP_K="${4:-10}"

mkdir -p "$OUT"

esl similar "$QUERY" "$CORPUS" \
  --mode auto \
  --distance cosine \
  --feature-set all \
  --top-k "$TOP_K" \
  --json "$OUT/similarity.json" \
  --csv "$OUT/similarity.csv"

echo "similarity outputs written under: $OUT"
