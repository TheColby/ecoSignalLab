#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/easy/17_extract_features_all.sh <input.wav> [out_vectors.npz] [meta.json]"
  exit 1
fi

IN="$1"
OUT_VEC="${2:-out/features/vectors.npz}"
OUT_META="${3:-out/features/vectors_meta.json}"

mkdir -p "$(dirname "$OUT_VEC")"
mkdir -p "$(dirname "$OUT_META")"

esl features extract "$IN" \
  --out "$OUT_VEC" \
  --feature-set all \
  --meta-json "$OUT_META"

echo "feature vectors written to: $OUT_VEC"
echo "feature metadata written to: $OUT_META"
