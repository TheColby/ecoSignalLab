#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/easy/14_pipeline_run.sh <input_dir> [out_dir]"
  exit 1
fi

IN_DIR="$1"
OUT="${2:-out_pipeline}"

esl pipeline run "$IN_DIR" \
  --out "$OUT" \
  --plot \
  --interactive \
  --similarity-matrix \
  --novelty-matrix \
  --ml-export

esl pipeline status --manifest "$OUT/pipeline_manifest.json"

echo "pipeline outputs written under: $OUT"
