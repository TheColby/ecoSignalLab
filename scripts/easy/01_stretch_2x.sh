#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/easy/01_stretch_2x.sh <input.wav> [output.wav]"
  exit 1
fi

IN="$1"
OUT="${2:-output_2x.wav}"

ffmpeg -y -i "$IN" -filter:a "atempo=0.5" "$OUT"
echo "wrote: $OUT"
