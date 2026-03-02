#!/usr/bin/env bash
set -euo pipefail

DEVICE="${1:-auto}"
FRAMES="${2:-16384}"
FEATURES="${3:-256}"
ITERS="${4:-20}"
OUT="${5:-out/benchmark_device.json}"

mkdir -p "$(dirname "$OUT")"

esl benchmark device \
  --device "$DEVICE" \
  --frames "$FRAMES" \
  --features "$FEATURES" \
  --iters "$ITERS" \
  --json-out "$OUT"

echo "benchmark report written to: $OUT"
