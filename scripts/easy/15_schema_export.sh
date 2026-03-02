#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-out/esl_schema.json}"

mkdir -p "$(dirname "$OUT")"
esl schema --out "$OUT"
echo "schema written to: $OUT"
