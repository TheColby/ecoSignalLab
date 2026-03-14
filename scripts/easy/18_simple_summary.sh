#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/easy/18_simple_summary.sh <input.wav>"
  exit 1
fi

IN="$1"
esl simple "$IN"
