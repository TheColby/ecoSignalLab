#!/usr/bin/env bash
set -euo pipefail

IN="${1:-}"

if command -v esl >/dev/null 2>&1; then
  if [[ -n "$IN" ]]; then
    esl doctor "$IN"
  else
    esl doctor
  fi
  exit 0
fi

if [[ -x ".venv/bin/python" ]]; then
  if [[ -n "$IN" ]]; then
    .venv/bin/python -m esl doctor "$IN"
  else
    .venv/bin/python -m esl doctor
  fi
  exit 0
fi

echo "esl is not installed yet."
echo "Run:"
echo "  python -m venv .venv"
echo "  source .venv/bin/activate"
echo "  pip install -e ."
exit 1
