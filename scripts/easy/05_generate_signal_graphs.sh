#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-docs/examples/signal_window_guide}"

python scripts/generate_signal_window_graphs.py --out "$OUT"
echo "signal/window visuals generated at: $OUT"
