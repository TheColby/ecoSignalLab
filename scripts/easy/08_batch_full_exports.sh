#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: bash scripts/easy/08_batch_full_exports.sh <input_dir> [out_dir]"
  exit 1
fi

IN_DIR="$1"
OUT="${2:-out_batch}"

esl batch "$IN_DIR" \
  --out "$OUT" \
  --metrics rms_dbfs,snr_db,spl_a_db,novelty_curve,ndsi,rt60_s \
  --report-metrics snr_db,spl_a_db,novelty_curve,ndsi,rt60_s \
  --csv --parquet --hdf5 --mat \
  --plot

echo "batch outputs written under: $OUT"
