# ARCHITECTURE

## Package layout

- `src/esl/core`
  - analysis orchestration, audio I/O, calibration, context, config
- `src/esl/metrics`
  - metric contracts, helpers, built-ins, registry
- `src/esl/io`
  - export writers (`json`, `csv`, `parquet`, `hdf5`, `mat`, industrial CSV)
- `src/esl/viz`
  - static PNG and optional Plotly interactive plotting
- `src/esl/ml`
  - frame/clip feature export and optional anomaly scores
- `src/esl/ingest`
  - online ingestion connectors
- `src/esl/project`
  - project/variant index for design comparisons
- `src/esl/schema`
  - output schema definition
- `src/esl/pipeline`
  - staged CLI pipeline runner and status manifest
- `src/esl/cli`
  - command-line entrypoint

## Runtime flow

1. CLI parses arguments into `AnalysisConfig`.
2. Audio loader decodes via `soundfile` or `ffmpeg` fallback.
3. `AnalysisContext` carries signal, calibration, and run config.
4. `MetricRegistry` computes selected plugins.
5. Result document assembled with provenance and metric specs.
6. Optional exports/plots/ML artifacts/project indexing are emitted.

## Staged pipeline flow

`esl pipeline run` executes explicit stages:
- `analyze`
- `plot` (optional)
- `ml_export` (optional)
- `digest`

Each run writes `pipeline_manifest.json` with per-stage status, timing, counts, and errors.

## Data model

Top-level result:
- provenance (`esl_version`, `analysis_time_utc`, `config_hash`)
- metadata (input, decode backend, calibration assumptions)
- metrics map with series/summary/confidence/spec

## Architectural acoustics path

IR-style metrics (`rt60`, `edt`, `c50`, `c80`, `d50`) are derived from energy decay via Schroeder integration with explicit confidence estimation.

## Reproducibility guarantees

- Deterministic seed capture
- Config hashing
- Version stamping
- Assumption logging
