# ecoSignalLab (`esl`)

`esl` is an open-source, production-oriented acoustic analytics SDK for environmental, architectural, and industrial audio workflows.

Core goals:
- Multi-channel native (`1..N`) with ambisonic-compatible handling
- Calibration-aware (`dBFS <-> SPL`, `dBA`, `dBC`, `dBZ`)
- ML-ready feature export (NumPy, PyTorch, Hugging Face)
- Architectural simulation compatible (IR metrics and project variants)
- Industrial measurement interoperability (HEAD/APx/SoundCheck CSV mappings)
- Reproducible by default (seed, config hash, version stamping)
- Plugin-extensible metric engine

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e .[dev,ml,plot,io]
```

System dependency for compressed audio decoding:
- `ffmpeg` / `ffprobe` available on PATH.

## CLI

### Analyze one file

```bash
esl analyze file.wav \
  --json output.json \
  --csv output.csv \
  --parquet output.parquet \
  --hdf5 output.h5 \
  --calibration calibration.yaml \
  --verbosity 2 \
  --debug 1 \
  --plot \
  --show \
  --ml-export \
  --project restaurant_design \
  --variant designA
```

### Batch mode

```bash
esl batch input_dir --out out_dir --csv --parquet --hdf5
```

### Staged pipeline mode (CLI-first)

```bash
esl pipeline run input_dir --out out_dir --plot --interactive --show --ml-export
esl pipeline status --manifest out_dir/pipeline_manifest.json
```

Pipeline mode persists:
- `pipeline_manifest.json` (stage status, timing, counts, errors)
- `pipeline_analysis_index.json` (analysis artifact index)
- `pipeline_digest.csv` and `pipeline_digest.json` (dataset-level summary)

### Plot from existing JSON

```bash
esl plot results.json --out plots --interactive
```

Plot controls:
- `--metrics spl_a_db,snr_db,novelty_curve`
- `--no-spectral` (skip spectrogram/mel/log/waterfall/LTSA suite)
- `--similarity-matrix` (generate self-similarity matrix plot)
- `--novelty-matrix` (generate Foote-style novelty matrix plot)
- `--show --show-limit 12` (spawn generated plots via system viewer/browser)

### Ingest datasets

```bash
esl ingest --source freesound --query "restaurant ambience" --limit 50 --out ingest_data
```

```bash
esl ingest --source huggingface --query "speechcolab/gigaspeech" --limit 10 --out ingest_data
```

### Print JSON schema

```bash
esl schema
```

## Input format support

Native (`soundfile`):
- WAV
- FLAC
- AIFF/AIF
- RF64
- CAF

Compressed (`ffmpeg` fallback):
- MP3
- AAC
- OGG
- Opus
- WMA
- ALAC
- M4A

Spatial:
- Ambisonic-compatible multichannel WAV workflows
- SOFA IR import (HDF5-based, first measurement decode)

Large files:
- Chunked mode via `--chunk-size`.

## Built-in metric families

`esl` now ships **74 built-in metrics** by default:

- Basic + QC: RMS/peak, clipping ratio/events, DC offset, dropout/silence, uptime/completeness/diurnal/site-comparability proxies
- Level + loudness: SPL A/C/Z, Leq/Lmax/Lmin/L10/L50/L90/L95, SEL/LAE, crest factor, LUFS (integrated/short-term/momentary), LRA, true peak, calibration drift
- Noise + SNR: robust percentile SNR estimator
- Spectral: centroid, bandwidth, flatness, rolloff, octave and third-octave band levels
- Temporal: zero crossing rate
- Ecoacoustics: bioacoustic index, ACI, NDSI, ADI, AEI, acoustic entropy, eco-octave trends
- Spatial + ambisonic: interchannel coherence, IACC/ILD/IPD/ITD, DOA azimuth proxy, FOA diffuseness and energy-vector azimuth/elevation
- Architectural acoustics + intelligibility: RT60, EDT, T20, T30, C50, C80, D50, Ts, G-strength proxy, LF/LFC proxies, bass ratio, STI proxy
- Anomaly + novelty: novelty curve, spectral change z-score, isolation-forest score, one-class-SVM score, autoencoder reconstruction-error proxy, change-point confidence

See full definitions in [`docs/METRICS_REFERENCE.md`](/Users/cleider/dev/ecoSignalLab/docs/METRICS_REFERENCE.md), including mathematical equations and plain-English interpretation for every metric ID.

## Reproducibility

Each run emits:
- `config_hash` (SHA-256 over canonicalized config)
- `esl_version`
- UTC timestamp
- local timestamp
- seed value
- runtime fingerprint (python/platform/hostname)
- channel layout hint
- calibration assumptions

## Calibration model

Calibration file supports:
- `dbfs_reference`
- `spl_reference_db`
- `mic_sensitivity_mv_pa`
- `weighting`: `A|C|Z`
- `calibration_tone_file`

Example: [`examples/calibration.yaml`](/Users/cleider/dev/ecoSignalLab/examples/calibration.yaml)

## Project mode

Project/variant analysis for design comparison:

```bash
esl analyze A.wav --project restaurant_design --variant A --out-dir out
esl analyze B.wav --project restaurant_design --variant B --out-dir out
```

Outputs:
- `out/projects/restaurant_design/index.json`
- `out/projects/restaurant_design/comparison.csv`

## Documentation

- System design: [`DESIGN.md`](/Users/cleider/dev/ecoSignalLab/DESIGN.md)
- Architecture: [`ARCHITECTURE.md`](/Users/cleider/dev/ecoSignalLab/ARCHITECTURE.md)
- Metrics reference: [`docs/METRICS_REFERENCE.md`](/Users/cleider/dev/ecoSignalLab/docs/METRICS_REFERENCE.md)
- Phase 1 ecosystem analysis: [`docs/PHASE1_ECOSYSTEM_GAP_ANALYSIS.md`](/Users/cleider/dev/ecoSignalLab/docs/PHASE1_ECOSYSTEM_GAP_ANALYSIS.md)

## License

MIT. See [`LICENSE`](/Users/cleider/dev/ecoSignalLab/LICENSE).
