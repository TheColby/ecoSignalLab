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
  --ml-export \
  --project restaurant_design \
  --variant designA
```

### Batch mode

```bash
esl batch input_dir --out out_dir --csv --parquet --hdf5
```

### Plot from existing JSON

```bash
esl plot results.json --out plots --interactive
```

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

- Basic: RMS, peak
- Level/loudness: SPL (A/C/Z), crest factor
- Noise/SNR: percentile SNR estimator
- Spectral: centroid, bandwidth, flatness, rolloff
- Temporal: zero crossing rate
- Ecoacoustics: bioacoustic index, acoustic complexity index
- Spatial: interchannel coherence
- Architectural acoustics: RT60, EDT, C50, C80, D50
- Anomaly/novelty: spectral flux novelty, change detection

## Reproducibility

Each run emits:
- `config_hash` (SHA-256 over canonicalized config)
- `esl_version`
- UTC timestamp
- seed value
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
- Phase 1 ecosystem analysis: [`docs/PHASE1_ECOSYSTEM_GAP_ANALYSIS.md`](/Users/cleider/dev/ecoSignalLab/docs/PHASE1_ECOSYSTEM_GAP_ANALYSIS.md)

## License

MIT. See [`LICENSE`](/Users/cleider/dev/ecoSignalLab/LICENSE).
