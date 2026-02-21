# Glossary (Plain English)

Beginner-friendly definitions used across `esl` docs.

## Audio basics

- `sample rate`: how many audio samples are stored per second (for example `44100 Hz`).
- `channel`: one audio stream inside a file (mono=1, stereo=2, multichannel=>2).
- `WAV`: uncompressed audio container commonly used for analysis.
- `spectrogram`: a picture of frequency content over time.

## Level and loudness

- `dBFS`: digital level relative to full scale (0 dBFS is digital ceiling).
- `SPL`: physical sound pressure level (real-world loudness measure with calibration).
- `dBA`: SPL with A-weighting (human-hearing emphasis).
- `LUFS`: broadcast/program loudness unit.
- `crest factor`: peak-to-average ratio; higher means more transient peaks.

## Analysis and metrics

- `novelty`: how much the sound changes from one frame to the next.
- `similarity matrix`: frame-to-frame similarity map; diagonal structure reveals repetition.
- `NDSI`: ecoacoustic ratio comparing biological-band vs anthropogenic-band energy.
- `RT60`: time for reverberation to decay by 60 dB (room acoustics metric).

## Spatial audio

- `ambisonics`: spatial audio representation using spherical harmonics.
- `FOA`: first-order ambisonics (typically 4 channels).
- `binaural`: two-channel rendering that simulates human ears.

## Reproducibility and exports

- `schema_version`: version of JSON output contract.
- `config_hash`: deterministic hash of analysis configuration.
- `pipeline_hash`: hash of config + metrics + runtime versions.
- `FrameTable`: frame-wise feature table for ML export.

## Calibration

- `calibration`: mapping between digital level (`dBFS`) and physical level (`SPL`).
- `dbfs_reference`: known digital reference point used for calibration.
- `spl_reference_db`: known physical SPL corresponding to the digital reference.

## Related Docs

- [`GETTING_STARTED.md`](GETTING_STARTED.md)
- [`TASK_RECIPES.md`](TASK_RECIPES.md)
- [`SCHEMA.md`](SCHEMA.md)
- [`METRICS_REFERENCE.md`](METRICS_REFERENCE.md)
