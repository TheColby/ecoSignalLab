# Task Recipes (Beginner)

Quick links: [Docs Index](INDEX.md) | [Getting Started](GETTING_STARTED.md) | [Task Recipes](TASK_RECIPES.md) | [Troubleshooting](TROUBLESHOOTING.md) | [Schema](SCHEMA.md) | [Metrics](METRICS_REFERENCE.md) | [References](REFERENCES.md)

If you have an audio file and want to understand it, start here.

Use `input.wav` as your source and copy/paste exactly.

Need one-command helpers instead of typing long flags?
- [`../scripts/easy/README.md`](../scripts/easy/README.md)

## Recipe Index (By Goal)

- [Recipe 1: Analyze one file and generate plots](#recipe-1-analyze-one-file-and-generate-plots)
- [Recipe 2: Extract the single most novel moment](#recipe-2-extract-the-single-most-novel-moment)
- [Recipe 3: Extract top-k moments instead of one](#recipe-3-extract-top-k-moments-instead-of-one)
- [Recipe 4: Export ML-ready features](#recipe-4-export-ml-ready-features)
- [Recipe 5: Batch analyze a folder](#recipe-5-batch-analyze-a-folder)
- [Recipe 6: Compare architectural variants](#recipe-6-compare-architectural-variants)
- [Recipe 7: Generate DSP signal/window reference graphs](#recipe-7-generate-dsp-signalwindow-reference-graphs)
- [Recipe 8: Find the most similar files in a folder](#recipe-8-find-the-most-similar-files-in-a-folder)

## Recipe Index by Device Type

- Phone/laptop built-in mic (single/dual channel):
  - [Recipe 1](#recipe-1-analyze-one-file-and-generate-plots), [Recipe 2](#recipe-2-extract-the-single-most-novel-moment), [Recipe 4](#recipe-4-export-ml-ready-features)
- USB stereo interface / handheld stereo recorder:
  - [Recipe 1](#recipe-1-analyze-one-file-and-generate-plots), [Recipe 5](#recipe-5-batch-analyze-a-folder), [Recipe 8](#recipe-8-find-the-most-similar-files-in-a-folder)
- Ambisonic B-format recorder (FOA, multichannel WAV/RF64):
  - [Recipe 1](#recipe-1-analyze-one-file-and-generate-plots), [Recipe 3](#recipe-3-extract-top-k-moments-instead-of-one), [Recipe 5](#recipe-5-batch-analyze-a-folder), [RF64 guide](RF64_AND_LARGE_FILES.md)
- Multichannel array / Atmos-capable workflow:
  - [Recipe 5](#recipe-5-batch-analyze-a-folder), [Recipe 6](#recipe-6-compare-architectural-variants), [Schema contract](SCHEMA.md)
- Remote sensor node / long-duration monitor:
  - [Recipe 2](#recipe-2-extract-the-single-most-novel-moment), [Recipe 3](#recipe-3-extract-top-k-moments-instead-of-one), [Moments workflow](MOMENTS_EXTRACTION.md), [RF64 guide](RF64_AND_LARGE_FILES.md)

## Recipe Index by Input Format

- Uncompressed (`WAV`, `RF64`, `FLAC`, `AIFF`, `CAF`):
  - [Recipe 1](#recipe-1-analyze-one-file-and-generate-plots), [Recipe 5](#recipe-5-batch-analyze-a-folder), [RF64 guide](RF64_AND_LARGE_FILES.md)
- Compressed (`MP3`, `AAC`, `OGG`, `Opus`, `M4A`, `WMA`, `ALAC`):
  - [Recipe 1](#recipe-1-analyze-one-file-and-generate-plots), [Recipe 8](#recipe-8-find-the-most-similar-files-in-a-folder), [Troubleshooting decode issues](TROUBLESHOOTING.md#ffmpeg-decode-errors-for-mp3aacogg)

## Recipe 1: Analyze one file and generate plots

```bash
esl analyze input.wav --out-dir out --json out/input.json --plot
```

What this does:
- computes core and extended acoustic metrics
- writes schema-stable JSON plus plots

Expected outputs:
- `out/input.json`
- `out/input_plots/`

## Recipe 2: Extract the single most novel moment

```bash
esl moments extract input.wav \
  --out out/moments \
  --single \
  --rank-metric novelty_curve \
  --event-window 8
```

What this does:
- scans for novelty peaks
- exports one top-ranked event plus timestamp table

Expected outputs:
- `out/moments/moments.csv`
- `out/moments/moments_report.json`
- `out/moments/clips/moment_0001.wav`

Choose clip window behavior:

- Symmetric window around event center:
  - `--event-window 8` gives roughly 4 s before + 4 s after.
- Asymmetric window around event center:
  - `--window-before 3 --window-after 7` gives 3 s before + 7 s after.
- If you set `--window-before/--window-after`, those are used directly.
- If you do not set either, `--event-window` (when provided) is used.
- If neither is set, default chunk-edge rolls (`--pre-roll/--post-roll`) are used.

Example (single most novel event, custom before/after):

```bash
esl moments extract input.wav \
  --out out/moments \
  --single \
  --rank-metric novelty_curve \
  --window-before 5 \
  --window-after 9
```

## Recipe 3: Extract top-k moments instead of one

```bash
esl moments extract input.wav \
  --out out/moments_top5 \
  --top-k 5 \
  --rank-metric novelty_curve \
  --event-window 10
```

What this does:
- extracts multiple high-interest events
- keeps per-event timestamps and ranks

Expected outputs:
- `out/moments_top5/moments.csv`
- `out/moments_top5/clips/moment_*.wav`

Choose clip window behavior:

- Symmetric window around each selected event center:
  - `--event-window 10` gives roughly 5 s before + 5 s after.
- Asymmetric window around each selected event center:
  - `--window-before 4 --window-after 9` gives 4 s before + 9 s after.
- If you set `--window-before/--window-after`, those are used directly.
- If you do not set either, `--event-window` (when provided) is used.
- If neither is set, default chunk-edge rolls (`--pre-roll/--post-roll`) are used.

Example (top-5 novel events, custom before/after):

```bash
esl moments extract input.wav \
  --out out/moments_top5 \
  --top-k 5 \
  --rank-metric novelty_curve \
  --window-before 4 \
  --window-after 9
```

## Recipe 4: Export ML-ready features

```bash
esl features extract input.wav \
  --out out/features.npz \
  --feature-set all \
  --meta-json out/features_meta.json
```

What this does:
- exports frame-wise features for modeling
- stores frame metadata and layout contract

Expected outputs:
- `out/features.npz`
- `out/features_meta.json`

## Recipe 5: Batch analyze a folder

```bash
esl batch input_dir \
  --out out_batch \
  --metrics rms_dbfs,snr_db,novelty_curve,spl_a_db \
  --report-metrics snr_db,spl_a_db,novelty_curve \
  --csv --parquet --hdf5 --mat \
  --plot
```

What this does:
- processes all supported audio files in a directory
- writes machine-readable outputs for each file
- controls what is computed via `--metrics`
- controls what summary columns appear in `batch_index.csv` via `--report-metrics`

Expected outputs:
- `out_batch/**/*.json`
- `out_batch/**/*.csv`
- `out_batch/**/*.parquet`
- `out_batch/**/*.h5`
- `out_batch/**/*.mat`
- `out_batch/**/_plots/`

How to specify what to report:

- `--metrics` controls the analysis metric set (what gets computed).
- `--report-metrics` controls which metric means are written as columns in `out_batch/batch_index.csv`.
- Output format switches control artifact types:
  - `--csv`, `--parquet`, `--hdf5`, `--mat`
- JSON is always written per analyzed file in batch mode.

## Recipe 6: Compare architectural variants

```bash
esl analyze A.wav --project restaurant_design --variant A --out-dir out
esl analyze B.wav --project restaurant_design --variant B --out-dir out
esl project compare --project restaurant_design --root out --baseline A
```

What this does:
- stores variant analysis results in a shared project context
- computes deltas for room/level/novelty metrics

Expected outputs:
- `out/projects/restaurant_design/comparison.csv`
- `out/projects/restaurant_design/comparison_report.json`

## Recipe 7: Generate DSP signal/window reference graphs

```bash
bash scripts/easy/05_generate_signal_graphs.sh docs/examples/signal_window_guide
```

What this does:
- creates waveform, frame/hop, window family, overlap-add, spectrogram, novelty-kernel, and multichannel/FOA figures

Expected output folder:
- `docs/examples/signal_window_guide/`

See full walkthrough:
- [`SIGNAL_WINDOWS_VISUAL_GUIDE.md`](SIGNAL_WINDOWS_VISUAL_GUIDE.md)

## Recipe 8: Find the most similar files in a folder

```bash
esl similar query.wav corpus_dir \
  --top-k 5 \
  --json out/similarity.json \
  --csv out/similarity.csv
```

What this does:
- analyzes a corpus folder relative to one query file
- ranks candidates by similarity
- writes machine-readable ranking outputs

How to specify what “similar” means:

- default (`--mode auto`) uses feature similarity
- single metric:
  - `--mode metric --metric rms_dbfs`
- multi-metric:
  - `--mode metrics --metrics rms_dbfs,snr_db,spl_a_db,novelty_curve`
- choose distance:
  - `--distance cosine|euclidean|manhattan`

Many useful options:

- `--feature-set auto|core|librosa|all`
- `--sample-rate`, `--frame-size`, `--hop-size`
- `--include-self`, `--no-recursive`, `--max-files`
- `--calibration` (metric modes)
- `--verbosity 0..3`, `--debug 0..2`

Expected outputs:
- `out/similarity.json`
- `out/similarity.csv`

## Which command should I use?

```mermaid
flowchart TD
    A["Need metrics and plots for one file"] --> B["esl analyze"]
    A --> C["Need interesting events and clips"]
    C --> D["esl moments extract"]
    A --> E["Need model-ready vectors"]
    E --> F["esl features extract"]
    A --> G["Need large-scale processing"]
    G --> H["esl batch"]
```

## Related Docs

- [`GETTING_STARTED.md`](GETTING_STARTED.md)
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)
- [`GLOSSARY.md`](GLOSSARY.md)
- [`MOMENTS_EXTRACTION.md`](MOMENTS_EXTRACTION.md)
- [`SIMILARITY_SEARCH.md`](SIMILARITY_SEARCH.md)
