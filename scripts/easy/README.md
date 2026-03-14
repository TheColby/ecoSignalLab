# Easy Scripts

Quick links: [Docs Index](../../docs/INDEX.md) | [Task Recipes](../../docs/TASK_RECIPES.md) | [Troubleshooting](../../docs/TROUBLESHOOTING.md)

These scripts are for someone who just downloaded the repo and wants immediate results.

Run all commands from repo root:

```bash
bash scripts/easy/<script>.sh ...
```

## Fastest Start

```bash
bash scripts/easy/00_doctor.sh input.wav
bash scripts/easy/02_analyze_and_plot.sh input.wav out
bash scripts/easy/18_simple_summary.sh input.wav
bash scripts/easy/03_extract_single_moment.sh input.wav out/moments
bash scripts/easy/17_extract_features_all.sh input.wav out/features/vectors.npz out/features/vectors_meta.json
```

## Script Index

### Single-file workflows

- `00_doctor.sh`
  - inspect environment readiness and optional input file metadata.
  - usage: `bash scripts/easy/00_doctor.sh [input.wav]`
- `01_stretch_2x.sh`
  - quick FFmpeg 2x stretch utility.
  - usage: `bash scripts/easy/01_stretch_2x.sh <input.wav> [output.wav]`
- `02_analyze_and_plot.sh`
  - analyze one file and generate plots.
  - usage: `bash scripts/easy/02_analyze_and_plot.sh <input.wav> [out_dir]`
- `18_simple_summary.sh`
  - print a compact human-readable summary for one file.
  - usage: `bash scripts/easy/18_simple_summary.sh <input.wav>`
- `10_analyze_with_calibration.sh`
  - analyze with calibration and export JSON/CSV/Parquet/HDF5/MAT.
  - usage: `bash scripts/easy/10_analyze_with_calibration.sh <input.wav> <calibration.yaml|json> [out_dir]`
- `11_plot_novelty_similarity.sh`
  - render similarity + novelty matrix plots from existing JSON.
  - usage: `bash scripts/easy/11_plot_novelty_similarity.sh <results.json> [out_dir] [audio.wav]`
- `17_extract_features_all.sh`
  - extract ML-ready feature vectors (`all` feature set).
  - usage: `bash scripts/easy/17_extract_features_all.sh <input.wav> [out_vectors.npz] [meta.json]`

### Moments extraction workflows

- `03_extract_single_moment.sh`
  - extract the single top-ranked moment as WAV + CSV row.
  - usage: `bash scripts/easy/03_extract_single_moment.sh <input.wav> [out_dir]`
- `06_extract_topk_moments.sh`
  - extract top-k ranked moments.
  - usage: `bash scripts/easy/06_extract_topk_moments.sh <input.wav> [out_dir] [top_k] [event_window_s]`
- `07_extract_all_moments.sh`
  - extract all detected moments with explicit pre/post window.
  - usage: `bash scripts/easy/07_extract_all_moments.sh <input.wav> [out_dir] [window_before_s] [window_after_s]`
- `16_24h_ambisonic_moments.sh`
  - long-file moment extraction defaults for 24h + high-SR multichannel workflows.
  - usage: `bash scripts/easy/16_24h_ambisonic_moments.sh <input_24h.wav> [out_dir] [chunk_size] [top_k]`

### Batch and search workflows

- `08_batch_full_exports.sh`
  - batch analyze folder with CSV/Parquet/HDF5/MAT exports + plots.
  - usage: `bash scripts/easy/08_batch_full_exports.sh <input_dir> [out_dir]`
- `09_similarity_search.sh`
  - query-to-corpus similarity ranking with JSON + CSV output.
  - usage: `bash scripts/easy/09_similarity_search.sh <query.wav> <corpus_dir> [out_dir] [top_k]`

### Validation, benchmark, and ops

- `04_compare_kpis.sh`
  - compare KPI behavior against time-stretched reference workflow.
  - usage: `bash scripts/easy/04_compare_kpis.sh <input.wav> [out_dir] [stretch_factor]`
- `05_generate_signal_graphs.sh`
  - generate waveform/window/spectrogram/novelty guide visuals.
  - usage: `bash scripts/easy/05_generate_signal_graphs.sh [out_dir]`
- `12_calibration_check.sh`
  - run calibration drift check from a tone recording.
  - usage: `bash scripts/easy/12_calibration_check.sh <tone.wav> <calibration.yaml|json> [out.json]`
- `13_benchmark_device.sh`
  - benchmark CPU/CUDA/MPS tensor backend.
  - usage: `bash scripts/easy/13_benchmark_device.sh [device] [frames] [features] [iters] [out_json]`
- `14_pipeline_run.sh`
  - run staged pipeline with plots + ML export, then print status.
  - usage: `bash scripts/easy/14_pipeline_run.sh <input_dir> [out_dir]`
- `15_schema_export.sh`
  - export active JSON schema.
  - usage: `bash scripts/easy/15_schema_export.sh [out_json]`

## Notes

- Scripts assume `esl` is installed and available on `PATH`.
- If `esl` is not found, run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```
