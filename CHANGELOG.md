# Changelog

All notable changes to `ecoSignalLab` (`esl`) are documented here.

## [0.2.0] - Unreleased

### Added
- First-time onboarding command:
  - `esl quickstart` prints copy-paste commands for analysis, moments extraction, and feature export.
- New user guide:
  - `docs/GETTING_STARTED.md` with install, first commands, troubleshooting, and mental model.
- New beginner docs:
  - `docs/TASK_RECIPES.md` (goal-oriented copy/paste tasks)
  - `docs/TROUBLESHOOTING.md` (error-first fixes)
  - `docs/GLOSSARY.md` (plain-English definitions)
- Easy scripts folder for just-downloaded workflows:
  - `scripts/easy/01_stretch_2x.sh`
  - `scripts/easy/02_analyze_and_plot.sh`
  - `scripts/easy/03_extract_single_moment.sh`
  - `scripts/easy/04_compare_kpis.sh`
  - `scripts/easy/README.md`
- Real-input algorithm KPI comparison:
  - `scripts/compare_time_stretch_kpis.py`
  - `docs/ALGORITHM_COMPARISON.md`
  - optional PVX inclusion via `--pvx-cmd` template
- `moments` extraction workflow:
  - `esl moments extract` for timestamped moment CSV + clip export
  - `src/esl/core/moments.py`
  - `docs/MOMENTS_EXTRACTION.md`
- Tag-driven release automation:
  - `.github/workflows/release.yml` (tests, docs HTML/PDF, schema export, dist packaging, GitHub release uploads)
- Versioned schema publication artifact:
  - `docs/schema/analysis-output-0.2.0.json`
- Schema compatibility test:
  - `tests/test_schema_artifact.py`
- FFmpeg decode provenance integration test:
  - `tests/test_ffmpeg_provenance.py`
- Documentation link regression tests:
  - `tests/test_docs_links.py`
- Canonical ML FrameTable contract docs and exports:
  - `docs/ML_FEATURES.md`
  - `src/esl/ml/export.py` (`FrameTable`, tensor/tabular contract exports)
- Similarity/novelty feature vector expansion:
  - optional librosa-rich feature extraction (`esl features extract`)
  - external vector-driven similarity/novelty plotting
- Maintainer release runbook:
  - `docs/RELEASE.md`

### Changed
- User-facing ergonomics:
  - friendlier CLI `--help` epilog with quickstart hint
  - friendlier runtime error hints for missing paths and invalid options
  - README onboarding now includes start-in-60-seconds, pick-your-goal navigation, and expected outputs
- Positioning language now explicitly states:
  - true multichannel and Atmos-aware/capable phase-vocoder workflow intent across README/design/architecture docs
- Hardened JSON output provenance and schema fields:
  - `schema_version`, `pipeline_hash`, `metric_catalog`, `library_versions`
  - decoder provenance (`decoder_used`, `ffmpeg_version`, `ffprobe` summary)
  - config snapshot, resolved metric list, channel summaries, validity flags
- CLI quality and consistency:
  - clearer `--verbosity` / `--debug` help semantics
  - consistent output switch coverage including batch `--mat`
  - `schema` command reports schema version
- Moments extraction selection and windowing controls:
  - `--single`, `--top-k`, `--all`, `--rank-metric`
  - event-centered clip window options: `--event-window`, `--window-before`, `--window-after`
  - ranking metadata exported in `moments.csv` and `moments_report.json`
- Docs generation now enforces visual + math rendering:
  - Mermaid rendering on all generated pages (auto visual-outline insertion when missing)
  - MathJax-based TeX rendering in HTML/PDF outputs
  - expanded rendered-equation coverage in metrics/novelty/moments docs
- CI quality gates (`.github/workflows/ci.yml`):
  - fatal lint gate
  - scoped strict mypy gate for touched core files
  - full pytest gate
  - golden test coverage floor gate
  - docs HTML build gate

### Fixed
- Documentation internal links no longer depend on local absolute filesystem paths.
- Project metadata URLs now point to the real GitHub repository.
