# Changelog

All notable changes to `ecoSignalLab` (`esl`) are documented here.

## [0.2.0] - Unreleased

### Added
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
- Hardened JSON output provenance and schema fields:
  - `schema_version`, `pipeline_hash`, `metric_catalog`, `library_versions`
  - decoder provenance (`decoder_used`, `ffmpeg_version`, `ffprobe` summary)
  - config snapshot, resolved metric list, channel summaries, validity flags
- CLI quality and consistency:
  - clearer `--verbosity` / `--debug` help semantics
  - consistent output switch coverage including batch `--mat`
  - `schema` command reports schema version
- CI quality gates (`.github/workflows/ci.yml`):
  - fatal lint gate
  - scoped strict mypy gate for touched core files
  - full pytest gate
  - golden test coverage floor gate
  - docs HTML build gate

### Fixed
- Documentation internal links no longer depend on local absolute filesystem paths.
- Project metadata URLs now point to the real GitHub repository.
