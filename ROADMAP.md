# Roadmap

Quick links: [README](README.md) | [Docs Index](docs/INDEX.md) | [Task Recipes](docs/TASK_RECIPES.md) | [Schema](docs/SCHEMA.md) | [Metrics](docs/METRICS_REFERENCE.md)

This roadmap reflects the current `esl` direction after long-file support, moments extraction, shard workflows, shard-level similarity search, and the first public release hardening pass.

## Snapshot

What already exists now:

- `esl shard index`
- `esl shard analyze`
- `esl shard moments`
- `esl shard similar`
- `esl shard retrieve`
- `esl shard insights`
- appendable `FrameTable` sidecars:
  - CSV
  - Parquet dataset directory
  - HDF5
- calibration verification fixtures with `esl calibrate verify`
- structured spatial / Ambisonics-aware metadata in analysis outputs

Plain English: we are no longer at the “can this open a big file?” stage. We are at the “can this run long archives in a way that is boringly reliable?” stage.

## Near Term

1. Stabilize the `v0.2.x` line.
2. Expand long-archive insight workflows.
3. Make multichannel and spatial metadata first-class.
4. Tighten calibration verification and reference fixtures.
5. Improve ML-ready FrameTable exports.

## Version Map

- `v0.2.x`
  - correctness, schema stability, long-file polish, docs hardening
- `v0.3.0`
  - shard moments, appendable FrameTable Parquet/HDF5, richer archive workflows
- `v0.4.0`
  - multichannel/spatial-first event ranking and metadata-aware pipelines
- `v0.5.0`
  - architectural simulation ingestion/report depth
- `v0.6.0`
  - ML dataset manifests, feature presets, retrieval baselines
- `v1.0.0`
  - stable schema/API, dependable production archive workflows, interop maturity

## Current Priorities

### 1. Long-Archive Operations

- `esl shard index`
- `esl shard analyze`
- `esl shard moments`
- `esl shard similar`
- `esl shard insights summary/scene/calmness/drift/report`
- resumable stream passes
- archive-level top-k novelty/event extraction
- archive-level query-to-shard retrieval
- archive-level query-to-event retrieval
- appendable FrameTable exports

### 2. Multichannel / Spatial

- structured spatial metadata in all analysis outputs
- Ambisonics-aware metadata, channel labels, channel maps, and normalization profiles
- explicit downmix vs per-channel ranking semantics
- array-geometry-aware spatial analysis metadata

### 3. Calibration

- drift checks for real calibration tones
- deterministic software verification fixtures and suite reports
- clearer Pa <-> dBFS <-> SPL audit paths
- reference reports suitable for CI and onboarding

### 4. ML / Dataset Readiness

- canonical FrameTable contract
- appendable CSV / Parquet / HDF5 exports
- tensor layout guarantees
- feature-set presets for ecoacoustics, anomaly detection, architectural acoustics, and spatial workflows

### 5. Documentation and UX

- copy-paste workflows for huge archives
- better task recipes by device and archive type
- more first-run diagnostics
- richer examples of moments, similarity, calibration, and multichannel analysis

## Suggested 30 / 60 / 90

### 30 Days

- stabilize shard moments, shard similarity, and shard insights
- finish appendable FrameTable export ergonomics
- improve docs and examples for huge archives
- increase regression coverage around ranking semantics
- add retrieval examples for “which day/hour sounds like this?”

### 60 Days

- add spatial-aware archive ranking modes
- expand calibration verification fixtures
- improve Parquet/HDF5 downstream ergonomics
- expand archive-scale overview plots beyond day/month/year rollups
- harden query-to-event retrieval beyond whole-shard retrieval

### 90 Days

- formalize dataset manifests
- strengthen simulation interoperability
- add more industrial interchange mappings
- publish a more stable `v0.3.x` workflow surface

## Guiding Principle

`esl` should keep getting better at one thing in particular:

turning very large, messy, multichannel acoustic recordings into reproducible, inspectable, ML-ready knowledge without pretending they are small files.

## Next 5 Concrete Builds

1. calibration sidecar export for device onboarding packets
2. spatial retrieval performance baselines across representative multichannel deployments
3. calendar-aware reporting in local time zones with explicit DST display policy
4. grouped dataset splits for site/device/time-block evaluation
5. formal Ambisonics convention validation fixtures for FOA and HOA recordings
