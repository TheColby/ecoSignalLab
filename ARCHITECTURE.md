# ARCHITECTURE

See related docs:
- [`/Users/cleider/dev/ecoSignalLab/DESIGN.md`](/Users/cleider/dev/ecoSignalLab/DESIGN.md)
- [`/Users/cleider/dev/ecoSignalLab/docs/METRICS_REFERENCE.md`](/Users/cleider/dev/ecoSignalLab/docs/METRICS_REFERENCE.md)
- [`/Users/cleider/dev/ecoSignalLab/docs/ALGORITHM_INDEX.md`](/Users/cleider/dev/ecoSignalLab/docs/ALGORITHM_INDEX.md)
- [`/Users/cleider/dev/ecoSignalLab/docs/REFERENCES.md`](/Users/cleider/dev/ecoSignalLab/docs/REFERENCES.md)
- [`/Users/cleider/dev/ecoSignalLab/docs/ATTRIBUTION.md`](/Users/cleider/dev/ecoSignalLab/docs/ATTRIBUTION.md)

## Package Layout

- `src/esl/core`
  - analysis orchestration, audio I/O, calibration, context, config
- `src/esl/metrics`
  - metric contracts, helpers, built-ins, extended set, registry
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
- `src/esl/docsgen`
  - Markdown -> hyperlink-rich HTML/PDF documentation builder

```mermaid
flowchart TD
    CLI["src/esl/cli"] --> CORE["src/esl/core"]
    CORE --> METRICS["src/esl/metrics"]
    CORE --> IO["src/esl/io"]
    CORE --> VIZ["src/esl/viz"]
    CORE --> ML["src/esl/ml"]
    CORE --> PROJECT["src/esl/project"]
    CLI --> PIPE["src/esl/pipeline"]
    CLI --> INGEST["src/esl/ingest"]
    CLI --> DOCS["src/esl/docsgen"]
    CORE --> SCHEMA["src/esl/schema"]
```

## Runtime Flow

1. CLI parses arguments into `AnalysisConfig`.
2. Audio loader decodes via `soundfile` or `ffmpeg` fallback.
3. `AnalysisContext` carries signal, calibration, and run config.
4. `MetricRegistry` computes selected plugins.
5. Result document assembled with provenance and metric specs.
6. Optional exports/plots/ML artifacts/project indexing are emitted.

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as esl.cli
    participant C as esl.core
    participant R as MetricRegistry
    participant E as Exporters
    U->>CLI: command
    CLI->>C: build AnalysisConfig
    C->>C: read/decode audio
    C->>R: compute metrics
    R-->>C: MetricResult map
    C->>E: write outputs
    E-->>U: artifacts + provenance
```

## Staged Pipeline Flow

`esl pipeline run` executes explicit stages:
- `analyze`
- `plot` (optional)
- `ml_export` (optional)
- `digest`

Each run writes `pipeline_manifest.json` with per-stage status, timing, counts, and errors.

```mermaid
stateDiagram-v2
    [*] --> analyze
    analyze --> plot: "if --plot"
    analyze --> ml_export: "if --ml-export and --plot not set"
    plot --> ml_export: "if --ml-export"
    analyze --> digest: "if no optional stages"
    ml_export --> digest
    plot --> digest: "if no ml_export"
    digest --> completed
    completed --> [*]
```

## Data Model

Top-level result:
- provenance (`esl_version`, `analysis_time_utc`, `config_hash`)
- metadata (input, decode backend, calibration assumptions)
- metrics map with series/summary/confidence/spec

```mermaid
erDiagram
    RESULT ||--|| METADATA : contains
    RESULT ||--o{ METRIC_PAYLOAD : contains
    METRIC_PAYLOAD ||--|| METRIC_SPEC : declares
    METRIC_PAYLOAD ||--o{ SERIES_POINT : optional

    RESULT {
      string esl_version
      string analysis_time_utc
      string config_hash
      string analysis_mode
    }
    METADATA {
      string input_path
      int sample_rate
      int channels
      float duration_s
      string backend
    }
    METRIC_SPEC {
      string name
      string category
      string units
      int window_size
      int hop_size
      bool streaming_capable
      bool calibration_dependency
      bool ml_compatible
    }
    SERIES_POINT {
      float timestamp_s
      float value
    }
```

## Architectural Acoustics Path

IR-style metrics (`rt60`, `edt`, `c50`, `c80`, `d50`, `t20`, `t30`, `ts`) are derived from energy decay via Schroeder integration with explicit confidence estimation.

```mermaid
flowchart LR
    A["Impulse Response"] --> B["Schroeder Decay"]
    B --> C["Decay Fit Windows"]
    C --> D["RT60 / EDT / T20 / T30"]
    A --> E["Early/Late Energy Split"]
    E --> F["C50 / C80 / D50"]
    A --> G["Energy Moments"]
    G --> H["Ts"]
```

## Spatial and Ambisonic Path

```mermaid
flowchart TD
    A["Multichannel / FOA Signal"] --> B["Frame Decomposition"]
    B --> C["Interaural Metrics\nIACC ILD IPD ITD"]
    B --> D["GCC-PHAT ITD"]
    D --> E["DOA Azimuth Proxy"]
    A --> F["FOA Energy/Intensity"]
    F --> G["Diffuseness + Energy Vector"]
```

## Novelty + Similarity Path

```mermaid
flowchart LR
    A["Audio STFT"] --> B["Log-Mel Features"]
    B --> C["Cosine Self-Similarity Matrix"]
    C --> D["Checkerboard Kernel Convolution"]
    D --> E["Novelty Curve"]
    E --> F["Peak Picking"]
    F --> G["Event Candidates"]
```

## Documentation Build Path

```mermaid
flowchart LR
    A["Markdown Docs"] --> B["esl docs / scripts/build_docs.py"]
    B --> C["HTML Rendering"]
    C --> D["Mermaid Runtime Expansion"]
    D --> E["Hyperlinked HTML Site"]
    C --> F["Playwright Chromium PDF"]
    F --> G["Per-Page + Combined PDF"]
```

## Reproducibility Guarantees

- Deterministic seed capture
- Config hashing
- Version stamping
- Assumption logging

```mermaid
flowchart TB
    A["Input + Config"] --> B["Seeded Execution"]
    B --> C["Metric Outputs"]
    C --> D["Schema Validation"]
    D --> E["Provenance-Rich Artifact"]
```

## Citation and Attribution in Code

Algorithm references are embedded near implementations in:
- [`/Users/cleider/dev/ecoSignalLab/src/esl/metrics/helpers.py`](/Users/cleider/dev/ecoSignalLab/src/esl/metrics/helpers.py)
- [`/Users/cleider/dev/ecoSignalLab/src/esl/metrics/builtin.py`](/Users/cleider/dev/ecoSignalLab/src/esl/metrics/builtin.py)
- [`/Users/cleider/dev/ecoSignalLab/src/esl/metrics/extended.py`](/Users/cleider/dev/ecoSignalLab/src/esl/metrics/extended.py)
- [`/Users/cleider/dev/ecoSignalLab/src/esl/viz/plotting.py`](/Users/cleider/dev/ecoSignalLab/src/esl/viz/plotting.py)
- [`/Users/cleider/dev/ecoSignalLab/src/esl/ml/export.py`](/Users/cleider/dev/ecoSignalLab/src/esl/ml/export.py)

Open-source attribution details are maintained in [`/Users/cleider/dev/ecoSignalLab/docs/ATTRIBUTION.md`](/Users/cleider/dev/ecoSignalLab/docs/ATTRIBUTION.md).
