# DESIGN

`esl` is designed as a unified SDK where environmental acoustics, architectural acoustics, and ML pipelines share one reproducible analysis contract.

See also:
- [`/Users/cleider/dev/ecoSignalLab/ARCHITECTURE.md`](/Users/cleider/dev/ecoSignalLab/ARCHITECTURE.md)
- [`/Users/cleider/dev/ecoSignalLab/docs/METRICS_REFERENCE.md`](/Users/cleider/dev/ecoSignalLab/docs/METRICS_REFERENCE.md)
- [`/Users/cleider/dev/ecoSignalLab/docs/ALGORITHM_INDEX.md`](/Users/cleider/dev/ecoSignalLab/docs/ALGORITHM_INDEX.md)
- [`/Users/cleider/dev/ecoSignalLab/docs/REFERENCES.md`](/Users/cleider/dev/ecoSignalLab/docs/REFERENCES.md)
- [`/Users/cleider/dev/ecoSignalLab/docs/ATTRIBUTION.md`](/Users/cleider/dev/ecoSignalLab/docs/ATTRIBUTION.md)

## Product Intent

- One deterministic analysis contract across DSP, ecoacoustics, architectural acoustics, and ML.
- CLI-first operation for production and reproducible research.
- Explicit handling of calibration and assumptions.

```mermaid
flowchart LR
    A["Acoustic Data"] --> B["Deterministic Analysis Contract"]
    B --> C["Research Reproducibility"]
    B --> D["Production Automation"]
    B --> E["Industrial Interop"]
```

## Design Principles

1. Multi-channel first
- Internal arrays are sample-major and channel-preserving: `[samples, channels]`.

2. Calibration-aware semantics
- Metrics can run without calibration, but outputs explicitly mark proxy vs calibrated SPL semantics.

3. Plugin contracts over ad hoc functions
- Each metric declares name, units, window, hop, streaming capability, calibration dependency, confidence logic, and ML compatibility.

4. Deterministic defaults
- Seed defaults to `42` and is persisted in provenance metadata.

5. Export interoperability
- Analysis output is schema-governed and exportable to scientific and industrial formats.

6. CLI-first orchestration
- Staged pipelines (`analyze`, `plot`, `ml_export`, `digest`) are first-class.
- GUI is optional, not required.

```mermaid
classDiagram
    class MetricSpec {
      +name: str
      +category: str
      +units: str
      +window_size: int
      +hop_size: int
      +streaming_capable: bool
      +calibration_dependency: bool
      +ml_compatible: bool
      +confidence_logic: str
      +description: str
    }
    class MetricResult {
      +summary: dict
      +series: list
      +timestamps_s: list
      +confidence: float
      +extra: dict
    }
    class MetricPlugin {
      +spec: MetricSpec
      +compute(ctx): MetricResult
    }
    MetricPlugin --> MetricSpec
    MetricPlugin --> MetricResult
```

## Confidence and Calibration Semantics

```mermaid
stateDiagram-v2
    [*] --> Uncalibrated
    Uncalibrated --> Calibrated: "Calibration profile provided"
    Calibrated --> DriftChecked: "Calibration tone validated"
    Calibrated --> ProxyLabeled: "Tone unavailable"
    Uncalibrated --> ProxyLabeled: "No calibration profile"
    DriftChecked --> Published
    ProxyLabeled --> Published
    Published --> [*]
```

## Processing Mode Decision

```mermaid
flowchart TD
    A["Start analyze"] --> B{"chunk_size set?"}
    B -- "No" --> C["Full analysis mode"]
    B -- "Yes" --> D{"All selected metrics streaming-capable?"}
    D -- "Yes" --> E["Streaming chunk mode"]
    D -- "No" --> C
    C --> F["Assemble result"]
    E --> F
```

## Reproducibility DAG

```mermaid
flowchart LR
    A["Config"] --> B["Canonicalization"]
    B --> C["Config Hash"]
    D["Version"] --> E["Provenance"]
    F["Seed"] --> E
    C --> E
    G["Runtime Fingerprint"] --> E
    E --> H["Result JSON"]
```

## Tradeoffs in v0.1.x

- Broad codec support depends on local [`ffmpeg`](https://ffmpeg.org/) availability.
- SOFA support is focused on IR extraction paths, not full SOFA convention processing.
- Streaming mode prioritizes streaming-capable metrics; mixed sets fallback to full mode.
- Some advanced metrics are explicitly marked as proxies when full geometry/hardware metadata is unavailable.

```mermaid
flowchart LR
    A["Speed"] <-- tradeoff --> B["Physical Fidelity"]
    C["Breadth of Metrics"] <-- tradeoff --> D["Strict Standards Completeness"]
    E["Portable Runtime"] <-- tradeoff --> F["Heavy External Dependencies"]
```

## Extension Model

External plugins can be exposed through the `esl.plugins` Python entry-point group and are auto-registered when importable.

```mermaid
sequenceDiagram
    participant P as Package
    participant EP as Entry Points
    participant R as MetricRegistry
    P->>EP: expose esl.plugins
    EP-->>R: plugin factory
    R->>R: register MetricSpec + compute()
    R-->>P: available in analyze/batch/pipeline
```

## Algorithm Citation Policy

- Core references are centralized in [`/Users/cleider/dev/ecoSignalLab/docs/REFERENCES.md`](/Users/cleider/dev/ecoSignalLab/docs/REFERENCES.md).
- Code-level references are embedded near implementations in:
  - [`/Users/cleider/dev/ecoSignalLab/src/esl/metrics/helpers.py`](/Users/cleider/dev/ecoSignalLab/src/esl/metrics/helpers.py)
  - [`/Users/cleider/dev/ecoSignalLab/src/esl/metrics/builtin.py`](/Users/cleider/dev/ecoSignalLab/src/esl/metrics/builtin.py)
  - [`/Users/cleider/dev/ecoSignalLab/src/esl/metrics/extended.py`](/Users/cleider/dev/ecoSignalLab/src/esl/metrics/extended.py)
  - [`/Users/cleider/dev/ecoSignalLab/src/esl/viz/plotting.py`](/Users/cleider/dev/ecoSignalLab/src/esl/viz/plotting.py)
- Open-source attribution and license notes are tracked in [`/Users/cleider/dev/ecoSignalLab/docs/ATTRIBUTION.md`](/Users/cleider/dev/ecoSignalLab/docs/ATTRIBUTION.md).
