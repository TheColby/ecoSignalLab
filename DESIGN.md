# DESIGN

## Product intent

`esl` is designed as a unified SDK where environmental acoustics, architectural acoustics, and ML pipelines share one reproducible analysis contract.

## Design principles

1. Multi-channel first
- Internal arrays are sample-major, channel-preserving: `[samples, channels]`.

2. Calibration-aware semantics
- Metrics can operate without calibration, but outputs explicitly annotate when SPL values are proxy values.

3. Plugin contracts over ad hoc functions
- Each metric declares:
  - Name
  - Units
  - Window and hop
  - Streaming capability
  - Calibration dependency
  - Confidence logic
  - ML compatibility

4. Deterministic defaults
- Seed defaults to `42` and is persisted in output metadata.

5. Export interoperability
- Analysis outputs are schema-governed and exportable to scientific, industrial, and ML-centric formats.

## Tradeoffs in v0.1.0

- Broad codec support depends on local `ffmpeg` availability.
- SOFA support is focused on IR extraction paths, not full SOFA convention processing.
- Streaming mode currently prioritizes streaming-capable metrics; mixed metric sets fallback to full mode.

## Extension model

External plugins can be exposed through the `esl.plugins` Python entry-point group and are auto-registered when importable.
