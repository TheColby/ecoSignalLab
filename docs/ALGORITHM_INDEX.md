# Algorithm Index

This index maps `esl` code paths to algorithm families and citation anchors.

Primary bibliography: [`docs/REFERENCES.md`](REFERENCES.md)

## Coverage Map

| Code Path | Algorithm Family | References |
|---|---|---|
| [`src/esl/metrics/helpers.py`](../src/esl/metrics/helpers.py) | STFT, spectral flux novelty, Schroeder decay, RT regression | [D1], [N4], [A1], [S3], [S4] |
| [`src/esl/metrics/builtin.py`](../src/esl/metrics/builtin.py) | Core spectral/temporal/architectural metrics | [D1], [N4], [S3], [S4], [A1] |
| [`src/esl/metrics/extended.py`](../src/esl/metrics/extended.py) | Loudness (LUFS/LRA), ecoacoustic indices, spatial cues, anomaly models | [S1], [S2], [E1], [E2], [E3], [P1], [M1], [M2], [M3] |
| [`src/esl/viz/plotting.py`](../src/esl/viz/plotting.py) | Similarity matrix, Foote novelty matrix, mel spectrograms | [D1], [D3], [N1], [N2], [N3] |
| [`src/esl/ml/export.py`](../src/esl/ml/export.py) | Isolation Forest anomaly scoring | [M1], [M3] |
| [`src/esl/core/audio.py`](../src/esl/core/audio.py) | Polyphase resampling for sample-rate conversion | [D2] |

## Attribution Cross-Link

For OSS-specific implementation inspiration and licensing context, see:
- [`docs/ATTRIBUTION.md`](ATTRIBUTION.md)

## Visual Traceability

```mermaid
flowchart TD
    A["Code Module"] --> B["Algorithm Family"]
    B --> C["Paper / Standard Citation"]
    C --> D["In-Code Comment"]
    D --> E["References + Attribution Docs"]
```
