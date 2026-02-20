# Novelty and Anomaly Semantics

This document specifies the exact novelty/anomaly methods implemented in `esl`, including defaults and peak-picking behavior.

Primary code paths:
- `src/esl/metrics/builtin.py`
- `src/esl/metrics/extended.py`
- `src/esl/viz/plotting.py`

## Novelty Curve Metric (`novelty_curve`)

Method:
- Compute STFT magnitude `M(f,t)` from mono downmix.
- Positive spectral flux:
  - `N(t) = sum_f max(M(f,t) - M(f,t-1), 0)`

Implementation details:
- Frame params: `frame_size=2048`, `hop_size=512`
- No explicit temporal smoothing
- Units: arbitrary (`a.u.`)

## Spectral Change Detection (`spectral_change_detection`)

Method:
- Start from novelty curve `N(t)`.
- Z-score normalization:
  - `Z(t) = (N(t) - mean(N)) / (std(N) + eps)`

Event extraction:
- Default event candidate threshold in implementation: `Z(t) > 2.5`
- Output:
  - metric series is `Z(t)`
  - `extra.event_frame_indices` stores detected frame indices

## Similarity Matrix Plot

Method:
- Frame-level feature vectors `X[t,:]` are extracted from mono downmix.
- Default `feature_set=auto`:
  - uses librosa-rich features when available (`log-mel`, MFCC(+delta,+delta2), chroma, spectral contrast, tonnetz, spectral and temporal descriptors)
  - falls back to built-in scipy core feature stack when librosa is unavailable
- You can force `feature_set`:
  - `core`, `librosa`, `all`, or `auto`
- Optional external feature vectors can be provided (`.npz`, `.npy`, `.csv`) and used directly.
- Per-frame centering + L2 normalization.
- Cosine self-similarity:
  - `S = X * X^T`, clipped to `[0,1]`

CLI:
- `esl analyze ... --similarity-matrix --plot`
- `esl plot results.json --similarity-matrix`
- `esl plot results.json --similarity-matrix --sim-feature-set all`
- `esl plot results.json --similarity-matrix --feature-vectors vectors.npz`

## Novelty Matrix Plot (Foote-style)

Method:
1. Compute self-similarity matrix `S`.
2. Build Gaussian checkerboard kernel `K` of size `(2L+1)^2`:
   - default `kernel_size=32` => `L=16` target (bounded by matrix size)
   - default `kernel_sigma_scale=0.5`
3. Novelty by diagonal convolution:
   - `novelty[i] = sum( S_local * K )`
4. Half-wave rectification + max normalization to `[0,1]`.
5. Peak picking (`scipy.signal.find_peaks`) with:
   - `distance = max(1, L//2)`
   - `prominence = 0.08` when non-zero novelty exists

```mermaid
flowchart LR
    A["Log-mel features"] --> B["Self-similarity matrix S"]
    B --> C["Checkerboard kernel K"]
    B --> D["Diagonal convolution with K"]
    C --> D
    D --> E["Novelty curve"]
    E --> F["Peak picking"]
```

CLI:
- `esl analyze input.wav --plot --novelty-matrix`
- `esl plot results.json --novelty-matrix`
- `esl plot results.json --novelty-matrix --sim-feature-set librosa`
- `esl plot results.json --novelty-matrix --feature-vectors vectors.csv`

## Model-Based Anomaly Metrics

- `isolation_forest_score`
  - Uses frame feature matrix from spectral descriptors.
  - Score semantics: larger score => more anomalous.
  - Fallback behavior: z-score L2 norm when model dependency unavailable.

- `ocsvm_score`
  - One-Class SVM score with same “higher = more anomalous” semantics.
  - Fallback behavior mirrors `isolation_forest_score`.

- `autoencoder_recon_error`
  - Low-rank PCA/SVD reconstruction proxy.
  - Frame-wise MSE residual as anomaly score.

- `change_point_confidence`
  - Uses novelty z-score peak structure.
  - Confidence maps peak prominence to `[0,1]`.

## Confidence and Interpretation

- Novelty and change metrics are relative descriptors; absolute thresholds are dataset-dependent.
- Model-based scores are unsupervised and should be calibrated against site/domain baselines.
- Use `metadata.validity_flags` and per-metric `confidence` together for gating.

## Related Docs

- [`METRICS_REFERENCE.md`](METRICS_REFERENCE.md)
- [`ML_FEATURES.md`](ML_FEATURES.md)
- [`SCHEMA.md`](SCHEMA.md)
- [`REFERENCES.md`](REFERENCES.md)
