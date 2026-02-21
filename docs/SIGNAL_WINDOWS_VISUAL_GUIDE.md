# Signal and Window Visual Guide

This guide provides generated graphs for core DSP concepts used across `esl`.

Generation script:
- [`../scripts/generate_signal_window_graphs.py`](../scripts/generate_signal_window_graphs.py)

Regenerate all images:

```bash
python scripts/generate_signal_window_graphs.py --out docs/examples/signal_window_guide
```

## Why these plots matter

- They show how frame-based processing sees your signal.
- They visualize windowing and overlap-add behavior.
- They connect waveform, STFT, novelty, and streaming chunk logic in one place.

## Core equations

$$
n_k = kH
$$

where \(n_k\) is start index of frame \(k\), and \(H\) is hop size.

Plain English: each new frame starts \(H\) samples after the previous one.

$$
x_k[n] = x[n_k+n]\cdot w[n], \quad 0\le n < N
$$

where \(x_k[n]\) is windowed frame sample, \(x[\cdot]\) is original signal, \(w[n]\) is window, and \(N\) is frame size.

Plain English: each frame is a chunk of signal multiplied by a taper window.

$$
\hat{x}[n] = \sum_k x_k[n-kH]
$$

where \(\hat{x}[n]\) is overlap-added reconstruction from shifted frames.

Plain English: shifted windowed frames are added together to reconstruct the signal path.

$$
\mathcal{F}(k) = \sum_f \max\left(|X(f,k)| - |X(f,k-1)|, 0\right)
$$

where \(\mathcal{F}(k)\) is positive spectral flux at frame \(k\).

Plain English: novelty rises when new spectral energy appears.

## Visual pack

### 1) Signal waveform

![Signal waveform](examples/signal_window_guide/signal_waveform.png)

### 2) Frame/hop overlay on waveform

![Frame hop overlay](examples/signal_window_guide/frame_hop_overlay.png)

### 3) Window family

![Window family](examples/signal_window_guide/window_family.png)

### 4) Hann overlap-add behavior (50% overlap)

![Overlap add](examples/signal_window_guide/overlap_add_hann_50pct.png)

### 5) Spectrogram with frame centers

![Spectrogram frames](examples/signal_window_guide/spectrogram_with_frames.png)

### 6) Spectral flux positive difference

![Spectral flux](examples/signal_window_guide/spectral_flux_positive_diff.png)

### 7) Foote checkerboard kernel

![Checkerboard kernel](examples/signal_window_guide/checkerboard_kernel.png)

### 8) Multichannel waveforms

![Multichannel waveforms](examples/signal_window_guide/multichannel_waveforms.png)

### 9) FOA channel example (WXYZ)

![FOA WXYZ](examples/signal_window_guide/foa_wxyz_waveforms.png)

### 10) Chunk-vs-frame timeline

![Chunk frame timeline](examples/signal_window_guide/chunk_vs_frame_timeline.png)

## Processing relationship

```mermaid
flowchart LR
    A["Waveform"] --> B["Frame/Hop Segmentation"]
    B --> C["Windowing"]
    C --> D["STFT"]
    D --> E["Spectral Flux / Novelty"]
    B --> F["Streaming Chunk Logic"]
```

## Related Docs

- [`METRICS_REFERENCE.md`](METRICS_REFERENCE.md)
- [`NOVELTY_ANOMALY.md`](NOVELTY_ANOMALY.md)
- [`MOMENTS_EXTRACTION.md`](MOMENTS_EXTRACTION.md)
- [`GETTING_STARTED.md`](GETTING_STARTED.md)
