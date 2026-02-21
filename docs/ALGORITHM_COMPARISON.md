# Algorithm Comparison (Real-Input KPIs)

This page explains how to compare time-stretch algorithms using numerical KPIs computed from a real input audio file.

Script:
- [`../scripts/compare_time_stretch_kpis.py`](../scripts/compare_time_stretch_kpis.py)

Easy wrapper:
- [`../scripts/easy/04_compare_kpis.sh`](../scripts/easy/04_compare_kpis.sh)

## Quick Run

```bash
python scripts/compare_time_stretch_kpis.py \
  --input input.wav \
  --out-dir out/kpi_compare \
  --factor 2.0
```

Outputs:
- `out/kpi_compare/kpi_summary.csv`
- `out/kpi_compare/kpi_summary.json`
- `out/kpi_compare/renders/*.wav`

## Include PVX in Comparison

If you have `pvx` installed, provide a command template:

```bash
python scripts/compare_time_stretch_kpis.py \
  --input input.wav \
  --out-dir out/kpi_compare \
  --factor 2.0 \
  --methods pvx_external,ffmpeg_atempo,scipy_resample,librosa_phase_vocoder \
  --pvx-cmd "pvx stretch --in {input} --out {output} --factor {factor}"
```

This runs PVX and other methods on the same input and reports KPIs side by side.

## KPI Definitions

### Duration error

$$
\Delta t_{\mathrm{ms}} = 1000 \cdot \left(t_{\mathrm{out}} - t_{\mathrm{target}}\right)
$$

where \(t_{\mathrm{out}}\) is rendered duration and \(t_{\mathrm{target}}\) is expected duration from stretch factor.

Plain English: closer to 0 ms means the algorithm hit the requested stretch more accurately.

### Realtime factor

$$
\mathrm{RTF} = \frac{t_{\mathrm{runtime}}}{t_{\mathrm{out}}}
$$

where \(t_{\mathrm{runtime}}\) is compute time and \(t_{\mathrm{out}}\) is output duration.

Plain English: lower is faster; `RTF < 1` is faster-than-realtime.

### Clipping ratio

$$
r_{\mathrm{clip}} = \frac{1}{N}\sum_{n=1}^{N}\mathbf{1}\left(|x[n]| \ge 0.999\right)
$$

where \(x[n]\) is output sample value and \(\mathbf{1}(\cdot)\) is indicator function.

Plain English: lower is better; high clipping means distortion risk.

### Spectral centroid delta (%)

$$
\Delta c_{\%}=100\cdot\frac{c_{\mathrm{out}}-c_{\mathrm{in}}}{|c_{\mathrm{in}}|+\varepsilon}
$$

where \(c_{\mathrm{in}}\) and \(c_{\mathrm{out}}\) are mean spectral centroids of input and output.

Plain English: large shifts may indicate timbral color changes.

### Spectral flatness delta (%)

$$
\Delta f_{\%}=100\cdot\frac{f_{\mathrm{out}}-f_{\mathrm{in}}}{|f_{\mathrm{in}}|+\varepsilon}
$$

where \(f_{\mathrm{in}}\) and \(f_{\mathrm{out}}\) are mean spectral flatness values.

Plain English: large shifts can indicate noisier or more tonal artifacts.

### Transient count ratio

$$
r_{\mathrm{transient}}=\frac{N_{\mathrm{peaks,out}}}{\max(1,N_{\mathrm{peaks,in}})}
$$

where \(N_{\mathrm{peaks,in}}\) and \(N_{\mathrm{peaks,out}}\) are novelty-peak counts from input/output.

Plain English: values far from 1 may indicate transient smearing or artificial events.

### KPI score (heuristic)

The script reports a combined `kpi_score` from duration, clipping, spectral deltas, and transient ratio.

Plain English: use it for quick ranking, then inspect detailed KPI columns before deciding.

## Related Docs

- [`GETTING_STARTED.md`](GETTING_STARTED.md)
- [`TASK_RECIPES.md`](TASK_RECIPES.md)
- [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)
