# Easy Scripts

These scripts are for someone who just downloaded the repo and wants immediate results.

Run from repo root:

```bash
bash scripts/easy/01_stretch_2x.sh input.wav
bash scripts/easy/02_analyze_and_plot.sh input.wav out
bash scripts/easy/03_extract_single_moment.sh input.wav out/moments
bash scripts/easy/04_compare_kpis.sh input.wav out/kpi_compare 2.0
```

Notes:
- `01_stretch_2x.sh` uses FFmpeg.
- `02_*` and `03_*` use `esl`.
- `04_compare_kpis.sh` produces real KPI numbers from the input file.
