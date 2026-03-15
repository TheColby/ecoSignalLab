# Troubleshooting

Quick links: [Docs Index](INDEX.md) | [Getting Started](GETTING_STARTED.md) | [Task Recipes](TASK_RECIPES.md) | [Troubleshooting](TROUBLESHOOTING.md) | [Schema](SCHEMA.md) | [Metrics](METRICS_REFERENCE.md) | [References](REFERENCES.md)

If a command failed, find the exact message below and use the fix command.

First move if you are not sure what is broken:

```bash
esl doctor input.wav
```

## `zsh: command not found: esl`

Cause:
- virtual environment is not active, or package not installed in editable mode.

Fix:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Check:

```bash
esl --help
```

## `No module named esl`

Cause:
- running outside environment or missing install.

Fix:

```bash
source .venv/bin/activate
pip install -e .
```

## FFmpeg decode errors for MP3/AAC/OGG

Cause:
- `ffmpeg` or `ffprobe` missing from `PATH`.

Fix:
- install FFmpeg for your OS
- verify:

```bash
ffmpeg -version
ffprobe -version
```

## `Input file not found`

Cause:
- path typo, wrong folder, or missing extension.

Fix:

```bash
ls -lah
```

Then rerun command with the exact file path.

## Very large WAV behaves unexpectedly (near/over 4 GB)

Cause:
- classic RIFF/WAV containers have 32-bit size fields and can hit a 4 GB boundary.

Fix:
- use RF64 container for long/high-rate/multichannel capture.
- review large-file workflow guidance in [`RF64_AND_LARGE_FILES.md`](RF64_AND_LARGE_FILES.md).
- for huge scans, prefer out-of-core analysis:
  `esl analyze input_long.wav --out-dir out --chunk-hours 1 --streamable-only --summary-only --frame-table-csv out/frame_table.csv --checkpoint-dir out/checkpoints --resume`

## `esl moments extract` produced no clips

Cause:
- thresholds too strict, or ranking metric has weak events.

Try this first:

```bash
esl moments extract input.wav \
  --out out/moments \
  --single \
  --rank-metric novelty_curve \
  --event-window 8
```

If still empty:
- lower thresholds in your `--rules` file
- or skip rules entirely and rank directly with `--rank-metric novelty_curve`
- increase `--event-window`
- test with `--top-k 5`
- for huge RF64/CAF archives, use the two-pass workflow in [Task Recipes: Recipe 14](TASK_RECIPES.md#recipe-14-find-the-top-33-most-novel-moments-in-a-ten-year-8-channel-file)

## Plots not opening with `--show`

Cause:
- no desktop opener in the environment, or headless session.

Fix:
- open files manually from the output folder
- or keep using static outputs (`.png`, `.html`)

## I expected creative transformation effects

Cause:
- `esl` is an acoustic analysis SDK, not a creative effects processor.

What to do:
- use `esl` commands when you need metrics, moments, plots, and reproducible exports
- use dedicated transformation tools for creative editing workflows
- see analysis-first recipes in [`TASK_RECIPES.md`](TASK_RECIPES.md)

## Related Docs

- [`GETTING_STARTED.md`](GETTING_STARTED.md)
- [`TASK_RECIPES.md`](TASK_RECIPES.md)
- [`GLOSSARY.md`](GLOSSARY.md)
- [`RF64_AND_LARGE_FILES.md`](RF64_AND_LARGE_FILES.md)
