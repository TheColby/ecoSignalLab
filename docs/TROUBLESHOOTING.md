# Troubleshooting

If a command failed, find the exact message below and use the fix command.

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
- increase `--event-window`
- test with `--top-k 5`

## Plots not opening with `--show`

Cause:
- no desktop opener in the environment, or headless session.

Fix:
- open files manually from the output folder
- or keep using static outputs (`.png`, `.html`)

## I only want quick sound changes, not analysis

Use task recipes:
- [`TASK_RECIPES.md`](TASK_RECIPES.md)

Quick stretch command:

```bash
ffmpeg -i input.wav -filter:a "atempo=0.5" output_2x.wav
```

## Related Docs

- [`GETTING_STARTED.md`](GETTING_STARTED.md)
- [`TASK_RECIPES.md`](TASK_RECIPES.md)
- [`GLOSSARY.md`](GLOSSARY.md)
