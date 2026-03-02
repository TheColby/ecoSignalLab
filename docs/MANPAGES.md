# Man Pages

Quick links: [Docs Index](INDEX.md) | [Getting Started](GETTING_STARTED.md) | [Task Recipes](TASK_RECIPES.md) | [Troubleshooting](TROUBLESHOOTING.md) | [Schema](SCHEMA.md) | [Metrics](METRICS_REFERENCE.md) | [References](REFERENCES.md)

`esl` ships installable man pages and an install script that copies them into the active prefix.

## Install

From repo root:

```bash
bash scripts/install.sh
```

This installs the package and places man pages under:

```text
<prefix>/share/man/man1
```

## Use

```bash
MANPATH="<prefix>/share/man:$MANPATH" man esl
MANPATH="<prefix>/share/man:$MANPATH" man esl-analyze
MANPATH="<prefix>/share/man:$MANPATH" man esl-batch
MANPATH="<prefix>/share/man:$MANPATH" man esl-moments
MANPATH="<prefix>/share/man:$MANPATH" man esl-similar
MANPATH="<prefix>/share/man:$MANPATH" man esl-calibrate
```

## Installed Pages

- `esl(1)`
- `esl-analyze(1)`
- `esl-batch(1)`
- `esl-moments(1)`
- `esl-similar(1)`
- `esl-calibrate(1)`
