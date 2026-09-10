# JAES manuscript draft

This directory contains an Audio Engineering Society Journal engineering-report draft based on the official `jaes` template, version 2026.08.

## Artifacts

- `manuscript.tex`: editable manuscript source
- `jaes.cls`, `jaes.bst`: JAES template files, version 2026.08
- `references.bib`: bibliography database
- `evaluation.py`: deterministic evaluation and figure generator
- `results/`: generated CSV and JSON evidence, plus `numbers.tex`
- `results/numbers.tex`: every quantity quoted in the manuscript, as LaTeX macros;
  `manuscript.tex` inputs it, so no value in the paper is transcribed by hand
- `figures/`: generated PNG and EPS figures
- `SUBMISSION_CHECKLIST.md`: author checks required before submission

The official author guidance and template pages are:

- <https://aes.org/publications/journal/journal-author-guidelines/>
- <https://www.overleaf.com/latex/templates/journal-of-the-audio-engineering-society-jaes-template/sbvwbwjptgts>

## Regenerate the evaluation

From the repository root:

```sh
uv run python paper/evaluation.py
```

The full run takes roughly 15 minutes; `--quick` shortens the duration sweep for
smoke testing. `--numbers-only` regenerates `results/numbers.tex` from an
existing `evaluation_summary.json` without recomputing anything.

## Compile the manuscript

With a TeX Live installation:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=paper/build paper/manuscript.tex
```

The curated submission PDF is written to `output/pdf/EcoSignalLab_JAES_draft.pdf` after visual verification.

