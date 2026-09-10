# Author checks before JAES submission

## Must do before submitting

- [ ] **Re-run the evaluation on the machine you want to report.** The current
      `results/` and every number in the manuscript come from a Linux container,
      not from the macOS host named in earlier drafts. Run
      `uv run python paper/evaluation.py` on the host you intend to describe;
      the manuscript takes all quoted values from `results/numbers.tex`, so the
      text updates itself. Expect the memory floor and slope to change.
- [ ] **Verify every reference and DOI by resolving it.** Parts of this
      manuscript were drafted with AI assistance, and fabricated or mis-numbered
      citations are the characteristic failure mode of that workflow. The entries
      added in this revision (ISO 12913-2/-3, IEC 61260-1, EBU Tech 3341, Welch
      1967, Carter 1987, Bello 2005, Lundeby 1995, McFee 2015, Ulloa 2021, Sueur
      2008, Steinmetz/Reiss pyloudnorm) have **not** been resolved against the
      publisher record.
- [ ] **Confirm the AI-assistance disclosure against current AES policy.** The
      acknowledgment now names both tools and the scope of their contribution.
      Check whether AES requires the declaration in the submission system rather
      than (or as well as) the acknowledgment, and confirm the wording is
      acceptable.
- [ ] **Decide the submission track.** With the dispersion experiment, the
      validity matrix, and the streaming equivalence result, this is no longer
      naturally an Engineering Report. Consider the full-paper track.
- [ ] **Check whether review is anonymized for that track.** If it is, the GitHub
      URL in Sec. 6.3 de-anonymizes the author regardless of the title page;
      substitute an anonymized archive link for review.
- [ ] Confirm author name, project affiliation, and correspondence email.
- [ ] Replace the ecoSignalLab mark in the author biography with a portrait if
      the journal requires one.
- [ ] Add AES membership status/number and ORCID if applicable.
- [ ] Confirm whether there is funding, a conflict of interest, or additional
      contributors requiring acknowledgment or authorship.

## Strongly recommended

- [ ] **Archive the artifacts with a DOI.** A GitHub URL is mutable. Tag the
      release, mint a Zenodo DOI, and cite that instead of (or alongside) the
      repository URL.
- [ ] **Add cross-implementation validation.** Sec. 6.2 names its absence as the
      largest gap in the paper, and a referee will agree. The cheapest credible
      additions are the EBU Tech 3341 compliance signals against `libebur128` or
      `pyloudnorm`, and the IEC 61672 tolerance envelopes for the A/C weighting
      responses.
- [ ] Re-run `uv run pytest` from the exact commit to be submitted.

## Known cosmetic build issues (not errors)

- The `jaes` class hardcodes the "DRAFT" mark on page 1 and the "DRAFT
  MANUSCRIPT" running head. There is no class option to disable them; this is
  the template's intended appearance for a submitted manuscript.
- `\biography` produces a 252 pt overfull `\hbox` at `\end{document}`. This comes
  from the class's `strip` environment, is present with the stock template, and
  does not affect the rendered page.
- BibTeX warns about missing `month` fields and one empty `doi`
  (`krause2008`, which has a URL instead). Harmless.

## What changed in this revision

Software (in `src/esl/`, all covered by the existing test suite):

- `interchannel_coherence` renamed to `interchannel_correlation_zerolag`; the old
  name is a deprecated alias that emits a warning into the provenance record.
- New `interchannel_coherence_msc`: a genuine Welch magnitude-squared coherence.
- Frame-aligned streaming (`AnalysisConfig.stream_continuity`, default on) makes
  chunked analysis bit-identical to full-file analysis.
- `AnalysisConfig.channel_pair` / `channel_semantics`: paired metrics honour a
  declared channel pair and record whether it was declared or inferred.
- Validity gating added for interchannel delay (GCC-PHAT peak-to-sidelobe ratio),
  spectral centroid (silence), decay times (dynamic range, slope, fit quality),
  and interchannel correlation (channel count, energy floor).
- Normalized correlation coefficients clamped to their theoretical range.

Paper:

- `evaluation.py` rewritten around experiments that can fail; it now also emits
  `results/numbers.tex`, from which the manuscript takes every quoted value.
- Figures regenerated at true single-column width (3.25 in), so they are no
  longer scaled down in LaTeX.
