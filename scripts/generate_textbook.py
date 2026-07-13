#!/usr/bin/env python3
"""Build the curated ecoSignalLab textbook Markdown, HTML, and PDF artifacts.

The textbook deliberately reuses the maintained documentation chapters instead
of copying their content into a second, manually divergent book. It adds a
course-style structure and learning objectives, then uses esl's documentation
renderer so the PDF preserves Mermaid, MathJax, and hyperlinks.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from esl.docsgen import build_docs

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "docs" / "TEXTBOOK.md"
DEFAULT_PDF = ROOT / "TEXTBOOK.pdf"
TEXTBOOK_LOGO = ROOT / "assets" / "logos" / "esl" / "minimal" / "esl_logo_04_circle_wave.png"


@dataclass(frozen=True, slots=True)
class Chapter:
    source: Path
    title: str
    objectives: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Figure:
    """One image embedded in a textbook chapter."""

    chapter_number: int
    figure_number: int
    caption: str

    @property
    def label(self) -> str:
        """Return the stable chapter-figure label used by the textbook."""
        return f"{self.chapter_number}-{self.figure_number}"


PARTS: tuple[tuple[str, tuple[Chapter, ...]], ...] = (
    (
        "Part I - Foundations and First Measurements",
        (
            Chapter(
                ROOT / "docs" / "USERGUIDE.md",
                "Orientation and First Workflows",
                (
                    "Identify the smallest complete esl workflow for a new file.",
                    "Distinguish digital level, calibrated level, and analysis provenance.",
                    "Choose an appropriate first output product for a practical question.",
                ),
            ),
            Chapter(
                ROOT / "docs" / "SIGNAL_WINDOWS_VISUAL_GUIDE.md",
                "Signals, Frames, Windows, and Chunks",
                (
                    "Relate frame and hop sizes to time and frequency resolution.",
                    "Explain why chunks manage memory while frames define features.",
                    "Recognize common windowing and overlap concepts in esl outputs.",
                ),
            ),
            Chapter(
                ROOT / "docs" / "METRICS_REFERENCE.md",
                "Metric Families and Measurement Contracts",
                (
                    "Select metrics from a stated question rather than by accumulation.",
                    "Interpret units, aggregation semantics, confidence, and streamability.",
                    "Locate mathematical definitions and algorithm references for every metric.",
                ),
            ),
        ),
    ),
    (
        "Part II - Events, Change, and Similarity",
        (
            Chapter(
                ROOT / "docs" / "NOVELTY_ANOMALY.md",
                "Novelty, Boundaries, and Anomaly Scores",
                (
                    "Explain the distinction between novelty, anomaly, and importance.",
                    "Choose between a similarity matrix and a novelty matrix.",
                    "Use candidate scores as a listening and review workflow.",
                ),
            ),
            Chapter(
                ROOT / "docs" / "MOMENTS_EXTRACTION.md",
                "Moments Extraction and Reviewable Event Clips",
                (
                    "Extract the single most novel event or a ranked set of events.",
                    "Control pre-event and post-event listening context.",
                    "Interpret per-channel and downmix ranking choices.",
                ),
            ),
            Chapter(
                ROOT / "docs" / "SIMILARITY_SEARCH.md",
                "Query by Example and Distance Functions",
                (
                    "Choose a feature, metric, or multi-metric similarity mode.",
                    "Understand cosine, Euclidean, and Manhattan distance behavior.",
                    "Read ranked search results without confusing similarity with identity.",
                ),
            ),
        ),
    ),
    (
        "Part III - Long Archives, Spatial Audio, and Field Operations",
        (
            Chapter(
                ROOT / "docs" / "RF64_AND_LARGE_FILES.md",
                "RF64 and Out-of-Core Analysis",
                (
                    "Recognize when classic WAV limits require RF64 or sharding.",
                    "Plan chunked analysis without assuming an entire recording fits in RAM.",
                    "Estimate storage and operational implications of long recordings.",
                ),
            ),
            Chapter(
                ROOT / "docs" / "SHARD_WORKFLOWS.md",
                "Shard Manifests and Archive-Scale Retrieval",
                (
                    "Build an ordered archive manifest with relative and absolute time.",
                    "Run resumable archive analysis and find novel or query-like moments.",
                    "Use spatial metadata sidecars and profile spatial retrieval cost.",
                ),
            ),
            Chapter(
                ROOT / "docs" / "INSIGHTS.md",
                "Archive Insights and Soundscape Reporting",
                (
                    "Generate scene-change, calmness, occupancy, drift, and storyboard views.",
                    "Treat insight outputs as explicit methods with assumptions.",
                    "Choose archive-level reports that remain inspectable at large scale.",
                ),
            ),
        ),
    ),
    (
        "Part IV - ML, Schema, Validation, and Scientific Practice",
        (
            Chapter(
                ROOT / "docs" / "ML_FEATURES.md",
                "FrameTables, Tensors, and Dataset Manifests",
                (
                    "Use the canonical FrameTable contract for tabular and tensor workflows.",
                    "Choose appendable CSV, Parquet, or HDF5 for large frame data.",
                    "Create deterministic dataset manifests while avoiding leakage.",
                ),
            ),
            Chapter(
                ROOT / "docs" / "SCHEMA.md",
                "Schema, Provenance, and Calibration Contracts",
                (
                    "Read schema versions, decoder provenance, and pipeline hashes.",
                    "Identify calibration and spatial metadata assumptions in outputs.",
                    "Use output contracts as a review and reproducibility tool.",
                ),
            ),
            Chapter(
                ROOT / "docs" / "VALIDATION.md",
                "Validation and Regression Practice",
                (
                    "Use reference fixtures and tolerance-aware assertions responsibly.",
                    "Separate numerical correctness from scientific interpretation.",
                    "Design a repeatable quality-control run for a deployment or release.",
                ),
            ),
            Chapter(
                ROOT / "docs" / "REFERENCES.md",
                "References, Attribution, and Further Study",
                (
                    "Locate the research and standards behind esl algorithms.",
                    "Distinguish original implementation work from cited external methods.",
                    "Build a defensible reading list for an acoustic-analysis project.",
                ),
            ),
        ),
    ),
)


# These cases keep the textbook's extended study material tied to the actual
# subject of each chapter rather than presenting a generic DSP essay 13 times.
CHAPTER_CASES: tuple[str, ...] = (
    "a new recording whose history and intended use are only partly known",
    "a signal whose time scale, frame boundaries, and spectral resolution must be chosen deliberately",
    "a metric report in which units, aggregation rules, and confidence determine whether values can be compared",
    "a score that marks change in a recording without claiming that the change is important",
    "a ranked event clip that must be reviewed with enough context to understand its cause",
    "a query-by-example result whose rank depends on the feature representation and distance geometry",
    "a long file whose storage, decoder behavior, and processing strategy cannot be treated as incidental",
    "an ordered archive whose gaps, clock assumptions, and shard boundaries affect every trend",
    "a long-term report that turns a large archive into reviewable evidence without hiding its aggregation choices",
    "a feature export that must remain meaningful after it becomes a table, tensor, or dataset split",
    "an output record whose provenance and calibration state determine how a later analyst may reuse it",
    "a regression or field check where expected behavior, tolerance, and failure evidence must be distinguished",
    "a literature and attribution trail that lets a reader trace a method back to its sources",
)


STUDY_LENSES: tuple[tuple[str, str], ...] = (
    ("The question before the command", "state the decision the analysis is supposed to support before selecting a metric or option"),
    ("The observed object", "separate the decoded sample sequence from the physical scene, source, or event it may represent"),
    ("Time scale", "choose a duration, frame, hop, chunk, or review interval that matches the phenomenon under study"),
    ("Channel semantics", "record whether the observation belongs to one channel, an aggregate, a downmix, or a spatial convention"),
    ("Units and references", "name the unit, reference quantity, calibration state, and any conversion that makes a value interpretable"),
    ("Preprocessing as a method", "treat decoding, resampling, filtering, normalization, and segmentation as visible analytical choices"),
    ("Parameter sensitivity", "test whether a conclusion survives a reasonable change in one important parameter"),
    ("The visual check", "compare the numerical output with a plot and, when appropriate, a bounded listening review"),
    ("The alternative explanation", "identify a plausible artifact, confound, or data-quality issue that could mimic the pattern"),
    ("Validity and confidence", "read flags, fit quality, uncertainty, and missing context before elevating a value into a claim"),
    ("The comparison set", "define what is held constant and what is allowed to differ when two results are compared"),
    ("The audit trail", "retain commands, resolved configuration, versions, inputs, and output identifiers together"),
    ("The human review loop", "give a listener, analyst, or domain expert a concrete artifact and a stated question to inspect"),
    ("Scale and cost", "estimate memory, storage, runtime, and review burden before committing to a large workflow"),
    ("Interoperability", "preserve enough schema, metadata, and naming information for a handoff to another tool or collaborator"),
    ("Ethics and access", "consider permissions, sensitive locations, speech, cultural material, and the difference between analysis and surveillance"),
    ("Failure as evidence", "keep a failed decode, invalid fit, clipped channel, or unexpected rank when it teaches something about the method"),
    ("A bounded conclusion", "write the smallest claim that the observed artifact, method record, and review actually support"),
    ("Replication", "make a second run possible without relying on memory, undocumented defaults, or a private graphical state"),
    ("The next measurement", "turn remaining uncertainty into a practical follow-up recording, calibration check, annotation, or comparison"),
    ("Stakeholders and use", "identify who will act on the result and what decision, design, or research question it is meant to inform"),
    ("Acoustic scene versus recording", "separate properties of the measured waveform from inferences about a place, source, listener, or system"),
    ("Instrumentation boundary", "document microphone, recorder, placement, gain, timing, and deployment facts that software cannot infer"),
    ("Reference baseline", "define a known interval, fixture, variant, or population against which change or similarity is judged"),
    ("Sampling and selection", "record how the chosen files or intervals entered the analysis and which material was excluded"),
    ("Feature representation", "explain what the selected representation emphasizes, compresses, or ignores before interpreting a score"),
    ("Normalization policy", "state whether values are normalized across frames, channels, files, or a corpus and how that changes comparison"),
    ("Aggregation hierarchy", "track the path from samples to frames, clips, channels, shards, days, or projects without losing level of meaning"),
    ("Thresholds and ranking", "treat cutoffs, peaks, top-k lists, and exclusion rules as declared decision policies rather than discoveries"),
    ("Missingness and gaps", "preserve absent channels, incomplete files, clock gaps, decode failures, and unavailable calibration as information"),
    ("Outliers and rare events", "distinguish a valuable rare observation from a transient artifact, configuration change, or data-entry mistake"),
    ("Model boundaries", "separate an implementation-aligned measurement, a research feature, and a proxy from a certified result"),
    ("Independent evidence", "seek field notes, labels, simulations, repeated measurements, or external observations that can challenge the output"),
    ("Version changes", "compare results across software, library, model, or configuration versions before treating a trend as signal change"),
    ("Storage and retention", "plan which raw audio, intermediate features, plots, and provenance records must be retained for later review"),
    ("Collaboration protocol", "make channel names, time bases, vocabulary, review labels, and expected deliverables legible to another analyst"),
    ("Privacy and consent", "consider whether speech, location, cultural material, or sensitive wildlife data requires access controls or redaction"),
    ("Communication", "choose a table, plot, clip, methods paragraph, or uncertainty statement that matches the audience without hiding caveats"),
    ("Decision reversibility", "prefer a workflow that permits review, reranking, re-extraction, and correction when later evidence changes the picture"),
    ("Long-term stewardship", "prepare outputs so that a future analyst can understand their provenance even after hardware, accounts, and personnel change"),
)


def _extended_chapter_study(chapter_number: int, chapter: Chapter) -> str:
    """Build a detailed, chapter-specific reading dossier.

    The repeated analytical frame intentionally develops a transferable method,
    while the case, objectives, and lens make its application specific to the
    source chapter. It is prose, not a shortcut around the maintained technical
    documentation that precedes it.
    """
    try:
        case = CHAPTER_CASES[chapter_number - 1]
    except IndexError as exc:  # Keep additions to PARTS explicit and reviewed.
        raise RuntimeError("Every textbook chapter needs an extended-study case.") from exc

    objectives = "; ".join(chapter.objectives)
    lines = [
        "## Extended Study Dossier",
        "",
        "This dossier slows the chapter down on purpose. The maintained chapter above defines the "
        "software contract, formulas, commands, and citations. The pages below develop the habits "
        "that determine whether those tools produce evidence: asking a bounded question, preserving "
        "conditions, looking for failure modes, and writing a conclusion that does not outrun its "
        "data. Read one lens at a time while working with a real but manageable example.",
        "",
        "Before the formal model below, note that it is a conceptual accounting device rather than a "
        "new metric. It states that an analytical claim depends jointly on the audio, chosen method, "
        "context, and review. Leaving one term undocumented does not make it disappear; it makes the "
        "result harder to interpret later.",
        "",
        "$$",
        r"\mathcal{C} = \mathcal{I}(x; \theta, \kappa, \pi, r)",
        "$$",
        "",
        "where `C` is a bounded claim, `x` is the decoded audio, `theta` is the resolved analysis "
        "configuration, `kappa` is the recorded context such as calibration and channel metadata, "
        "`pi` is the provenance record, and `r` is the human or automated review procedure. Plain "
        "English: the same waveform can support different, equally valid statements when the question "
        "and the recorded method differ.",
        "",
        "```mermaid",
        "flowchart LR",
        '    A["Question"] --> B["Declared method"]',
        '    B --> C["Artifact and flags"]',
        '    C --> D["Review"]',
        '    D --> E["Bounded claim"]',
        '    E -. "new uncertainty" .-> A',
        "```",
        "",
    ]
    for lens_number, (lens, action) in enumerate(STUDY_LENSES, start=1):
        objective = chapter.objectives[(lens_number - 1) % len(chapter.objectives)]
        lines.extend(
            (
                f"### {lens}",
                "",
                f"In Chapter {chapter_number}, **{chapter.title}**, begin with {case}. The practical "
                f"discipline for this lens is to {action}. This is not paperwork added after the signal "
                "processing is complete. It decides which result can be compared, which output deserves "
                "review, and which conclusion would be too strong. The chapter's learning objectives are "
                f"to {objectives}. For this pass, concentrate on one objective: {objective}",
                "",
                "Work with an example small enough to inspect from end to end. Write down the input "
                "identifier, the time interval, the channel policy, the configuration, and the expected "
                "shape of a useful result before you run the command. Then retain the unedited output and "
                "compare it with one visual or listening observation. The comparison should be literal at "
                "first: a peak at a timestamp, a changed rank, a decay slope, a missing field, or a stable "
                "block in a matrix. Literal observations are easier to audit than immediate stories about "
                "cause, quality, species, listener experience, or design success.",
                "",
                "Next, make one controlled challenge to the first result. Change a parameter only when the "
                "chapter's method makes that change meaningful, or select a second documented interval, "
                "channel, file, or simulation variant. Ask what stayed stable, what moved, and whether the "
                "difference is larger than a plotting, framing, decoding, or calibration uncertainty. A "
                "result that changes is not automatically wrong; it may be telling you that the question "
                "has a scale, representation, or data-quality dependency that belongs in the final report.",
                "",
                "Finally, write a two-part note. The first sentence states what the artifact shows under the "
                "recorded conditions. The second states one thing it does not establish and the next piece "
                "of evidence that would reduce that uncertainty. This habit turns every chapter into a "
                "reusable analytical protocol instead of a collection of software features. It also gives "
                "a collaborator a precise place to disagree productively: the input, configuration, "
                "interpretation, or required validation step rather than the vague claim that a number "
                "'looks wrong.'",
                "",
            )
        )
    return "\n".join(lines).rstrip()


def _strip_document_title(markdown: str) -> str:
    """Remove the source document's first H1; the textbook supplies one."""
    return re.sub(r"\A\s*# [^\n]+\n+", "", markdown, count=1)


def _slugify(value: str) -> str:
    """Mirror the simple heading-anchor convention used by Markdown output."""
    normalized = re.sub(r"[^a-z0-9 -]", "", value.lower())
    return re.sub(r"[-\s]+", "-", normalized).strip("-")


_IMAGE_PATTERN = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", flags=re.MULTILINE)


def _number_figures(chapter_number: int, markdown: str) -> tuple[str, tuple[Figure, ...]]:
    """Replace Markdown images with numbered, hyperlinkable HTML figures.

    Markdown itself has no portable figure-caption contract. The generated HTML
    keeps the source image URL unchanged, while this small wrapper gives the PDF
    and HTML editions the same stable ``chapter-figure`` identifiers.
    """
    figures: list[Figure] = []

    def replace(match: re.Match[str]) -> str:
        figure_number = len(figures) + 1
        figure = Figure(
            chapter_number=chapter_number,
            figure_number=figure_number,
            caption=match.group(1).strip() or "Untitled figure",
        )
        figures.append(figure)
        image_path = html.escape(match.group(2).strip(), quote=True)
        caption = html.escape(figure.caption)
        return "\n".join(
            (
                f'<figure id="figure-{figure.label}" class="textbook-figure">',
                f'  <img src="{image_path}" alt="{caption}" />',
                f'  <figcaption><strong>Figure {figure.label}.</strong> {caption}</figcaption>',
                "</figure>",
            )
        )

    return _IMAGE_PATTERN.sub(replace, markdown), tuple(figures)


def _chapter_markdown(chapter_number: int, chapter: Chapter) -> tuple[str, tuple[Figure, ...]]:
    if not chapter.source.exists():
        raise FileNotFoundError(f"Textbook source is missing: {chapter.source}")
    source_markdown = _strip_document_title(chapter.source.read_text(encoding="utf-8")).strip()
    source_markdown, figures = _number_figures(chapter_number, source_markdown)
    source_rel = chapter.source.relative_to(ROOT).as_posix()
    objectives = "\n".join(f"- {objective}" for objective in chapter.objectives)
    study_questions = "\n".join(
        f"{index}. Demonstrate that you can: {objective}"
        for index, objective in enumerate(chapter.objectives, start=1)
    )
    chapter_markdown = "\n".join(
        (
            f"# Chapter {chapter_number}: {chapter.title}",
            "",
            f"**Maintained source chapter:** [`{source_rel}`]({source_rel.removeprefix('docs/')})",
            "",
            "## Learning Objectives",
            "",
            objectives,
            "",
            "## Why This Chapter Matters",
            "",
            f"{chapter.title} is not a collection of commands to memorize. It establishes the "
            "measurement choices that make a later result interpretable: the representation of "
            "the audio, the time scale, the channel policy, the units, and the provenance that "
            "allows another person to inspect the work. Treat those choices as part of the result "
            "rather than backstage detail.",
            "",
            "As you read, connect each method to a concrete decision you expect to make with an "
            "audio file. Ask what the method observes, what it deliberately ignores, and which "
            "assumption would change the conclusion. The aim is not to produce an impressive "
            "number quickly; it is to make a claim that survives a second look.",
            "",
            "## Reading Strategy",
            "",
            "Read the formulas, tables, and diagrams together. A formula defines a quantity; a "
            "table specifies units and aggregation; a diagram shows where the calculation fits "
            "in the workflow. When the chapter provides a command, run it on a small known file "
            "first, preserve the output JSON, and compare the visible plot with what you hear. "
            "That small loop is more valuable than a large unreviewed batch run.",
            "",
            "## Chapter Text",
            "",
            source_markdown,
            "",
            _extended_chapter_study(chapter_number, chapter),
            "",
            "## Check Your Understanding",
            "",
            study_questions,
            "",
            "## Practice Prompt",
            "",
            "Choose a small, legal-to-share recording or a documented project artifact. Apply one workflow from this chapter, preserve its command and output JSON, then write down one assumption that could change the interpretation. This is deliberately less glamorous than a demo and much closer to real acoustic practice.",
        )
    )
    return chapter_markdown, figures


HOMEWORK: tuple[tuple[str, str, str], ...] = (
    ("Inspect before measuring", "Run `esl doctor` and `esl analyze` on one WAV or FLAC file. Identify the decoder, sample rate, channel count, and duration.", "A 250-word provenance note plus the JSON output."),
    ("Peak, RMS, and crest", "Create or locate a short sine, transient, and quiet recording. Compare peak, RMS, and crest factor using the same analysis settings.", "One table and a two-sentence explanation of each difference."),
    ("Window trade-off", "Analyze the same clip with a short and a long frame/window. Compare the time and frequency detail in the plots.", "Two plots with captions that state the trade-off."),
    ("Hop-size sensitivity", "Hold the window fixed and change only hop size. Determine which values change computational cost and which change the underlying frame estimate.", "A command log and a concise conclusion."),
    ("Calibration boundary", "Run an uncalibrated analysis, then configure a documented 0 dBFS mapping or calibration profile if one is available. Do not invent a calibration value.", "A before/after units table and a statement of what remains unknown."),
    ("Clipping audit", "Find or synthesize a clipped waveform and verify that the validity flag changes. Compare its crest factor with an unclipped reference.", "JSON excerpts and a short risk assessment."),
    ("DC-offset audit", "Construct a signal with a known DC offset, run the quality-control metrics, and explain why offset matters to some level estimates.", "A reproducible script or command sequence and result table."),
    ("Impulse-response decay", "Generate or obtain a safe, documented impulse response. Estimate EDT and RT60 and inspect the fit-quality fields.", "Decay plot, estimates, and a paragraph on fit validity."),
    ("Clarity comparison", "Compare C50, C80, and D50 for two impulse responses or two simulated variants. State which source/receiver assumptions differ.", "A design-variant comparison table."),
    ("Novelty candidate", "Use `esl moments extract --single` on a recording. Listen to the resulting clip and decide whether the top score is actually useful.", "The WAV, CSV row, and a reviewer note."),
    ("Top-k review", "Extract ten novel moments with non-overlap and a stated pre/post context. Label each as useful, duplicate, artifact, or uncertain.", "An annotated ten-row CSV."),
    ("Similarity matrix", "Render a similarity matrix for a clip with at least two contrasting sections. Mark two high-similarity and two low-similarity regions.", "The plot and a listening-based interpretation."),
    ("Novelty matrix", "Render the Foote-style novelty matrix and vary the kernel size. Explain one expected and one surprising change.", "Two figures and a parameter record."),
    ("Distance functions", "Search a folder with cosine, Euclidean, and Manhattan distance. Keep the feature set fixed and compare the top five ranks.", "A rank-agreement table."),
    ("Musical analysis: form, timbre, and harmonic color", "Choose a legal-to-share music excerpt. Extract `librosa` features, render similarity and novelty matrices, and use listening to identify one repeated passage, one boundary, and one timbral or harmonic-color contrast. Do not claim chord, key, beat, or transcription labels.", "Feature metadata, two annotated plots, a timestamped listening log, and a 500-word analysis that states feature/back-end assumptions."),
    ("Stereo semantics", "Analyze a two-channel file whose channels differ. Compare per-channel output, aggregate output, and a downmix-led event ranking.", "A channel-semantics diagram or table."),
    ("Ambisonics sidecar", "Prepare a spatial metadata sidecar for a documented B-format file or provided fixture. Validate that order, convention, and labels are recorded.", "The sidecar and provenance JSON excerpt."),
    ("Archive manifest", "Build a shard manifest from at least three chronologically ordered files. Include a calendar start only when it is truly known.", "The manifest and a timeline screenshot."),
    ("Calendar rollup", "Produce daily or hourly archive summaries. Explain the difference between archive-relative time and civil time.", "One plot and a time-zone policy note."),
    ("Large-file estimate", "Estimate storage and processing implications for a 24-hour, multichannel recording at a chosen rate and format.", "A calculation sheet using GB and an operational plan."),
    ("RF64 decision", "Explain whether a proposed long WAV should use classic WAV, RF64, FLAC, or sharded files. Include one interoperability constraint.", "A one-page format decision memo."),
    ("FrameTable inspection", "Export a FrameTable and inspect timestamps, feature names, and metadata. Check that column names are deterministic.", "A schema excerpt and a five-row sample."),
    ("Tensor layout", "Export a tensor where available and verify the declared `[channels, frames, features]` layout against the tabular export.", "Shape evidence and a mapping note."),
    ("Leakage-resistant split", "Create a dataset manifest with a split strategy that prevents adjacent or same-deployment shards from leaking across splits.", "Manifest plus a leakage audit."),
    ("Anomaly baseline", "Run a novelty or anomaly workflow on a recording with an intentionally documented reference period. Separate unusual from merely loud.", "A plot and a false-positive discussion."),
    ("Ecoacoustic bands", "Construct or select material with energy in both anthropogenic and biophonic bands. Check the NDSI sign and band assumptions.", "A band-definition table and a sanity-check result."),
    ("Index skepticism", "Choose one ecoacoustic index and read two cited papers that report limitations. Propose a validation observation independent of the index.", "A 500-word critique with citations."),
    ("Long-term drift", "Analyze a repeated recording set or simulated drift sequence. Separate likely sensor drift from plausible environmental change.", "A trend plot and competing hypotheses."),
    ("Architectural variants", "Analyze two room-simulation outputs as variants. Compare decay and clarity without claiming compliance unless the measurement assumptions support it.", "A technical comparison report."),
    ("Decoder provenance", "Analyze one supported compressed file and one lossless file. Record decoder provenance and describe one reason decoder context belongs in results.", "A provenance comparison table."),
    ("Regression fixture", "Create a small synthetic fixture for one metric and write a tolerance-based test or pseudo-test plan.", "Fixture description, expected values, and tolerances."),
    ("Reproducible handoff", "Package one analysis with input identifiers, config, pipeline hash, outputs, plots, and a README for another analyst.", "A handoff directory and a reproduction checklist."),
    ("Capstone: defensible claim", "Use a small archive or simulation set to make one narrow acoustic claim. State the data, method, uncertainty, counterexample, and next validation step.", "A 1,000-word report with commands, figures, and references."),
)


@dataclass(frozen=True, slots=True)
class HomeworkGuide:
    """Pedagogical material that turns a short prompt into a complete lab."""

    purpose: str
    control: str
    question: str
    extension: str


HOMEWORK_GUIDES: tuple[HomeworkGuide, ...] = (
    HomeworkGuide("Provenance is the minimum context needed to interpret a number later. This lab makes the file and decoder part of the measurement, not an afterthought.", "Keep the same input but compare the `doctor` inspection with the recorded analysis provenance. Note every field that is inferred rather than supplied by you.", "Which later comparison would become impossible if decoder, channel count, or sample rate were absent?", "Repeat with a compressed file and explain how its decoder provenance differs from the lossless file."),
    HomeworkGuide("Peak, RMS, and crest factor describe different aspects of level. A signal can be quiet on average while still producing a large instantaneous excursion.", "Use equal-duration clips and do not normalize them after selection. Keep the sample rate, window, and channel policy fixed.", "Which metric best distinguishes the transient from the steady sine, and why does that not make it a loudness measure?", "Add a fourth clip with broadband noise and describe where it falls in the comparison."),
    HomeworkGuide("A frame length is a scientific choice: shorter frames localize change, while longer frames resolve frequency more clearly. There is no universal correct value.", "Change only the frame/window setting. Preserve the same source region, hop ratio, feature set, color scale, and plot dimensions.", "Which event becomes easier to localize with the short frame, and which spectral structure becomes clearer with the long frame?", "Try a middle setting and justify a final choice for a specific question, not for general beauty."),
    HomeworkGuide("Hop size controls how densely a workflow samples time. It can change compute and temporal sampling without changing the analysis window itself.", "Keep frame length, source, metric list, and all preprocessing fixed. Record elapsed time and the number of generated frames.", "Which result is genuinely a different estimate, and which result is merely a denser representation of the same windowed analysis?", "Create a small plot of frame count versus hop size and state where diminishing returns begin."),
    HomeworkGuide("Calibration converts a digital representation into a stated physical-level assumption. It is useful only when the supporting calibration information is real and retained.", "Use the identical file and settings for both runs. Record the calibration source, weighting, date, and every field that remains unavailable.", "Which outputs may now be compared in physical-level terms, and which remain only digital measurements?", "Write a calibration intake form for a future recorder deployment, including a reference-tone verification step."),
    HomeworkGuide("Clipping is not merely ugly; it changes peaks, spectra, and some derived metrics. Validity flags should be reviewed before interpretation.", "Use matched unclipped and clipped versions of the same source. Document the clipping method or source condition and avoid automatic normalization.", "How do the waveform, crest factor, and clipping flag agree or disagree?", "Test a near-clipped signal and propose a conservative operational threshold for flagging review."),
    HomeworkGuide("DC offset wastes headroom and can bias level or low-frequency interpretation. This is a quality-control question before it is a signal-processing trick.", "Start from a zero-mean synthetic or documented source. Add a known constant offset without changing its other content.", "Which reported quantity reveals the offset most directly, and which metric could be misread if the offset were ignored?", "Compare a deliberate offset with a low-frequency tone and explain why they are not interchangeable."),
    HomeworkGuide("Decay metrics rely on a defensible decay region, not a magical regression line. Fit quality and tail noise determine whether an RT estimate deserves trust.", "Hold impulse-response length, onset alignment, and analysis band assumptions fixed. Preserve the raw decay curve and fit diagnostics.", "What evidence shows that the chosen decay interval is linear enough for the reported RT60 or EDT?", "Add noise to the tail of a synthetic decay and document when the fit becomes unreliable."),
    HomeworkGuide("C50, C80, and D50 partition early and late energy differently. They become architectural evidence only with explicit source, receiver, and simulation assumptions.", "Use the same analysis thresholds for both variants and record any differences in source position, receiver position, materials, or rendering.", "Which metric changes most between the variants, and what physical or design explanation is plausible?", "Repeat the comparison in one octave or band-limited region if the source material permits it."),
    HomeworkGuide("Novelty is a candidate-generation score, not an automatic claim that a sound is important. Listening is part of the validation loop.", "Fix the novelty configuration and clip context. Save the source timestamp, rank score, and resulting extract without editing it.", "Does the highest-scoring moment contain a meaningful event, a boundary, an artifact, or a mixture of these?", "Have a second listener review the same clip without seeing the score, then compare judgments."),
    HomeworkGuide("A ranked event list becomes useful when it is reviewable and nonredundant. Context windows and non-overlap rules define what a reviewer actually hears.", "Use the same rank metric and a stated minimum separation for all ten clips. Keep the pre-event and post-event durations visible in the CSV.", "How many high-ranked clips describe the same underlying event, and how does the separation rule affect that count?", "Create a reviewer guide with explicit labels and examples for artifact, duplicate, uncertain, and useful."),
    HomeworkGuide("A self-similarity matrix shows recurrence and contrast across time. It should be checked against listening, not treated as a visual Rorschach test.", "Use a clip with documented section boundaries or create one from clearly different source regions. Hold features, normalization, and distance policy fixed.", "Which diagonal blocks correspond to repeated material, and what acoustic evidence supports that interpretation?", "Compare a spectral feature representation with a temporal one and describe the changed structure."),
    HomeworkGuide("Foote-style novelty derives local change from a similarity structure and a checkerboard kernel. Kernel scale defines what counts as a boundary.", "Keep the source, features, normalization, and peak-picking policy fixed while changing only kernel size.", "Which boundary persists across scales, and which candidate appears only at one scale?", "Overlay manually marked boundaries or listening notes to assess precision and missed events."),
    HomeworkGuide("Distance functions encode different notions of closeness. A ranking that changes with distance is telling you the feature geometry matters.", "Use one query, one candidate folder, and one fixed feature representation. Normalize inputs identically before each search.", "Which candidates remain stable across all three distances, and which are metric-sensitive?", "Add a clearly irrelevant file and observe whether each distance keeps it near the bottom."),
    HomeworkGuide("Musical analysis in `esl` is feature-level and listening-led. Chroma and Tonnetz describe harmonic-color evidence, MFCCs and spectral contrast describe timbre, and similarity/novelty matrices describe recurrence and change; none is a substitute for a symbolic transcription claim.", "Use one short, legal-to-share excerpt with a clearly audible repetition or transition. Keep the sample rate, frame/hop settings, `librosa` feature set, normalization, and novelty-kernel settings fixed while comparing the marked passages.", "Which observed matrix block or feature trajectory supports the claimed repetition, boundary, or timbral contrast, and which musical interpretation remains intentionally unclaimed?", "Repeat with `--feature-set core` and explain which interpretation survives when chroma, Tonnetz, MFCC deltas, spectral contrast, and onset strength are unavailable."),
    HomeworkGuide("Multichannel analysis must state whether a result belongs to an individual channel, an aggregate, or a downmix. These are different questions.", "Choose stereo material with an intentional left/right difference. Retain the per-channel output and record the aggregate policy.", "When does downmix ranking hide a channel-specific event, and when does it provide a useful overview?", "Swap the channel order in a copy and verify that labels, per-channel values, and aggregate semantics remain coherent."),
    HomeworkGuide("Spatial and Ambisonics metadata cannot be guessed safely from wishful thinking. A sidecar makes the convention, ordering, and confidence inspectable.", "Use only a documented B-format source or a provided fixture. Verify decoded channel count before and after applying the sidecar.", "Which spatial fields are facts supplied by the source, and which are assumptions introduced by the sidecar?", "Create an intentionally invalid sidecar and record the validation failure without forcing the analysis through."),
    HomeworkGuide("A shard manifest turns files into a time-ordered archive with explicit boundaries. The clock is data, not a filename decoration.", "Use three or more files whose sequence is known. Do not invent absolute calendar time; use archive-relative time when the deployment start is uncertain.", "What archive interval does each shard represent, and where could a missing or overlapping file break interpretation?", "Add a deliberate gap to a copy of the manifest and describe the correct reporting behavior."),
    HomeworkGuide("Civil time, UTC, and archive-relative time answer different questions. A chart can be precise but still wrong for a local-day interpretation if its time policy is unstated.", "Keep the manifest and metric configuration fixed while producing at least two rollups. Record the UTC offset or named time zone used.", "What changes near a day boundary when you move from archive-relative to civil time?", "Describe how daylight-saving transitions should be represented in a deployment report."),
    HomeworkGuide("Long-recording planning is an engineering calculation: duration, channels, sample rate, bit depth, compression, I/O, and recovery behavior all matter.", "State every assumed parameter and distinguish nominal uncompressed size from compressed estimates. Use GB consistently and include headroom for outputs.", "Which parameter dominates storage growth, and which operational bottleneck is most likely before storage is exhausted?", "Recalculate for a 30-day archive and propose a shard duration justified by recovery and transfer constraints."),
    HomeworkGuide("RF64 extends WAV when classic RIFF size fields are inadequate, but compatibility remains a deployment constraint. Format selection is a workflow decision.", "Specify duration, sample format, channels, sample rate, downstream tools, and whether random access or compression is required.", "Which requirement rules out the most attractive format option, and why?", "Design a lossless preservation-plus-delivery policy using RF64 or WAV alongside FLAC derivatives where appropriate."),
    HomeworkGuide("A FrameTable is a contract between signal processing and ML. Deterministic columns and timestamps make it possible to reproduce feature preparation.", "Export from a fixed source with explicit window and hop settings. Preserve the metadata sidecar with the tabular artifact.", "Which column identifies time, which columns identify features, and how would a reader recover channel semantics?", "Export the same source twice and compare schemas and values to identify any nondeterministic field."),
    HomeworkGuide("Tensor layout errors silently invalidate ML experiments. The declared axes must agree with the tabular representation and model input expectation.", "Use a short multichannel source so individual channels and frames are easy to inspect. Keep the FrameTable export for cross-checking.", "How does one tensor element map back to a channel, timestamp, and feature name?", "Transpose a copy deliberately and write a test that would catch the error before model training."),
    HomeworkGuide("Random row-level splits are often leakage machines for recordings. A defensible split follows the unit that matters to the intended generalization claim.", "Define the grouping unit before writing the manifest: site, recorder, date block, deployment, or subject. Document why adjacent data are related.", "What information could leak from training into evaluation under a naive sequential or random split?", "Propose a second split that tests a harder and more relevant generalization condition."),
    HomeworkGuide("Novelty and anomaly scores depend on a baseline. An unusual loud event is not necessarily an anomalous acoustic pattern, and vice versa.", "Choose a reference period with a stated rationale. Keep the feature representation, normalization, and threshold policy visible.", "Which high score is caused mostly by level, and which reflects a change in spectral or temporal structure?", "Simulate a benign but loud event and a quiet but structurally rare event to compare score behavior."),
    HomeworkGuide("NDSI is a band-energy contrast with chosen bands, not a direct biodiversity meter. Band definitions and local context determine what the sign can mean.", "State sample rate, frequency bands, weighting, and channel policy. Verify that both target bands are represented below Nyquist.", "What signal construction confirms the expected sign, and what real-world confound could reverse the interpretation?", "Repeat with an alternate documented band boundary and report sensitivity rather than selecting the prettier result."),
    HomeworkGuide("Ecoacoustic indices are summaries that require independent validation. A useful critique separates mathematical definition, ecological claim, and evidence quality.", "Select two peer-reviewed limitations papers from the reference list and one local observation method that does not reuse the same index.", "What would count as a failed validation for the chosen index, even if the computation itself is correct?", "Draft a preregistered hypothesis that uses the index as one measurement among several."),
    HomeworkGuide("Long-term trends can arise from ecology, weather, recorder gain, placement, or processing changes. Trend interpretation requires competing hypotheses.", "Use repeated units with known timing and preserve device/calibration metadata. Plot uncertainty, gaps, and maintenance events when known.", "Which feature of the trend favors sensor drift over environmental change, and what observation could discriminate them?", "Insert a known gain change into a simulated series and document which metrics react first."),
    HomeworkGuide("Simulation variants are comparable only when their modeling assumptions travel with the acoustic results. The report should distinguish prediction from measurement.", "For both variants, record geometry, material assumptions, source, receiver, rendering, sample rate, and analysis window.", "Which conclusion is supported by the comparison, and which would require a physical measurement or standards-compliant procedure?", "Add a third variant that changes only one design parameter and state the expected direction of change before analyzing it."),
    HomeworkGuide("Decoding is part of the analysis pipeline for compressed media. Lossless and compressed inputs may share content while carrying different codec and decoder provenance.", "Use two versions with matching source content when possible. Record codec, duration, sample rate, channel layout, and decoder backend.", "Which result differences are plausibly codec-related, and which should not be attributed to decoding without further evidence?", "Re-run the compressed file after a decoder update in a controlled environment and compare provenance hashes."),
    HomeworkGuide("A golden fixture turns an analytical expectation into an executable contract. Tolerance selection should reflect numerical reality, not a wish for exact floating-point equality.", "Choose a metric with a mathematically controlled input, such as a sine, offset, clipped waveform, or exponential decay. Record the expected result derivation.", "Which tolerance is justified by windowing, numerical precision, or fitting behavior, and what regression would it detect?", "Create a second adversarial fixture that should trigger a validity flag instead of a confident scalar result."),
    HomeworkGuide("A reproducible handoff lets another analyst inspect, rerun, and challenge a result. The deliverable is an analysis package, not a screenshot.", "Use a small legal-to-share input or an immutable input identifier. Include all configuration, environment/version information, and output artifacts.", "Could a reviewer identify every assumption and regenerate the principal result without asking you a question?", "Have another person follow the checklist, then amend it based on the first failure or ambiguity they encounter."),
    HomeworkGuide("A defensible acoustic claim is narrow, falsifiable, and explicit about uncertainty. The capstone rewards an auditable argument rather than a dramatic metric.", "Select a scope small enough to review manually: a brief archive period, a few design variants, or a controlled simulation set. Define the claim before inspecting the final result.", "What observation would weaken the claim, and what evidence is still missing after the analysis?", "Prepare a five-minute methods briefing that a skeptical collaborator could use to reproduce or challenge the conclusion."),
)


LAB_NOTEBOOK_LENSES: tuple[tuple[str, str], ...] = (
    ("Scope and prediction", "state a narrow prediction and the practical decision that would change if the result disagrees"),
    ("Input lineage", "trace the file, source interval, decoder, channel labels, calibration state, and permissions"),
    ("Parameter ledger", "record defaults and deliberate choices so the analysis can be rerun instead of remembered"),
    ("Evidence reading", "separate what a plot, table, score, waveform, or listening note literally shows from the explanation attached to it"),
    ("Skeptical review", "name artifacts and confounds that could produce the same pattern, then seek a targeted check"),
    ("Handoff and next action", "package the result for a second analyst and identify the smallest follow-up that would reduce the most important uncertainty"),
)


def _lab_notebook_markdown(number: int, title: str, task: str, guide: HomeworkGuide) -> str:
    """Return the extended research-notebook material for one laboratory.

    Each assignment gets the same complete evidence structure, but the title,
    task, and guide keep its prose attached to a distinct analytical exercise.
    """
    lines = [
        "### Extended Research Notebook",
        "",
        "This laboratory is intentionally longer than a command recipe. Use the following notebook "
        "prompts to create a record that another person can inspect without having to infer your "
        "intent. The goal is not to produce more prose for its own sake. It is to preserve the "
        "decision points that determine whether the assignment demonstrates a method, a property of "
        "the selected recording, or merely an accidental feature of one run.",
        "",
    ]
    for lens, action in LAB_NOTEBOOK_LENSES:
        lines.extend(
            (
                f"#### {lens}",
                "",
                f"For Assignment {number}, **{title}**, begin by choosing language that a skeptical "
                f"reader could test. Your immediate task is: {task} The notebook action is to {action}. "
                "Keep the claim proportional to the artifact. A value, rank, fit, or plot region is an "
                "observation under a particular configuration; it is not automatically an explanation "
                "of the source, environment, listener, or design. The assignment's stated purpose is "
                f"also a useful boundary: {guide.purpose}",
                "",
                "Document the decision before the result is visible whenever possible. Identify what is "
                "held constant, what changes, what output counts as evidence, and what result would make "
                "you revise the initial expectation. Save the raw result alongside the reviewed artifact, "
                "including warnings, empty fields, and validity flags. During review, distinguish a "
                "reproducible observation from a plausible interpretation. If the result is ambiguous, "
                "write the ambiguity plainly and propose a smaller, controlled comparison rather than "
                "quietly changing the workflow until the graph tells a more satisfying story.",
                "",
            )
        )
    lines.extend(
        (
            "#### Closing record",
            "",
            "Close this laboratory with a short methods memo: name the input, command/configuration, "
            "software version, units, channel policy, calibration state, principal artifact, observed "
            "result, alternative explanation, and next action. Link or list the exact files needed for "
            "a repeat run. The final sentence should be a bounded claim, not a verdict. A clear limit "
            "is evidence of good practice, especially when an audio file is being asked to answer a "
            "question that belongs partly to a microphone, a place, a listener, or a study design.",
            "",
        )
    )
    return "\n".join(lines).rstrip()


def _homework_markdown() -> str:
    """Build the end-of-book 33-assignment laboratory bank."""
    if len(HOMEWORK) != len(HOMEWORK_GUIDES):
        raise RuntimeError("Every homework prompt must have expanded laboratory guidance.")
    lines = ["# Homework and Laboratory Assignments", "", "These 33 assignments use real `esl` workflows. They are deliberately framed as reproducible evidence exercises, not scavenger hunts for the largest number a command can print. Keep every command, configuration file, software version, and output artifact.", "", "Each assignment includes a purpose, a controlled comparison, a protocol, evidence requirements, interpretation questions, and an extension. The repeated structure is intentional: the habit of recording conditions is more valuable than memorizing a single command. In the PDF edition, every assignment begins on a fresh page so its evidence record remains a distinct unit of work.", "", "The assignments become progressively more demanding, but none requires heroic hardware or a giant proprietary dataset. Start with a short, legal-to-share file or a synthetic fixture. Small controlled examples reveal more about a metric than a large archive whose assumptions are unknown. When a task asks for listening, the listening note is evidence; when it asks for a plot, the plot needs a timestamped explanation rather than a decorative caption.", "", "## Submission Standard", "", "Every submission should identify the input, the command/configuration, relevant units, calibration state, channel convention, output location, and one uncertainty. If you cannot state an assumption, you have probably found the next thing to learn.", "", "For each lab, preserve a `README.md` or notebook cell that records the input identifier, the exact command or configuration, the software version, output paths, and the date. Do not include sensitive recordings in a submission when a derived artifact or private identifier is sufficient.", "", "A strong laboratory record separates three things: what the software measured, what the analyst observed in plots or listening, and what interpretation remains plausible but unproven. This separation is not academic bureaucracy. It prevents a configuration choice, a decoding artifact, or a loud transient from silently becoming a scientific conclusion.", ""]
    for number, ((title, task, deliverable), guide) in enumerate(zip(HOMEWORK, HOMEWORK_GUIDES, strict=True), start=1):
        lines.extend(("<div class=\"assignment-start\" aria-hidden=\"true\"></div>", "", f"## Assignment {number}: {title}", "", "### Why This Matters", "", guide.purpose, "", "The central question in this assignment is deliberately narrower than its title. A useful result identifies what changed, what was held fixed, and what the selected metric or workflow can actually support. If the first result is surprising, treat surprise as a reason to inspect the data and assumptions, not as proof that the result is important.", "", "### Before You Begin", "", "Create a small working copy of the input or record an immutable identifier for it. Confirm duration, sample rate, channel count, decoder, and calibration status before changing any analysis parameter. Decide in advance which output will count as evidence: a scalar with provenance, a frame table, a plot, a timestamped clip, or a comparison table.", "", "Write one sentence predicting what you expect to see and why. The prediction may be wrong; that is useful. It makes the later comparison between expectation and result visible instead of allowing an after-the-fact story to masquerade as a plan.", "", "### Controlled Comparison", "", guide.control, "", "The comparison is the assignment's guardrail. Change the stated factor and keep the input, preprocessing, output scale, and review procedure stable. If a practical constraint forces an additional change, record it plainly and reduce the strength of the conclusion accordingly.", "", "### Method Narrative", "", f"The task is: {task}", "", f"Approach this as a short methods study, not a button-pressing exercise. The goal is to connect the resulting artifact to the mechanism described above: {guide.purpose} Keep the configuration and evidence together so a reader can distinguish a change in the signal from a change introduced by the analysis choices.", "", "### Protocol", "", "1. Create a dedicated working directory named for this assignment. Preserve the input identifier, a copy of the command/configuration, and the unedited output JSON before making interpretive notes.", f"2. {task}", "3. Repeat the workflow with the controlled comparison stated above. Change one analytical decision at a time; if you must change more than one, record why.", "4. Inspect the result numerically and by listening or plotting where appropriate. Record validity flags, confidence fields, units, channel semantics, and provenance rather than copying only the headline metric.", "5. Write a short conclusion that distinguishes an observed result from an interpretation. Include one alternative explanation that the present data cannot rule out.", "6. Archive the exact command, configuration, and evidence artifacts in one place. A reviewer should be able to find the input identifier, recreate the analysis, and understand why you made the final comparison without reading your mind.", "", _lab_notebook_markdown(number, title, task, guide), "", "### Evidence to Submit", "", "**Required artifact.** " + deliverable, "", "Also include: the exact command/configuration, a short provenance excerpt, the relevant plot or CSV row, and a sentence identifying the channel and calibration policy. Label figures and tables with the input identifier and time interval. If an output is uncertain or invalid, include it and explain the flag rather than quietly replacing it with a better-looking result.", "", "### Evidence Quality Check", "", "Before submitting, ask whether the artifact answers the stated question at the correct time scale. Check that axes, units, time zones, channel labels, and normalization choices are visible. Then ask whether a second analyst could reproduce the same artifact using only the record you provide. If not, add the missing configuration or provenance now; future-you is also a second analyst.", "", "### Interpretation Questions", "", f"- {guide.question}", "- Which result is robust to the controlled comparison, and which result depends on a chosen parameter or representation?", "- What independent observation, listening review, calibration check, or ground-truth label would make the conclusion more credible?", "", "### Interpretation Walkthrough", "", "Start with a literal description of the artifact: identify the timestamp, value, rank, trend, or plot region without attaching a cause. Next connect that observation to the metric definition and the controls you held fixed. Only then state a constrained interpretation and the alternative explanation that remains. This ordering keeps the conclusion proportional to the evidence.", "", "A high score, a clean plot, or a stable rank is not automatically a validated result. Consider decoding differences, calibration state, channel aggregation, windowing, feature normalization, and the possibility that a familiar artifact can mimic the pattern you expected. The correct response to ambiguity is usually a targeted second comparison, not a longer adjective.", "", "### Optional Extension", "", guide.extension, "", "### Reading and Method Note", "", "Read the relevant chapter and its cited references before generalizing. Use [Metrics Reference](METRICS_REFERENCE.md), [Novelty and Anomaly](NOVELTY_ANOMALY.md), [Similarity Search](SIMILARITY_SEARCH.md), [Shard Workflows](SHARD_WORKFLOWS.md), and [References](REFERENCES.md) as appropriate; no metric output is a substitute for the method record.", "", "Carry one sentence from this assignment into the next: name the measurement, its assumptions, and the strongest alternative explanation. That small habit turns a sequence of exercises into a defensible analytical practice.", ""))
    return "\n".join(lines).rstrip()


def _introduction_markdown() -> str:
    """Return the front-of-book introduction that establishes the method."""
    sections = (
        "# Introduction",
        "## Sound Is Easy to Hear and Difficult to Measure Well",
        "Most audio work begins with an ordinary sentence: *something happened in this recording; can we find it, describe it, compare it, or explain why it matters?* The sentence may concern a frog chorus, a factory floor, a classroom, a concert, a room simulation, a monitoring deployment, or a decade of archive files. The question sounds simple because hearing is fast. The evidence problem is not. An audio file is a sequence of numbers produced by a chain of physical and technical events, and every link in that chain can affect the meaning of a later result.",
        "A source excites air or a structure. A microphone or sensor responds with some sensitivity, directionality, noise floor, and saturation behavior. A recorder applies gain, timing, conversion, encoding, and storage. A decoder turns stored bytes into samples. An analyst then chooses channels, windows, features, normalizations, thresholds, plots, and words. Each stage can be appropriate. Each stage can also change what a later result means. This book treats that chain as part of the subject rather than as an inconvenience before the interesting part.",
        "The aim is deliberately practical: make a result that a second person can inspect. That means more than a number or a colorful spectrogram. It means an artifact connected to its source, settings, units, uncertainty, and intended use. A good analysis should help you decide what to listen to next, what to compare, which assumption needs checking, or what can be said honestly about the recording in front of you. It should not make a dramatic conclusion feel inevitable merely because software printed it with many decimal places.",
        "## Begin with Listening, Then State the Question",
        "Listening is not an embarrassing preliminary step that sophisticated analysis eventually eliminates. It is how a human notices whether a feature aligns with the event, source, boundary, defect, or quality that prompted the analysis. A novelty peak might be a useful call, a door slam, wind across a microphone, a gain switch, or a file boundary. A high level might be a source change, a distance change, a calibration mistake, or clipping. A clean diagonal block in a similarity matrix might represent musical form, repetition, a steady machine, or a repeated analysis artifact. The computation narrows review; it does not replace the need to inspect evidence.",
        "The first discipline is to change a vague request into a testable question. Instead of asking whether a recording is interesting, ask whether there are time intervals with unusually large spectral change under a stated feature, normalization, and kernel scale. Instead of asking whether a room sounds better, ask how two simulated impulse responses differ in decay fit, clarity, or early-to-late energy under declared source and receiver conditions. Instead of asking whether an archive is calmer, define the level, event rate, band, or proxy being summarized and the time interval to which it applies.",
        "The smaller question is not a weaker question. It creates a place where the answer can be checked. It also creates a place where the answer stops. A well-bounded result may support a claim about selected recordings, a selected representation, and a selected comparison. It normally does not, without more evidence, support a claim about every listener, every season, every room, every species, or every source of a similar-looking pattern.",
        "```mermaid\nflowchart LR\n    A[\"Listen and observe\"] --> B[\"State a bounded question\"]\n    B --> C[\"Declare signal context\"]\n    C --> D[\"Compute a documented artifact\"]\n    D --> E[\"Review plot, clip, and flags\"]\n    E --> F[\"Write a constrained claim\"]\n    F -. \"remaining uncertainty\" .-> B\n```",
        "Plain English: analysis is a loop. A result should improve the next question, not close the investigation by force of formatting.",
        "## The Measurement Chain Is Part of the Result",
        "Before using equations, it helps to name what they are for. The following model does not describe every acoustic process in detail. It is an accounting model: it reminds us that the object we compute is a function of recorded samples and of the conditions under which those samples became evidence.",
        "$$\n\\mathcal{R} = \\mathcal{A}(x; \\theta, \\kappa, \\pi)\n$$",
        "where `R` is a reported result, `x` is the decoded sample sequence, `A` is the analysis procedure, `theta` is the resolved configuration such as window and hop, `kappa` is physical and recording context such as calibration and channel metadata, and `pi` is provenance such as decoder and library versions. Plain English: the same samples can yield different useful results when the analytical question changes, but those changes must be visible.",
        "This is why the distinction between dBFS and SPL matters. dBFS describes a digital relationship to full scale in the decoded signal. It can be useful for quality control, comparison within a stable digital system, and many feature workflows. SPL and weighted levels require a defensible physical reference. A calibration statement is not a decoration attached after analysis; it is the bridge between a digital sample value and a physical-level interpretation. When the bridge is missing, `esl` should say so plainly rather than pretending the file contains a sound-level meter certificate.",
        "Likewise, a file with four or eight channels is not automatically a spatial recording in a known convention. Multichannel audio can mean independent microphones, a multitrack recorder, a binaural render, a surround bed, or an Ambisonic signal with a named channel order and normalization. Preserving the channels is the safe default. Aggregating them is a stated analytical choice. The chapters ahead repeatedly ask a simple question: does this value belong to a channel, a downmix, a spatial transform, a frame, a clip, a day, or an archive? The answer determines what comparison is valid.",
        "## The Same Toolkit Serves Different Acoustic Questions",
        "Environmental acoustics, architectural acoustics, industrial measurement, and machine-learning preparation share a surprising amount of infrastructure. All need reliable decoding, explicit time bases, stable feature names, measured or declared units, plots, and records that survive a handoff. They do not share all interpretations. A feature becomes ecological through a study design and a validated relationship to an environmental question, not because it was calculated outdoors. A room metric becomes architectural evidence through source, receiver, bandwidth, fit, and simulation or measurement assumptions, not because the filename contains the word `room`.",
        "This textbook moves across domains while resisting false equivalence. An archive researcher may use novelty and similarity to select clips for human annotation. A consultant may compare impulse-response variants with RT60, EDT, C50, C80, and D50. A production engineer may audit clipping, DC offset, calibration state, and spectral change. An ML practitioner may export a FrameTable or tensor and build a dataset manifest that prevents adjacent audio from leaking across train and evaluation splits. The computations may overlap; the evidence needed to support a decision does not.",
        "The common practice is provenance. Keep the input identifier, command or configuration, decoder information, channel interpretation, calibration state, metric list, output artifact, and review note together. This package is more valuable than a screenshot because it gives a collaborator a way to reproduce, question, improve, or reject the analysis without guessing what happened in a terminal several months ago. See [Schema](SCHEMA.md), [Metrics Reference](METRICS_REFERENCE.md), and [Validation](VALIDATION.md) for the contracts behind this practice.",
        "## Scale Changes the Workflow, Not the Duty of Care",
        "One short WAV can be inspected by eye and ear. A 24-hour multichannel RF64 file cannot be treated as an array you casually load into memory. A decade-long archive cannot be treated as a single spectrogram or a single similarity matrix. At larger scales, the workflow must become chunked, sharded, resumable, and selective. The result is not less rigorous. It is rigorous in a different way: it records shard boundaries, archive clocks, summary intervals, rank policies, and the cost of the calculation.",
        "This changes what finding the interesting parts means. You do not ask a computer to know what is interesting in the human sense. You specify a reproducible candidate policy: top novelty events under a given representation, level changes over a threshold, intervals similar to a query, or deviations from a documented baseline. `esl moments extract` can export a bounded review set with a CSV of timestamps and WAV clips that include requested pre-event and post-event context. The final judgment remains reviewable because the selection rule and the original archive context travel with the clips.",
        "Large scale also makes storage and time visible. Sample rate, bit depth, channel count, duration, compression, feature density, and plot choice all have costs. Use GB consistently when planning storage. Estimate before a long job begins. Choose summary intervals that match the decision. Retain enough raw data and provenance for a challenge or a rerun. The chapters on [RF64](RF64_AND_LARGE_FILES.md), [shards](SHARD_WORKFLOWS.md), manifests, and archive insights turn these principles into operational workflows.",
        "## Models, Metrics, and Plots Are Not Oracles",
        "It is tempting to treat a named metric as a complete interpretation. A metric is more modest: a defined transformation from a signal and its settings to a value, series, matrix, or rank. Some metrics are standards-aligned under specified conditions. Some are research features. Some are explicitly proxies. All need a reader to ask what was measured, at what time scale, in what units, over which channels, and against which reference.",
        "The same caution applies to machine learning. A tensor layout is not a model. A model score is not a ground-truth label. Similarity in a feature space is not identity of source or cause. A dataset split is not valid merely because it is random; adjacent shards from the same deployment can make a model appear better than it will be on a new place or time. The ML chapters provide stable FrameTable, tensor, and manifest contracts precisely so that these assumptions can be inspected instead of buried in notebook state.",
        "Visualization is part of the safeguard. A plot can reveal a bad channel, a windowing mistake, a quantization boundary, an implausible fit, or a rank driven by a narrow artifact. Use a plot to test whether a number resembles its claimed mechanism. Use listening to test whether an extracted event contains the phenomenon you care about. Use an independent observation, annotation, calibration check, or controlled fixture when the conclusion matters beyond a first-pass screen.",
        "## How to Read the Rest of This Book",
        "Part I establishes the working vocabulary: files, channels, levels, windows, metrics, and reproducible first runs. Do not rush past it. Those choices shape every later plot and export. Part II develops event candidates, novelty, similarity, and reviewable clips. It is the practical route from a long file to a small set of moments someone can inspect. Part III addresses the realities of scale and spatial context: RF64, shard manifests, archive reports, and metadata that should not be guessed. Part IV connects features to datasets, provenance to schema, calibration to physical interpretation, and validation to a release or research handoff.",
        "The laboratory assignments turn every topic into a controlled evidence exercise. Work through them with a small legal-to-share recording or a synthetic fixture before asking a large, irreplaceable archive to carry the burden of your first experiment. Read the high-level explanation. Study the equation or diagram. Note the `where` statement and units. Run the smallest command that creates a real artifact. Inspect the artifact. Record one assumption and one alternative explanation. Then proceed.",
        "The first part begins with a deliberately modest promise: take one file, learn what it is, make one result whose method you can explain, and keep enough record to do it again. That is how a durable acoustic practice starts.",
    )
    return "\n\n".join(sections)


def build_textbook_markdown() -> str:
    """Return the complete textbook source from curated maintained chapters."""
    contents_rows = ["- [Introduction](#introduction)"]
    chapter_number = 1
    for part_index, (part_title, chapters) in enumerate(PARTS, start=1):
        part_label = part_title.replace(" - ", ": ", 1)
        contents_rows.append(f"- [{part_label}](#part-{part_index})")
        for chapter in chapters:
            # Four spaces produce a nested Markdown list in Python-Markdown.
            contents_rows.append(f"    - [Chapter {chapter_number}: {chapter.title}](#chapter-{chapter_number}-{_slugify(chapter.title)})")
            chapter_number += 1
    contents_rows.extend(("- [Homework and Laboratory Assignments](#homework-and-laboratory-assignments)", "- [Appendix: A First Laboratory Session](#appendix-a-first-laboratory-session)", "- [Appendix: Attribution and Source Map](#appendix-attribution-and-source-map)", "- [Colophon](#colophon)"))

    chapter_fragments: list[str] = []
    figures: list[Figure] = []
    chapter_number = 1
    for part_index, (part_title, chapters) in enumerate(PARTS, start=1):
        chapter_fragments.extend(("---", "", f"<a id=\"part-{part_index}\"></a>", "", f"# {part_title}", ""))
        for chapter in chapters:
            chapter_markdown, chapter_figures = _chapter_markdown(chapter_number, chapter)
            chapter_fragments.extend((chapter_markdown, "", "---", ""))
            figures.extend(chapter_figures)
            chapter_number += 1

    figure_rows = ["| Figure | Description |", "| --- | --- |"]
    if figures:
        for figure in figures:
            figure_rows.append(f"| [Figure {figure.label}](#figure-{figure.label}) | {figure.caption} |")
    else:
        figure_rows.append("| None | This edition has no raster figures. |")
    lines = [
        '<section class="textbook-title-page">',
        '<img class="textbook-title-logo" src="assets/logos/esl/minimal/esl_logo_04_circle_wave.png" alt="ecoSignalLab logo" />',
        "<h1>ecoSignalLab</h1>",
        "<h2>Environmental, Architectural, and Long-Archive Acoustic Analysis</h2>",
        "<h3>A Practical Textbook and Reproducible Laboratory Companion for <code>esl</code></h3>",
        "<p><strong>Edition 0.2.0</strong></p>",
        "<p>Colby Leider and ecoSignalLab contributors</p>",
        "<p>Licensed under the <a href=\"../LICENSE\">MIT License</a>.</p>",
        "<p>This textbook is generated from version-controlled documentation. Its claims, algorithms, source links, and laboratory workflows are intended to be inspectable rather than ornamental.</p>",
        "</section>",
        "",
        "---",
        "",
        "# Preface",
        "",
        "This is a working textbook for people who need to make defensible statements about sound. It begins with one file and a few commands, then expands into calibration, multichannel semantics, event review, long archives, spatial metadata, architectural impulse responses, ML exports, and validation. It is not a promise that a metric can answer every question. It is a guide to recording what the metric actually did.",
        "",
        "The shortest route through the book is Part I, one event-analysis chapter from Part II, and the laboratory appendix. The rigorous route is all of it, including the parts where the data tell you that your first interpretation was too confident. Those parts are not bugs.",
        "",
        "## Intended Readers",
        "",
        "- New users with an audio file and a practical question.",
        "- Researchers who need reproducible acoustic features and exports.",
        "- Engineers working with multichannel, calibrated, or room-simulation material.",
        "- Instructors looking for short laboratory assignments with concrete artifacts.",
        "",
        "## Conventions Used in This Book",
        "",
        "- Commands are shown in code blocks and are intended to be copied, then adapted to real file paths.",
        "- A displayed equation is followed by a `where` statement and a plain-English interpretation when the source chapter supplies one.",
        "- Figure labels use **chapter-figure** notation. For example, Figure 4-2 is the second figure in Chapter 4.",
        "- A calibrated result is not interchangeable with dBFS. Read the units and provenance before comparing values.",
        "",
        '<div class="textbook-page-start" aria-hidden="true"></div>',
        "",
        "## Table of Contents",
        "",
        *contents_rows,
        "",
        '<div class="textbook-page-start" aria-hidden="true"></div>',
        "",
        "## Table of Figures",
        "",
        *figure_rows,
        "",
        '<div class="textbook-page-start" aria-hidden="true"></div>',
        "",
        "## How to Use This Book",
        "",
        "Quick links: [README](../README.md) | [User Guide](USERGUIDE.md) | [Docs Index](INDEX.md) | [References](REFERENCES.md)",
        "",
        "This textbook starts with a friendly working manual, then moves into the",
        "maintained technical chapters that define methods, assumptions, data",
        "contracts, and references. Read Part I before running a new workflow.",
        "Use Parts II and III when investigating events or large archives. Use Part",
        "IV before training models, publishing measurements, or handing a project",
        "to another analyst.",
        "",
        "Every chapter is generated from source-controlled documentation. That keeps",
        "the textbook aligned with the codebase rather than becoming a handsome but",
        "historically inaccurate fossil.",
        "",
        "```mermaid",
        "flowchart LR",
        "    A[\"Part I: foundations\"] --> B[\"Part II: events and similarity\"]",
        "    B --> C[\"Part III: long archives and spatial audio\"]",
        "    C --> D[\"Part IV: ML, schema, validation\"]",
        "    D --> E[\"Reproducible acoustic practice\"]",
        "```",
        "",
        "## A Note on Claims",
        "",
        "`esl` can compute a broad set of metrics. The meaning of a metric depends on",
        "the signal, calibration, windowing, channel convention, and study design.",
        "A number becomes evidence only after those conditions are stated. The book",
        "therefore preserves warnings, confidence fields, formulas, and citations",
        "instead of treating a command-line result as a verdict.",
        "",
        "---",
        "",
        _introduction_markdown(),
        "",
    ]
    lines.extend(chapter_fragments)
    lines.extend(
        (
            "---",
            "",
            _homework_markdown(),
            "",
            "---",
            "",
            "# Appendix: A First Laboratory Session",
            "",
            "Work through this sequence with a small, legal-to-share WAV or FLAC file:",
            "",
            "```bash",
            "esl doctor input.wav",
            "esl analyze input.wav --out-dir out --plot --ml-export",
            "esl moments extract input.wav --out out/moments --single --event-window 10",
            "esl schema",
            "```",
            "",
            "Write down: the input duration, channels, sample rate, decoder, selected",
            "metric IDs, one plot observation, one candidate moment, and whether the",
            "result is calibrated. That tiny record is the beginning of a reproducible",
            "analysis notebook.",
            "",
            "# Appendix: Attribution and Source Map",
            "",
            "The textbook contains documentation written for ecoSignalLab and material",
            "that cites the underlying scientific and open-source work. Consult",
            "[Attribution](ATTRIBUTION.md) and [References](REFERENCES.md) when adapting",
            "algorithms, comparing implementations, or preparing a publication.",
            "",
            "# Colophon",
            "",
            "This is the `ecoSignalLab` textbook and reproducible laboratory companion, edition 0.2.0. It is maintained by Colby Leider and ecoSignalLab contributors and generated from version-controlled source documentation in the same repository as the software. The book is designed to be inspected alongside the code, configuration contracts, tests, examples, and cited methods rather than treated as a detached manual.",
            "",
            "The book face is [TeX Gyre Schola](https://www.gust.org.pl/projects/e-foundry/tex-gyre), an open typeface in the New Century Schoolbook tradition. Code and literal command names use a monospace face. The cover mark is the `esl` circle-wave logo from [`assets/logos/esl/minimal/`](../assets/logos/esl/minimal/), reproduced above the title. The logo remains a project asset; its appearance is not a claim of certification, calibration, or standards compliance.",
            "",
            "Markdown source is rendered by the `esl` documentation builder. Mathematics is interpreted with [MathJax](https://www.mathjax.org/) and frozen as SVG in generated HTML/PDF output so equations remain visible without a live network connection. Diagrams are rendered with [Mermaid](https://mermaid.js.org/), and PDFs are produced through [Playwright](https://playwright.dev/python/) with Chromium. The production path is deliberately documented because typography, mathematics, diagrams, and pagination are part of the artifact a reader receives.",
            "",
            "Algorithms, third-party APIs, and implementation context are documented in [Attribution](ATTRIBUTION.md) and [References](REFERENCES.md). The software and this documentation are distributed under the [MIT License](../LICENSE); the bundled TeX Gyre Schola font is accompanied by its [GUST Font License](../src/esl/docsgen/assets/fonts/GUST-FONT-LICENSE.txt). For a reproducible rebuild, run `python scripts/generate_textbook.py` in an environment with the documented dependencies installed.",
            "",
        )
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the ecoSignalLab textbook as Markdown, HTML, and PDF")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Generated textbook Markdown path")
    parser.add_argument("--out", default=str(DEFAULT_PDF), help="Final PDF output path")
    parser.add_argument(
        "--build-dir",
        default=str(ROOT / "docs" / "build" / "textbook"),
        help="Directory for intermediate HTML/PDF artifacts",
    )
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(build_textbook_markdown(), encoding="utf-8")

    output_pdf = Path(args.out).resolve()
    build_dir = Path(args.build_dir)
    # Textbook chapters retain relative `examples/...` links. The documentation
    # renderer emits this page as html/docs/TEXTBOOK.html, so stage the existing
    # image examples exactly where that page resolves them before PDF rendering.
    staged_examples = build_dir / "html" / "docs" / "examples"
    shutil.copytree(ROOT / "docs" / "examples", staged_examples, dirs_exist_ok=True)
    staged_logo = build_dir / "html" / "docs" / "assets" / "logos" / "esl" / "minimal"
    staged_logo.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEXTBOOK_LOGO, staged_logo / TEXTBOOK_LOGO.name)
    report = build_docs(
        root=ROOT,
        output_root=build_dir,
        formats={"html", "pdf"},
        title="ecoSignalLab Textbook",
        docs_files=[source_path],
    )
    textbook_pdf = next((path for path in report.pdf_pages if path.name == "TEXTBOOK.pdf"), None)
    if textbook_pdf is None:
        raise RuntimeError("Documentation renderer did not produce the textbook PDF.")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(textbook_pdf, output_pdf)
    print(f"markdown: {source_path}")
    print(f"html: {report.output_root / 'html' / 'docs' / 'TEXTBOOK.html'}")
    print(f"pdf: {output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
