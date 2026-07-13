"""Regression tests for the generated textbook's essential front/back matter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def _load_generator() -> ModuleType:
    script = ROOT / "scripts" / "generate_textbook.py"
    spec = importlib.util.spec_from_file_location("generate_textbook", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_textbook_front_matter_figures_and_assignments_are_generated() -> None:
    generator = _load_generator()
    markdown = generator.build_textbook_markdown()

    assert '<section class="textbook-title-page">' in markdown
    assert 'class="textbook-title-logo"' in markdown
    assert 'assets/logos/esl/minimal/esl_logo_04_circle_wave.png' in markdown
    assert markdown.count("## How to Use This Book") == 1
    assert markdown.count("# Introduction") == 1
    assert markdown.index("- [Introduction](#introduction)") < markdown.index("- [Part I:")
    assert "## Sound Is Easy to Hear and Difficult to Measure Well" in markdown
    assert "## The Measurement Chain Is Part of the Result" in markdown
    assert "## How to Read the Rest of This Book" in markdown
    assert "#### A transparent, inspectable implementation" in markdown
    assert "#### A command-line tool with optional visual evidence" in markdown
    assert "#### A composable SDK, not a locked analysis service" in markdown
    assert "#### Not a standards-certification engine" in markdown
    assert "## Table of Contents" in markdown
    assert "    - [Chapter 1: Orientation and First Workflows]" in markdown
    assert "    - [Chapter 13: References, Attribution, and Further Study]" in markdown
    assert "## Table of Figures" in markdown
    assert markdown.count('class="textbook-page-start"') == 3
    assert "- [Colophon](#colophon)" in markdown
    assert "[Figure 2-1](#figure-2-1)" in markdown
    assert '<figcaption><strong>Figure 2-1.</strong>' in markdown
    assert "# Homework and Laboratory Assignments" in markdown
    assert "## Assignment 15: Musical analysis: form, timbre, and harmonic color" in markdown
    assert "Do not claim chord, key, beat, or transcription labels." in markdown
    assert "## Assignment 1: Inspect before measuring" in markdown
    assert "## Assignment 33: Capstone: defensible claim" in markdown
    assert "# Colophon" in markdown
    assert "The book face is [TeX Gyre Schola]" in markdown
    assert len(generator.HOMEWORK) == 33
    assert len(generator.HOMEWORK_GUIDES) == 33
    assert len(generator.CHAPTER_CASES) == 13
    assert len(generator.STUDY_LENSES) == 40
    assert len(generator.LAB_NOTEBOOK_LENSES) == 6
    assert markdown.count("## Extended Study Dossier") == 13
    assert markdown.count("### Extended Research Notebook") == 33
    assert markdown.count("### The question before the command") == 13
    assert markdown.count("#### Scope and prediction") == 33
    assert markdown.count("### Why This Matters") == 33
    assert markdown.count("### Before You Begin") == 33
    assert markdown.count("### Controlled Comparison") == 33
    assert markdown.count("### Method Narrative") == 33
    assert markdown.count("### Evidence to Submit") == 33
    assert markdown.count("### Evidence Quality Check") == 33
    assert markdown.count("### Interpretation Questions") == 33
    assert markdown.count("### Interpretation Walkthrough") == 33
    assert markdown.count("### Optional Extension") == 33
    assert markdown.count('class="assignment-divider"') == 0
    assert markdown.count('class="assignment-start"') == 33
    assert markdown.count("## Why This Chapter Matters") == 13
    assert markdown.count("## Reading Strategy") == 13
