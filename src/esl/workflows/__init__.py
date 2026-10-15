"""Command registration for review, monitoring, and archive workflows."""

from __future__ import annotations

import argparse
from importlib import import_module


def register_workflows(subparsers: argparse._SubParsersAction) -> None:
    """Register additive workflows without changing existing command contracts."""
    for module_name in (
        "review",
        "monitor",
        "uncertainty",
        "spatial_timeline",
        "qa",
        "feature_store",
        "embeddings",
        "protocol",
        "privacy",
        "archive_jobs",
    ):
        module = import_module(f"esl.workflows.{module_name}")
        module.register(subparsers)
