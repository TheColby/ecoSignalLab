"""Metric registry with plugin extensibility."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Iterable

from esl.metrics.base import MetricPlugin, MetricResult, MetricSpec
from esl.metrics.builtin import builtin_plugins
from esl.metrics.extended import extended_plugins

if TYPE_CHECKING:
    from esl.core.context import AnalysisContext

METRIC_CATALOG_VERSION = "esl-metrics-1.1.0"

DEPRECATED_METRIC_ALIASES: dict[str, tuple[str, str]] = {
    "interchannel_coherence": (
        "interchannel_correlation_zerolag",
        "'interchannel_coherence' was a zero-lag normalized cross-correlation in [-1, 1], "
        "not magnitude-squared coherence. It has been renamed to "
        "'interchannel_correlation_zerolag'; use 'interchannel_coherence_msc' for "
        "magnitude-squared coherence in [0, 1].",
    ),
}


class MetricRegistry:
    """Runtime registry for builtin + external metric plugins."""

    def __init__(self) -> None:
        self._plugins: dict[str, MetricPlugin] = {}
        self._aliases: dict[str, str] = {}
        self.alias_warnings: list[str] = []

    def register(self, plugin: MetricPlugin) -> None:
        name = plugin.spec.name
        if name in self._plugins:
            raise ValueError(f"Metric already registered: {name}")
        self._plugins[name] = plugin

    def register_alias(self, alias: str, target: str) -> None:
        if alias in self._plugins:
            raise ValueError(f"Alias collides with a registered metric: {alias}")
        self._aliases[alias] = target

    def resolve(self, name: str) -> str:
        """Map a possibly deprecated metric name onto its current name."""
        if name in self._plugins:
            return name
        target = self._aliases.get(name)
        if target is None:
            return name
        note = DEPRECATED_METRIC_ALIASES.get(name)
        message = note[1] if note else f"Metric '{name}' is deprecated; use '{target}'."
        if message not in self.alias_warnings:
            self.alias_warnings.append(message)
        return target

    def aliases(self) -> dict[str, str]:
        return dict(self._aliases)

    def get(self, name: str) -> MetricPlugin:
        resolved = self.resolve(name)
        if resolved not in self._plugins:
            raise KeyError(f"Metric not found: {name}")
        return self._plugins[resolved]

    def names(self) -> list[str]:
        return sorted(self._plugins.keys())

    def specs(self) -> list[MetricSpec]:
        return [self._plugins[name].spec for name in self.names()]

    def compute(
        self,
        ctx: AnalysisContext,
        metric_names: Iterable[str] | None = None,
    ) -> dict[str, MetricResult]:
        names = list(metric_names) if metric_names else self.names()
        out: dict[str, MetricResult] = {}
        for name in names:
            plugin = self.get(name)
            out[name] = plugin.compute(ctx)
        return out


def load_external_plugins(registry: MetricRegistry) -> None:
    """Load plugins from entry-point group `esl.plugins`."""
    eps = entry_points()
    group = eps.select(group="esl.plugins") if hasattr(eps, "select") else eps.get("esl.plugins", [])
    for ep in group:
        plugin_factory = ep.load()
        plugin = plugin_factory() if callable(plugin_factory) else plugin_factory
        registry.register(plugin)


def create_registry(with_external: bool = True) -> MetricRegistry:
    """Create registry preloaded with built-in metrics."""
    reg = MetricRegistry()
    for plugin in builtin_plugins():
        reg.register(plugin)
    for plugin in extended_plugins():
        reg.register(plugin)
    for alias, (target, _reason) in DEPRECATED_METRIC_ALIASES.items():
        reg.register_alias(alias, target)
    if with_external:
        try:
            load_external_plugins(reg)
        except Exception:
            # External plugins are optional; failure should not block core workflow.
            pass
    return reg
