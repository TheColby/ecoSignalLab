"""Visualization APIs."""

from .plotting import compute_novelty_matrix, plot_analysis, plot_from_json, plot_novelty_matrix
from .spawn import spawn_plot_paths
from .feature_vectors import extract_feature_vectors, load_feature_vectors, save_feature_vectors, similarity_matrix_from_features

__all__ = [
    "plot_analysis",
    "plot_from_json",
    "compute_novelty_matrix",
    "plot_novelty_matrix",
    "spawn_plot_paths",
    "extract_feature_vectors",
    "save_feature_vectors",
    "load_feature_vectors",
    "similarity_matrix_from_features",
]
