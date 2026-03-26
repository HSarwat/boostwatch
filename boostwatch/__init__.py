"""Boostwatch: Training-time observability for gradient boosting models."""

from .core.observer import GBDTObserver
from .analysis.feature_stats import compute_feature_stats
from .viz.plotting import (
    plot_splits_per_iteration,
    plot_feature_usage_over_time,
    plot_feature_stats,
    plot_confidence_distribution,
    plot_confidence_vs_errors
)

__version__ = "0.1.0"
__author__ = "Boostwatch Contributors"

__all__ = [
    "GBDTObserver",
    "compute_feature_stats",
    "plot_splits_per_iteration",
    "plot_feature_usage_over_time",
    "plot_feature_stats",
    "plot_confidence_distribution",
    "plot_confidence_vs_errors"
]