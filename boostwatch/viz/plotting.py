"""Backward-compatible plot functions.

These are thin wrappers around :mod:`boostwatch.viz.charts` that display the
figure immediately (equivalent to the pre-refactor behaviour).

For programmatic use — saving figures, building reports — import from
:mod:`boostwatch.viz.charts` directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from . import charts as _charts


def plot_splits_per_iteration(logs: List[Any]) -> None:
    _charts.plot_splits_per_iteration(logs)
    plt.show()


def plot_feature_usage_over_time(
    logs: List[Any], feature_names: Optional[List[str]] = None
) -> None:
    _charts.plot_feature_usage_over_time(logs, feature_names)
    plt.show()


def plot_feature_stats(
    stats: Dict[int, Dict], feature_names: Optional[List[str]] = None
) -> None:
    _charts.plot_feature_stats(stats, feature_names)
    plt.show()


def plot_confidence_distribution(confidence: np.ndarray, bins: int = 10) -> None:
    _charts.plot_confidence_distribution(confidence, bins)
    plt.show()


def plot_confidence_vs_errors(confidence: np.ndarray, errors: np.ndarray) -> None:
    _charts.plot_confidence_vs_errors(confidence, errors)
    plt.show()


def plot_tree_complexity(logs: List[Any]) -> None:
    _charts.plot_tree_complexity(logs)
    plt.show()


def plot_summary(logs: List[Any], feature_names: Optional[List[str]] = None) -> None:
    _charts.plot_summary(logs, feature_names)
    plt.show()


def plot_feature_heatmap(
    logs: List[Any],
    top_k: int = 15,
    features: Optional[List] = None,
    metric: str = "gain_share",
    smoothing_window: int = 10,
    cmap: str = "YlOrRd",
    figsize: tuple = (14, 8),
) -> None:
    _charts.plot_feature_heatmap(logs, top_k, features, metric, smoothing_window, cmap, figsize)
    plt.show()
