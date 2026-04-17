"""Data extraction functions for gradient boosting model analysis.

All functions accept a list of ``IterationLog`` objects and return plain Python
structures or pandas DataFrames.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from ..analysis.feature_stats import compute_feature_stats
from ..analysis.tree_analysis import (
    compute_tree_stats,
    compute_leaf_distribution,
    compute_split_depth_distribution,
)
from ._helpers import _resolve_feature_names


def get_feature_stats(
    logs: List[Any], feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """Return a DataFrame of per-feature split statistics.

    Columns: ``feature_index``, ``name``, ``count``, ``total_gain``, ``avg_gain``.
    Sorted by ``total_gain`` descending.

    Args:
        logs: List of ``IterationLog`` objects.
        feature_names: Optional list of feature names indexed by feature index.
    """
    names = _resolve_feature_names(logs, feature_names)
    stats = compute_feature_stats(logs, names or None)
    rows = []
    for feat_idx, s in stats.items():
        rows.append({
            "feature_index": feat_idx,
            "name": s["name"] or "f{}".format(feat_idx),
            "count": s["count"],
            "total_gain": s["total_gain"],
            "avg_gain": s["avg_gain"],
        })
    df = pd.DataFrame(rows, columns=["feature_index", "name", "count", "total_gain", "avg_gain"])
    return df.sort_values("total_gain", ascending=False).reset_index(drop=True)


def get_iteration_metrics(logs: List[Any]) -> pd.DataFrame:
    """Return a long-format DataFrame of eval metrics over iterations.

    Columns: ``iteration``, ``metric``, ``value``.
    Returns an empty DataFrame if no metrics were logged.

    Args:
        logs: List of ``IterationLog`` objects.
    """
    rows = []
    for log in logs:
        if hasattr(log, "metrics"):
            for metric, value in log.metrics.items():
                rows.append({"iteration": log.iteration, "metric": metric, "value": value})
    return pd.DataFrame(rows, columns=["iteration", "metric", "value"])


def get_tree_stats(logs: List[Any]) -> Dict:
    """Return a dict of per-iteration tree complexity statistics.

    Keys: ``iterations``, ``avg_depth``, ``avg_leaves``, ``total_splits``, ``num_trees``.
    Each value is a list aligned by iteration index.

    Args:
        logs: List of ``IterationLog`` objects.
    """
    return compute_tree_stats(logs)


def get_split_depth_distribution(logs: List[Any]) -> pd.DataFrame:
    """Return a DataFrame of split counts per depth level.

    Columns: ``depth``, ``split_count``. Sorted by depth ascending.

    Args:
        logs: List of ``IterationLog`` objects.
    """
    dist = compute_split_depth_distribution(logs)
    rows = [{"depth": d, "split_count": c} for d, c in sorted(dist.items())]
    return pd.DataFrame(rows, columns=["depth", "split_count"])


def get_leaf_distribution(logs: List[Any]) -> Dict:
    """Return leaf output values and sample counts across all trees.

    Keys: ``leaf_values`` (List[float]), ``leaf_counts`` (List[int]).

    Args:
        logs: List of ``IterationLog`` objects.
    """
    return compute_leaf_distribution(logs)
