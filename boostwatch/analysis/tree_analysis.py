"""Tree-level analysis functions for gradient boosting models."""

from __future__ import annotations

from typing import Any, Dict, List


def compute_tree_stats(observer_logs: List[Any]) -> Dict[str, Any]:
    """Compute per-iteration tree complexity statistics.

    Args:
        observer_logs: List of :class:`~boostwatch.core.log_schema.IterationLog`
            objects.

    Returns:
        Dict with lists keyed by::

            {
                "iterations":    List[int],   # iteration index
                "avg_depth":     List[float], # mean tree depth per iteration
                "avg_leaves":    List[float], # mean leaf count per iteration
                "total_splits":  List[int],   # total splits across all trees
                "num_trees":     List[int],   # number of trees per iteration
            }
    """
    result: Dict[str, List] = {
        "iterations": [],
        "avg_depth": [],
        "avg_leaves": [],
        "total_splits": [],
        "num_trees": [],
    }

    for log in observer_logs:
        if not hasattr(log, "trees") or not log.trees:
            continue

        depths = [t.depth for t in log.trees]
        leaves = [t.num_leaves for t in log.trees]
        splits = [len(t.splits) for t in log.trees]

        result["iterations"].append(log.iteration)
        result["avg_depth"].append(sum(depths) / len(depths))
        result["avg_leaves"].append(sum(leaves) / len(leaves))
        result["total_splits"].append(sum(splits))
        result["num_trees"].append(len(log.trees))

    return result


def compute_leaf_distribution(observer_logs: List[Any]) -> Dict[str, List]:
    """Collect leaf values and sample counts across all trees and iterations.

    Args:
        observer_logs: List of :class:`~boostwatch.core.log_schema.IterationLog`
            objects.

    Returns:
        Dict with::

            {
                "leaf_values":  List[float],  # all leaf output values
                "leaf_counts":  List[int],    # samples per leaf
            }
    """
    leaf_values: List[float] = []
    leaf_counts: List[int] = []

    for log in observer_logs:
        if not hasattr(log, "trees"):
            continue
        for tree in log.trees:
            for leaf in tree.leaves:
                leaf_values.append(leaf.leaf_value)
                leaf_counts.append(leaf.leaf_count)

    return {"leaf_values": leaf_values, "leaf_counts": leaf_counts}


def compute_split_depth_distribution(observer_logs: List[Any]) -> Dict[int, int]:
    """Count how many splits occur at each depth level across all trees.

    Args:
        observer_logs: List of :class:`~boostwatch.core.log_schema.IterationLog`
            objects.

    Returns:
        Dict mapping depth level (int) to total number of splits at that depth.
    """
    distribution: Dict[int, int] = {}

    for log in observer_logs:
        if not hasattr(log, "trees"):
            continue
        for tree in log.trees:
            for split in tree.splits:
                depth = split.depth
                distribution[depth] = distribution.get(depth, 0) + 1

    return distribution
