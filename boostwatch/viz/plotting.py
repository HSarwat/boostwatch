"""Visualization functions for gradient boosting model analysis.

All functions accept either the new :class:`~boostwatch.core.log_schema.IterationLog`
dataclass format or the legacy flat-dict format for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers — support both IterationLog dataclass and legacy dicts
# ---------------------------------------------------------------------------

def _iter_log(log: Any):
    """Yield (iteration, splits_list) from either format."""
    if hasattr(log, "trees"):
        # New IterationLog dataclass
        splits = [s for tree in log.trees for s in tree.splits]
        return log.iteration, splits
    # Legacy dict
    return log["iteration"], log.get("splits", [])


def _split_feature(split: Any) -> int:
    if hasattr(split, "feature_index"):
        return split.feature_index
    return split["feature"]


def _split_gain(split: Any) -> float:
    if hasattr(split, "gain"):
        return float(split.gain)
    return float(split.get("gain", 0.0))


def _resolve_feature_names(logs: List[Any], feature_names: Optional[List[str]]) -> List[str]:
    """Try to determine feature names from logs if not supplied."""
    if feature_names:
        return list(feature_names)
    # Attempt to collect from SplitInfo.feature_name
    names: Dict[int, str] = {}
    for log in logs:
        if hasattr(log, "trees"):
            for tree in log.trees:
                for split in tree.splits:
                    if getattr(split, "feature_name", None):
                        names[split.feature_index] = split.feature_name
    if names:
        max_idx = max(names.keys())
        return [names.get(i, "f{}".format(i)) for i in range(max_idx + 1)]
    return []


# ---------------------------------------------------------------------------
# Existing API (backward compatible)
# ---------------------------------------------------------------------------

def plot_splits_per_iteration(logs: List[Any]) -> None:
    """Plot the total number of splits per boosting iteration.

    Args:
        logs: List of ``IterationLog`` objects or legacy dicts.
    """
    iterations = []
    split_counts = []

    for log in logs:
        iteration, splits = _iter_log(log)
        iterations.append(iteration)
        split_counts.append(len(splits))

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, split_counts, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Number of splits")
    plt.title("Tree Complexity Over Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_usage_over_time(logs: List[Any], feature_names: Optional[List[str]] = None) -> None:
    """Plot split count per feature across boosting iterations.

    Args:
        logs: List of ``IterationLog`` objects or legacy dicts.
        feature_names: List of feature names (optional if embedded in logs).
    """
    names = _resolve_feature_names(logs, feature_names)
    if not names:
        raise ValueError("feature_names must be provided when logs do not contain feature names.")

    n_features = len(names)
    n_iters = len(logs)
    usage_matrix = np.zeros((n_iters, n_features))

    for i, log in enumerate(logs):
        _, splits = _iter_log(log)
        for split in splits:
            f = _split_feature(split)
            if 0 <= f < n_features:
                usage_matrix[i, f] += 1

    plt.figure(figsize=(12, 8))
    for f in range(n_features):
        plt.plot(range(n_iters), usage_matrix[:, f], label=names[f], linewidth=2)

    plt.xlabel("Iteration")
    plt.ylabel("Split Count")
    plt.title("Feature Usage Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_stats(stats: Dict[int, Dict], feature_names: Optional[List[str]] = None) -> None:
    """Plot split frequency, total gain, and average gain per feature.

    Args:
        stats: Dict from :func:`~boostwatch.analysis.feature_stats.compute_feature_stats`.
        feature_names: Optional override for feature name labels.
    """
    sorted_keys = sorted(stats.keys())
    names = []
    counts = []
    gains = []
    avg_gains = []

    for i in sorted_keys:
        s = stats[i]
        label = (feature_names[i] if feature_names and i < len(feature_names)
                 else s.get("name") or "f{}".format(i))
        names.append(label)
        counts.append(s["count"])
        gains.append(s["total_gain"])
        avg_gains.append(s["avg_gain"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].barh(names, counts)
    axes[0].set_title("Split Frequency")
    axes[0].set_xlabel("Count")

    axes[1].barh(names, gains)
    axes[1].set_title("Total Gain")
    axes[1].set_xlabel("Gain")

    axes[2].barh(names, avg_gains)
    axes[2].set_title("Average Gain per Split")
    axes[2].set_xlabel("Average Gain")

    plt.tight_layout()
    plt.show()


def plot_confidence_distribution(confidence: np.ndarray, bins: int = 10) -> None:
    """Plot a histogram of prediction confidence values.

    Args:
        confidence: Array of prediction confidence values (e.g. max class probability).
        bins: Number of histogram bins.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(confidence, bins=bins, edgecolor="black", alpha=0.7)
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_confidence_vs_errors(confidence: np.ndarray, errors: np.ndarray) -> None:
    """Scatter plot of sample confidence coloured by prediction correctness.

    Args:
        confidence: Array of prediction confidence values.
        errors: Boolean array where ``True`` indicates a misclassification.
    """
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(range(len(confidence)), confidence, c=errors, cmap="coolwarm", alpha=0.7)
    plt.title("Confidence vs Errors")
    plt.xlabel("Sample Index")
    plt.ylabel("Confidence")
    plt.colorbar(scatter, label="Error (True = Misclassified)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# New API
# ---------------------------------------------------------------------------

def plot_tree_complexity(logs: List[Any]) -> None:
    """Plot average tree depth and leaf count over boosting iterations.

    Args:
        logs: List of :class:`~boostwatch.core.log_schema.IterationLog` objects.
    """
    from ..analysis.tree_analysis import compute_tree_stats

    stats = compute_tree_stats(logs)
    if not stats["iterations"]:
        print("No tree structure data available in logs.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(stats["iterations"], stats["avg_depth"], marker="o", color="steelblue")
    axes[0].set_title("Average Tree Depth per Iteration")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Avg Depth")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(stats["iterations"], stats["avg_leaves"], marker="o", color="darkorange")
    axes[1].set_title("Average Leaf Count per Iteration")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Avg Leaves")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_summary(logs: List[Any], feature_names: Optional[List[str]] = None) -> None:
    """Display a multi-panel summary of the training run.

    Panels shown (where data is available):
    - Splits per iteration
    - Average tree depth over time
    - Feature split frequency (top features)
    - Eval metrics over time (if logged)

    Args:
        logs: List of :class:`~boostwatch.core.log_schema.IterationLog` objects
            or legacy dicts.
        feature_names: Optional feature names for the feature importance panel.
    """
    from ..analysis.feature_stats import compute_feature_stats
    from ..analysis.tree_analysis import compute_tree_stats

    names = _resolve_feature_names(logs, feature_names)
    stats = compute_feature_stats(logs, names or None)
    tree_stats = compute_tree_stats(logs)

    # Determine how many metric series we have
    metric_names: List[str] = []
    if logs and hasattr(logs[0], "metrics"):
        metric_names = list(logs[0].metrics.keys())

    n_panels = 2 + (1 if names else 0) + (1 if metric_names else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    panel = 0

    # Panel 1: splits per iteration
    iterations = []
    split_counts = []
    for log in logs:
        it, splits = _iter_log(log)
        iterations.append(it)
        split_counts.append(len(splits))

    axes[panel].plot(iterations, split_counts, marker="o", color="steelblue")
    axes[panel].set_title("Splits per Iteration")
    axes[panel].set_xlabel("Iteration")
    axes[panel].set_ylabel("Splits")
    axes[panel].grid(True, alpha=0.3)
    panel += 1

    # Panel 2: avg tree depth
    if tree_stats["iterations"]:
        axes[panel].plot(tree_stats["iterations"], tree_stats["avg_depth"],
                         marker="o", color="darkorange")
        axes[panel].set_title("Avg Tree Depth")
        axes[panel].set_xlabel("Iteration")
        axes[panel].set_ylabel("Depth")
        axes[panel].grid(True, alpha=0.3)
    else:
        axes[panel].set_title("Avg Tree Depth (N/A)")
        axes[panel].axis("off")
    panel += 1

    # Panel 3: feature split frequency (top 15)
    if names and stats:
        sorted_feats = sorted(stats.items(), key=lambda x: x[1]["count"], reverse=True)[:15]
        feat_labels = [
            stats[i].get("name") or (names[i] if i < len(names) else "f{}".format(i))
            for i, _ in sorted_feats
        ]
        feat_counts = [s["count"] for _, s in sorted_feats]

        axes[panel].barh(feat_labels[::-1], feat_counts[::-1], color="mediumseagreen")
        axes[panel].set_title("Feature Split Frequency (Top {})".format(len(sorted_feats)))
        axes[panel].set_xlabel("Count")
        axes[panel].grid(True, alpha=0.3)
        panel += 1

    # Panel 4: eval metrics
    if metric_names:
        for metric in metric_names:
            metric_iters = []
            metric_vals = []
            for log in logs:
                if hasattr(log, "metrics") and metric in log.metrics:
                    metric_iters.append(log.iteration)
                    metric_vals.append(log.metrics[metric])
            if metric_vals:
                axes[panel].plot(metric_iters, metric_vals, marker="o", label=metric)

        axes[panel].set_title("Eval Metrics")
        axes[panel].set_xlabel("Iteration")
        axes[panel].legend(fontsize=8)
        axes[panel].grid(True, alpha=0.3)

    fig.suptitle("Boostwatch Training Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
