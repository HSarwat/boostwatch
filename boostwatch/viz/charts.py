# boostwatch/viz/charts.py
"""Plot functions for gradient boosting model analysis.

Every function returns a ``matplotlib.Figure``. None call ``plt.show()``.
The caller decides whether to display or save the figure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from ._helpers import (
    _iter_log,
    _split_feature,
    _split_gain,
    _get_framework,
    _all_equal,
    _note_axis,
    _constant_note,
    _resolve_feature_names,
)


def plot_splits_per_iteration(logs: List[Any]) -> plt.Figure:
    """Return a Figure showing total splits per boosting iteration."""
    iterations = []
    split_counts = []
    for log in logs:
        iteration, splits = _iter_log(log)
        iterations.append(iteration)
        split_counts.append(len(splits))

    if _all_equal(split_counts):
        fig, ax = plt.subplots(figsize=(8, 3))
        _note_axis(
            ax,
            "Splits per Iteration \u2014 Constant",
            _constant_note(_get_framework(logs), "splits", split_counts[0], len(logs)),
        )
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iterations, split_counts, marker="o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Number of splits")
    ax.set_title("Tree Complexity Over Time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feature_usage_over_time(
    logs: List[Any], feature_names: Optional[List[str]] = None
) -> plt.Figure:
    """Return a Figure showing split count per feature across iterations."""
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

    fig, ax = plt.subplots(figsize=(12, 8))
    for f in range(n_features):
        ax.plot(range(n_iters), usage_matrix[:, f], label=names[f], linewidth=2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Split Count")
    ax.set_title("Feature Usage Over Time")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_feature_stats(
    stats: Dict[int, Dict], feature_names: Optional[List[str]] = None
) -> plt.Figure:
    """Return a Figure with split frequency, total gain, and avg gain per feature.

    Note: takes a pre-computed stats dict from compute_feature_stats(), not raw logs.
    """
    sorted_keys = sorted(stats.keys())
    names_list = []
    counts = []
    gains = []
    avg_gains = []

    for i in sorted_keys:
        s = stats[i]
        label = (feature_names[i] if feature_names and i < len(feature_names)
                 else s.get("name") or "f{}".format(i))
        names_list.append(label)
        counts.append(s["count"])
        gains.append(s["total_gain"])
        avg_gains.append(s["avg_gain"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].barh(names_list, counts)
    axes[0].set_title("Split Frequency")
    axes[0].set_xlabel("Count")

    axes[1].barh(names_list, gains)
    axes[1].set_title("Total Gain")
    axes[1].set_xlabel("Gain")

    axes[2].barh(names_list, avg_gains)
    axes[2].set_title("Average Gain per Split")
    axes[2].set_xlabel("Average Gain")

    fig.tight_layout()
    return fig


def plot_confidence_distribution(confidence: np.ndarray, bins: int = 10) -> plt.Figure:
    """Return a Figure with a histogram of prediction confidence values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(confidence, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_title("Prediction Confidence Distribution")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_confidence_vs_errors(confidence: np.ndarray, errors: np.ndarray) -> plt.Figure:
    """Return a Figure with confidence scatter coloured by prediction correctness."""
    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(range(len(confidence)), confidence, c=errors, cmap="coolwarm", alpha=0.7)
    ax.set_title("Confidence vs Errors")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Confidence")
    fig.colorbar(scatter, ax=ax, label="Error (True = Misclassified)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_tree_complexity(logs: List[Any]) -> plt.Figure:
    """Return a Figure with average tree depth and leaf count over iterations."""
    from ..analysis.tree_analysis import compute_tree_stats

    stats = compute_tree_stats(logs)
    if not stats["iterations"]:
        fig, ax = plt.subplots(figsize=(8, 3))
        _note_axis(ax, "Tree Complexity", "No tree structure data available in logs.")
        fig.tight_layout()
        return fig

    framework = _get_framework(logs)
    n_iters = len(logs)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if _all_equal(stats["avg_depth"]):
        value = round(stats["avg_depth"][0])
        _note_axis(axes[0], "Tree Depth \u2014 Constant",
                   _constant_note(framework, "depth", value, n_iters))
    else:
        axes[0].plot(stats["iterations"], stats["avg_depth"], marker="o", color="steelblue")
        axes[0].set_title("Average Tree Depth per Iteration")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Avg Depth")
        axes[0].grid(True, alpha=0.3)

    if _all_equal(stats["avg_leaves"]):
        value = round(stats["avg_leaves"][0])
        _note_axis(axes[1], "Leaf Count \u2014 Constant",
                   _constant_note(framework, "leaves", value, n_iters))
    else:
        axes[1].plot(stats["iterations"], stats["avg_leaves"], marker="o", color="darkorange")
        axes[1].set_title("Average Leaf Count per Iteration")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Avg Leaves")
        axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_summary(logs: List[Any], feature_names: Optional[List[str]] = None) -> plt.Figure:
    """Return a multi-panel summary Figure of the training run."""
    from ..analysis.feature_stats import compute_feature_stats
    from ..analysis.tree_analysis import compute_tree_stats

    names = _resolve_feature_names(logs, feature_names)
    stats = compute_feature_stats(logs, names or None)
    tree_stats = compute_tree_stats(logs)
    framework = _get_framework(logs)
    n_iters = len(logs)

    iterations = []
    split_counts = []
    for log in logs:
        it, splits = _iter_log(log)
        iterations.append(it)
        split_counts.append(len(splits))

    metric_names: List[str] = []
    if logs and hasattr(logs[0], "metrics"):
        metric_names = list(logs[0].metrics.keys())

    n_panels = 2 + (1 if names else 0) + (1 if metric_names else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    panel = 0

    # Panel 1: splits per iteration
    if _all_equal(split_counts):
        _note_axis(axes[panel], "Splits per Iteration \u2014 Constant",
                   _constant_note(framework, "splits", split_counts[0], n_iters))
    else:
        axes[panel].plot(iterations, split_counts, marker="o", color="steelblue")
        axes[panel].set_title("Splits per Iteration")
        axes[panel].set_xlabel("Iteration")
        axes[panel].set_ylabel("Splits")
        axes[panel].grid(True, alpha=0.3)
    panel += 1

    # Panel 2: avg tree depth
    if tree_stats["iterations"] and _all_equal(tree_stats["avg_depth"]):
        value = round(tree_stats["avg_depth"][0])
        _note_axis(axes[panel], "Tree Depth \u2014 Constant",
                   _constant_note(framework, "depth", value, n_iters))
    elif tree_stats["iterations"]:
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
    fig.tight_layout()
    return fig


def plot_feature_heatmap(
    logs: List[Any],
    top_k: int = 15,
    features: Optional[List] = None,
    metric: str = "gain_share",
    smoothing_window: int = 10,
    cmap: str = "YlOrRd",
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """Return a Figure with a feature x iteration heatmap of training dynamics."""
    import pandas as pd
    from matplotlib import gridspec

    if not logs:
        fig, ax = plt.subplots(figsize=(8, 3))
        _note_axis(ax, "Feature Heatmap", "No logs to plot.")
        fig.tight_layout()
        return fig

    n_iters = len(logs)

    names_map: Dict[int, str] = {}
    for log in logs:
        if hasattr(log, "trees"):
            for tree in log.trees:
                for split in tree.splits:
                    if getattr(split, "feature_name", None):
                        names_map[split.feature_index] = split.feature_name

    total_feat_gain: Dict[int, float] = {}
    for log in logs:
        _, splits = _iter_log(log)
        for split in splits:
            f = _split_feature(split)
            g = _split_gain(split)
            total_feat_gain[f] = total_feat_gain.get(f, 0.0) + g

    if not total_feat_gain:
        fig, ax = plt.subplots(figsize=(8, 3))
        _note_axis(ax, "Feature Heatmap",
                   "No split data in logs — tree structure may not be populated yet.")
        fig.tight_layout()
        return fig

    name_to_idx: Dict[str, int] = {v: k for k, v in names_map.items()}

    if features is not None:
        if features and isinstance(features[0], str):
            selected = [name_to_idx[f] for f in features if f in name_to_idx]
        else:
            selected = [int(f) for f in features]
    else:
        by_gain = sorted(total_feat_gain, key=lambda f: total_feat_gain[f], reverse=True)
        selected = by_gain[:top_k]

    if not selected:
        fig, ax = plt.subplots(figsize=(8, 3))
        _note_axis(ax, "Feature Heatmap", "No matching features found.")
        fig.tight_layout()
        return fig

    n_selected = len(selected)
    selected_set = set(selected)

    matrix = np.zeros((n_selected, n_iters))
    total_gain_per_iter = np.zeros(n_iters)
    iter_numbers: List[int] = []

    for col, log in enumerate(logs):
        it, splits = _iter_log(log)
        iter_numbers.append(int(it))
        feat_gain: Dict[int, float] = {}
        feat_count: Dict[int, int] = {}
        iter_total = 0.0
        for split in splits:
            f = _split_feature(split)
            g = _split_gain(split)
            iter_total += g
            if f in selected_set:
                feat_gain[f] = feat_gain.get(f, 0.0) + g
                feat_count[f] = feat_count.get(f, 0) + 1
        total_gain_per_iter[col] = iter_total
        for row, f in enumerate(selected):
            if metric == "gain_share":
                matrix[row, col] = feat_gain.get(f, 0.0) / iter_total if iter_total > 0 else 0.0
            elif metric == "count":
                matrix[row, col] = float(feat_count.get(f, 0))
            else:
                matrix[row, col] = feat_gain.get(f, 0.0)

    if smoothing_window > 1 and n_iters >= smoothing_window:
        for row in range(n_selected):
            matrix[row] = (
                pd.Series(matrix[row])
                .rolling(smoothing_window, min_periods=1, center=True)
                .mean()
                .to_numpy()
            )

    feat_labels = [names_map.get(f, "feat_{}".format(f)) for f in selected]
    tick_step = max(1, n_iters // 10)
    tick_positions = list(range(0, n_iters, tick_step))
    tick_labels = [str(iter_numbers[i]) for i in tick_positions]

    metric_labels = {"gain_share": "Gain Share", "count": "Split Count", "raw_gain": "Total Gain"}
    metric_label = metric_labels.get(metric, metric)

    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5], hspace=0.06, figure=fig)
    ax_curve = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])

    ax_curve.fill_between(range(n_iters), total_gain_per_iter, alpha=0.25, color="steelblue")
    ax_curve.plot(range(n_iters), total_gain_per_iter, color="steelblue", linewidth=1.2)
    ax_curve.set_xlim(-0.5, n_iters - 0.5)
    ax_curve.set_xticks([])
    ax_curve.set_ylabel("Total\nGain", fontsize=8, labelpad=4)
    ax_curve.tick_params(axis="y", labelsize=7)
    ax_curve.grid(True, alpha=0.2)

    im = ax_heat.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0)
    ax_heat.set_yticks(range(n_selected))
    ax_heat.set_yticklabels(feat_labels, fontsize=9)
    ax_heat.set_xticks(tick_positions)
    ax_heat.set_xticklabels(tick_labels, fontsize=8)
    ax_heat.set_xlabel("Iteration", fontsize=10)

    cbar = plt.colorbar(im, ax=ax_heat, shrink=0.7, pad=0.02)
    cbar.set_label(metric_label, fontsize=9)

    title = "Feature {} over Training".format(metric_label)
    if smoothing_window > 1:
        title += "  (smoothing={})".format(smoothing_window)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    return fig


def plot_learning_curve(logs: List[Any]) -> plt.Figure:
    """Return a Figure showing eval metrics over boosting iterations."""
    metric_names: List[str] = []
    if logs and hasattr(logs[0], "metrics"):
        metric_names = list(logs[0].metrics.keys())

    if not metric_names:
        fig, ax = plt.subplots(figsize=(8, 3))
        _note_axis(
            ax, "Learning Curve",
            "No eval metrics found in logs.\n\n"
            "Pass an eval set to your model's fit() call\n"
            "(e.g. eval_set=[(X_val, y_val)] for XGBoost/LightGBM).",
        )
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(10, 5))
    for metric in metric_names:
        iters = []
        vals = []
        for log in logs:
            if hasattr(log, "metrics") and metric in log.metrics:
                iters.append(log.iteration)
                vals.append(log.metrics[metric])
        if vals:
            ax.plot(iters, vals, marker="o", markersize=3, label=metric, linewidth=1.5)

    ax.set_title("Learning Curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Metric Value")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_leaf_distribution(logs: List[Any]) -> plt.Figure:
    """Return a Figure with a histogram of leaf output values across all trees."""
    from ..analysis.tree_analysis import compute_leaf_distribution

    dist = compute_leaf_distribution(logs)
    if not dist["leaf_values"]:
        fig, ax = plt.subplots(figsize=(8, 3))
        _note_axis(ax, "Leaf Distribution", "No leaf data found in logs.")
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(dist["leaf_values"], bins=40, edgecolor="none", alpha=0.8, color="steelblue")
    ax.set_title("Leaf Output Value Distribution")
    ax.set_xlabel("Leaf Value")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_split_depth_dist(logs: List[Any]) -> plt.Figure:
    """Return a Figure showing total splits per depth level across all trees."""
    from ..analysis.tree_analysis import compute_split_depth_distribution

    dist = compute_split_depth_distribution(logs)
    if not dist:
        fig, ax = plt.subplots(figsize=(8, 3))
        _note_axis(ax, "Split Depth Distribution", "No split depth data found in logs.")
        fig.tight_layout()
        return fig

    depths = sorted(dist.keys())
    counts = [dist[d] for d in depths]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(depths, counts, color="mediumseagreen", edgecolor="none")
    ax.set_title("Splits per Depth Level")
    ax.set_xlabel("Depth")
    ax.set_ylabel("Total Splits")
    ax.set_xticks(depths)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig
