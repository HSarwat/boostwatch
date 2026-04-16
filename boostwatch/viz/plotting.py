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


def _get_framework(logs: List[Any]) -> Optional[str]:
    """Return the framework name from the first log entry, or None."""
    if logs and hasattr(logs[0], "framework"):
        return logs[0].framework
    return None


def _all_equal(values: list) -> bool:
    """Return True if *values* is non-empty and every element is the same."""
    if not values:
        return False
    return len(set(round(v) for v in values)) == 1


def _note_axis(ax: Any, title: str, body: str) -> None:
    """Render a muted informational note on a turned-off axes panel."""
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    ax.text(
        0.5, 0.5, body,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=9,
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )


def _constant_note(framework: Optional[str], prop: str, value: int, n_iters: int) -> str:
    """Return a human-readable note for a metric that is constant across all iterations."""
    if prop == "splits":
        if value == 0:
            return "No split data found in logs.\n\nTree structure may not be populated yet\n(e.g. call observer.finalize(model) for CatBoost)."
        if framework == "catboost":
            return (
                "CatBoost uses symmetric (oblivious) trees — one split condition per depth level.\n\n"
                "Each tree has a fixed {} split(s) per tree,\n"
                "constant across all {} iterations.".format(value, n_iters)
            )
        return "Split count is constant at {} across all {} iterations.".format(value, n_iters)

    if prop == "depth":
        if framework == "catboost":
            return (
                "CatBoost uses symmetric (oblivious) trees.\n\n"
                "Depth is a hyperparameter — all {} trees share\na fixed depth of {}.".format(
                    n_iters, value)
            )
        if framework == "xgboost":
            return (
                "All XGBoost trees reached max_depth={}.\n\n"
                "Depth is uniform across all {} iterations.".format(value, n_iters)
            )
        if framework == "sklearn":
            return (
                "All sklearn GBT trees grew to depth {} (max_depth).\n\n"
                "Depth is uniform across all {} iterations.".format(value, n_iters)
            )
        return "Tree depth is constant at {} across all {} iterations.".format(value, n_iters)

    if prop == "leaves":
        if framework == "catboost":
            return (
                "CatBoost symmetric trees always have 2^depth leaves.\n\n"
                "All {} trees have exactly {} leaves.".format(n_iters, value)
            )
        if framework == "lightgbm":
            return (
                "All LightGBM trees grew to the maximum num_leaves={}.\n\n"
                "Leaf count is uniform across all {} iterations.".format(value, n_iters)
            )
        return "Leaf count is constant at {} across all {} iterations.".format(value, n_iters)

    return "This metric is constant ({}) across {} iterations.".format(value, n_iters)


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

    If split count is identical across all iterations (e.g. CatBoost oblivious
    trees, or any framework whose trees consistently hit their structural limit),
    an informational note is shown instead of a flat, uninformative line chart.

    Args:
        logs: List of ``IterationLog`` objects or legacy dicts.
    """
    iterations = []
    split_counts = []
    for log in logs:
        iteration, splits = _iter_log(log)
        iterations.append(iteration)
        split_counts.append(len(splits))

    if _all_equal(split_counts):
        _, ax = plt.subplots(figsize=(8, 3))
        _note_axis(
            ax,
            "Splits per Iteration \u2014 Constant",
            _constant_note(_get_framework(logs), "splits", split_counts[0], len(logs)),
        )
        plt.tight_layout()
        plt.show()
        return

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

    _, axes = plt.subplots(1, 3, figsize=(18, 5))

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

    If depth or leaf count is identical across all iterations (e.g. CatBoost
    oblivious trees, LightGBM trees consistently filling num_leaves, XGBoost/
    sklearn trees consistently hitting max_depth), an informational note is
    shown for that panel instead of a flat, uninformative line chart.

    Args:
        logs: List of :class:`~boostwatch.core.log_schema.IterationLog` objects.
    """
    from ..analysis.tree_analysis import compute_tree_stats

    stats = compute_tree_stats(logs)
    if not stats["iterations"]:
        print("No tree structure data available in logs.")
        return

    framework = _get_framework(logs)
    n_iters = len(logs)
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Depth panel
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

    # Leaf count panel
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

    plt.tight_layout()
    plt.show()


def plot_summary(logs: List[Any], feature_names: Optional[List[str]] = None) -> None:
    """Display a multi-panel summary of the training run.

    Panels shown (where data is available):
    - Splits per iteration
    - Average tree depth over time
    - Feature split frequency (top features)
    - Eval metrics over time (if logged)

    For any panel where the metric is constant across all iterations, an
    informational note is shown instead of a flat, uninformative line chart.

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
    framework = _get_framework(logs)
    n_iters = len(logs)

    # Pre-compute split counts so we can detect constancy before drawing
    iterations = []
    split_counts = []
    for log in logs:
        it, splits = _iter_log(log)
        iterations.append(it)
        split_counts.append(len(splits))

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
    plt.tight_layout()
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
    """Plot a feature \u00d7 iteration heatmap showing how feature usage evolves over training.

    The top panel shows total gain per iteration (the learning curve) so you can
    see when the model hits diminishing returns.  The heatmap below encodes how
    each feature's contribution shifts across iterations, revealing:

    - **Early-dominant features** (hot bands on the left) \u2014 the primary signal
      drivers when the residuals are largest.
    - **Late-rising features** (bands that intensify on the right) \u2014 features
      the model leans on once the main signal is captured, which can indicate
      overfitting to noise.

    Args:
        logs: List of :class:`~boostwatch.core.log_schema.IterationLog` objects.
        top_k: Number of top features to display, ranked by total gain across all
            iterations.  Ignored when *features* is provided.
        features: Explicit list of feature indices (``int``) or feature names
            (``str``) to display.  Overrides *top_k*.
        metric: Metric encoded in each heatmap cell:

            - ``"gain_share"`` *(default)* \u2014 each feature's fraction of the total
              gain in that iteration.  Best for comparing early vs late importance
              because it cancels the natural decline in absolute gain.
            - ``"count"`` \u2014 raw split count per feature per iteration.
            - ``"raw_gain"`` \u2014 total gain attributed to the feature per iteration.

        smoothing_window: Rolling-window width applied to each feature's series
            before rendering.  Set to ``1`` to disable.  Defaults to ``10``.
        cmap: Matplotlib colormap name.  Defaults to ``"YlOrRd"``.
        figsize: Figure ``(width, height)`` in inches.

    Note:
        For CatBoost, per-split gain is approximated as
        ``global_importance / split_count`` (the best available proxy without
        per-split loss-reduction data).  ``"count"`` is a more reliable metric
        for CatBoost logs.
    """
    import pandas as pd
    from matplotlib import gridspec

    if not logs:
        print("No logs to plot.")
        return

    n_iters = len(logs)

    # --- Resolve feature name → index mapping from embedded SplitInfo data ----
    names_map: Dict[int, str] = {}
    for log in logs:
        if hasattr(log, "trees"):
            for tree in log.trees:
                for split in tree.splits:
                    if getattr(split, "feature_name", None):
                        names_map[split.feature_index] = split.feature_name

    # --- Accumulate total gain per feature for top-k selection ----------------
    total_feat_gain: Dict[int, float] = {}
    for log in logs:
        _, splits = _iter_log(log)
        for split in splits:
            f = _split_feature(split)
            g = _split_gain(split)
            total_feat_gain[f] = total_feat_gain.get(f, 0.0) + g

    if not total_feat_gain:
        print("No split data in logs — tree structure may not be populated yet.")
        return

    # --- Resolve which features to show ---------------------------------------
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
        print("No matching features found.")
        return

    n_selected = len(selected)
    selected_set = set(selected)

    # --- Build raw matrix (n_selected \u00d7 n_iters) and total-gain curve ----------
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
            else:  # raw_gain
                matrix[row, col] = feat_gain.get(f, 0.0)

    # --- Apply rolling-window smoothing per feature row -----------------------
    if smoothing_window > 1 and n_iters >= smoothing_window:
        for row in range(n_selected):
            matrix[row] = (
                pd.Series(matrix[row])
                .rolling(smoothing_window, min_periods=1, center=True)
                .mean()
                .to_numpy()
            )

    # --- Feature labels (y-axis) ----------------------------------------------
    feat_labels = [names_map.get(f, "feat_{}".format(f)) for f in selected]

    # --- Build x-axis ticks mapped to actual iteration numbers ----------------
    tick_step = max(1, n_iters // 10)
    tick_positions = list(range(0, n_iters, tick_step))
    tick_labels = [str(iter_numbers[i]) for i in tick_positions]

    # --- Plot -----------------------------------------------------------------
    metric_labels = {
        "gain_share": "Gain Share",
        "count":      "Split Count",
        "raw_gain":   "Total Gain",
    }
    metric_label = metric_labels.get(metric, metric)

    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5], hspace=0.06, figure=fig)
    ax_curve = fig.add_subplot(gs[0])
    ax_heat  = fig.add_subplot(gs[1])

    # Learning curve (raw, unsmoothed — shows actual training dynamics)
    ax_curve.fill_between(range(n_iters), total_gain_per_iter, alpha=0.25, color="steelblue")
    ax_curve.plot(range(n_iters), total_gain_per_iter, color="steelblue", linewidth=1.2)
    ax_curve.set_xlim(-0.5, n_iters - 0.5)
    ax_curve.set_xticks([])
    ax_curve.set_ylabel("Total\nGain", fontsize=8, labelpad=4)
    ax_curve.tick_params(axis="y", labelsize=7)
    ax_curve.grid(True, alpha=0.2)

    # Heatmap
    im = ax_heat.imshow(
        matrix, aspect="auto", cmap=cmap,
        interpolation="nearest", vmin=0,
    )
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
    plt.show()
