# boostwatch/viz/_helpers.py
"""Internal helpers shared across all viz modules.

Do not import this module directly from outside boostwatch/viz/.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _iter_log(log: Any):
    """Return (iteration, splits_list) from either IterationLog or legacy dict."""
    if hasattr(log, "trees"):
        splits = [s for tree in log.trees for s in tree.splits]
        return log.iteration, splits
    return log["iteration"], log.get("splits", [])


def _split_feature(split: Any) -> int:
    """Return the feature index from a SplitInfo dataclass or legacy dict."""
    if hasattr(split, "feature_index"):
        return split.feature_index
    return int(split["feature"])


def _split_gain(split: Any) -> float:
    """Return the gain value from a SplitInfo dataclass or legacy dict."""
    if hasattr(split, "gain"):
        return float(split.gain)
    return float(split.get("gain", 0.0))


def _get_framework(logs: List[Any]) -> Optional[str]:
    """Return the framework name from the first log entry, or None."""
    if logs and hasattr(logs[0], "framework"):
        return logs[0].framework
    return None


def _all_equal(values: list) -> bool:
    """Return True if values is non-empty and every element is the same."""
    if not values:
        return False
    return max(values) - min(values) < 0.5


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
            return (
                "No split data found in logs.\n\n"
                "Tree structure may not be populated yet\n"
                "(e.g. call observer.finalize(model) for CatBoost)."
            )
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
