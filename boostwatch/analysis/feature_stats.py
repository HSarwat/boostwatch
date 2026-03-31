"""Feature statistics computation for gradient boosting models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union


def compute_feature_stats(
    observer_logs: List[Any],
    feature_names: Optional[List[str]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Compute per-feature split statistics from observer logs.

    Accepts both the new :class:`~boostwatch.core.log_schema.IterationLog`
    dataclass format and the legacy flat-dict format produced by the original
    ``GBDTObserver`` for backward compatibility.

    Args:
        observer_logs: List of ``IterationLog`` objects or legacy dicts.
        feature_names: Optional list of feature names indexed by feature index.
            When ``None`` the names stored inside each split are used if
            available.

    Returns:
        Dict mapping feature index to::

            {
                "count":       int,    # total number of splits on this feature
                "total_gain":  float,  # cumulative split gain
                "avg_gain":    float,  # average gain per split
                "name":        str,    # feature name (or None)
            }
    """
    stats: Dict[int, Dict[str, Any]] = {}

    for log in observer_logs:
        splits = _iter_splits(log)
        for split in splits:
            feat_idx, gain, feat_name = _unpack_split(split)

            if feat_idx not in stats:
                stats[feat_idx] = {"count": 0, "total_gain": 0.0, "name": None}

            stats[feat_idx]["count"] += 1
            stats[feat_idx]["total_gain"] += gain

            # Prefer feature_names argument; fall back to name embedded in split
            if feature_names and feat_idx < len(feature_names):
                stats[feat_idx]["name"] = feature_names[feat_idx]
            elif feat_name and stats[feat_idx]["name"] is None:
                stats[feat_idx]["name"] = feat_name

    # Compute averages and fill in any feature indices implied by feature_names
    if feature_names:
        for i, name in enumerate(feature_names):
            if i not in stats:
                stats[i] = {"count": 0, "total_gain": 0.0, "name": name}

    for feat_idx, s in stats.items():
        s["avg_gain"] = s["total_gain"] / s["count"] if s["count"] > 0 else 0.0

    return stats


# ---------------------------------------------------------------------------
# Internal helpers — support both IterationLog dataclass and legacy dicts
# ---------------------------------------------------------------------------

def _iter_splits(log: Any):
    """Yield split objects from either an IterationLog or a legacy log dict."""
    # New dataclass format
    if hasattr(log, "trees"):
        for tree in log.trees:
            for split in tree.splits:
                yield split
        return
    # Legacy flat-dict format: {"iteration": ..., "splits": [...]}
    for split in log.get("splits", []):
        yield split


def _unpack_split(split: Any):
    """Return (feature_index, gain, feature_name) from either format."""
    if hasattr(split, "feature_index"):
        # SplitInfo dataclass
        return split.feature_index, split.gain, getattr(split, "feature_name", None)
    # Legacy dict: {"feature": int, "gain": float}
    return split["feature"], float(split.get("gain", 0.0)), None
