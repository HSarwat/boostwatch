"""CatBoost integration for boostwatch.

CatBoost does not expose full tree structure through training callbacks, so
boostwatch uses a two-tier approach:

1. **Live metrics** — the ``_CatBoostCallback`` captures per-iteration eval
   metrics via ``after_iteration(info)``.
2. **Post-fit tree extraction** — call ``observer.finalize(model)`` after
   ``model.fit()`` to back-fill tree structure into the existing logs using
   CatBoost's internal model object (``model._object``).

The internal ``_get_tree_splits(tree_idx, pool)`` method returns the exact
feature index and threshold for each split level in every oblivious tree,
giving accurate per-tree feature usage without requiring file I/O or a Pool.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ..core.base import BaseObserver
from ..core.log_schema import IterationLog, LeafInfo, SplitInfo, TreeLog


class CatBoostObserver(BaseObserver):
    """Observer for CatBoost training.

    Usage::

        observer = CatBoostObserver(feature_names=list(X.columns))
        model.fit(X, y, callbacks=observer.callbacks())
        # Back-fill tree structure after training:
        observer.finalize(model)

    Or use the :func:`~boostwatch.watch` factory.
    """

    def callbacks(self) -> list:
        """Return a CatBoost-compatible callback list."""
        return [_CatBoostCallback(self)]

    def finalize(self, model: Any) -> None:
        """Back-fill tree structure into logs after model.fit() completes.

        Uses ``model._object._get_tree_splits(tree_idx, None)`` to read the
        exact feature index and threshold at each split level of every
        oblivious tree, and ``model._object._get_leaf_weights()`` for actual
        sample counts at each leaf.

        Feature split count (across all trees) is the primary importance proxy.
        Per-split gain is set to ``global_importance / split_count`` so that
        ``feature_stats().avg_gain ≈ global_importance`` per feature.

        Args:
            model: A fitted CatBoost model (CatBoostClassifier,
                CatBoostRegressor, etc.)
        """
        obj = getattr(model, "_object", None)
        if obj is None:
            return

        try:
            n_trees = int(obj._get_tree_count())
            leaf_counts = obj._get_tree_leaf_counts()   # int array, one per tree
            leaf_weights_flat = obj._get_leaf_weights() # float array, all leaves
            leaf_values_flat  = obj._get_leaf_values()  # float array, all leaves
        except Exception:
            return

        # --- First pass: count splits per feature across all trees -----------
        # Used to compute the per-split gain proxy.
        split_counts: Dict[int, int] = {}
        for i in range(min(n_trees, len(self.logs))):
            try:
                raw = obj._get_tree_splits(i, None)
            except Exception:
                continue
            for entry in raw:
                feat_idx = _parse_split_feat(entry)
                if feat_idx >= 0:
                    split_counts[feat_idx] = split_counts.get(feat_idx, 0) + 1

        # --- Global importance as gain proxy ---------------------------------
        # PredictionValuesChange: different from LGB/XGB gain (loss reduction
        # per split), but the best per-feature signal available without data.
        try:
            global_imp = model.get_feature_importance()
        except Exception:
            global_imp = None

        per_split_gain: Dict[int, float] = {}
        if global_imp is not None:
            for feat_idx, count in split_counts.items():
                if feat_idx < len(global_imp) and count > 0:
                    per_split_gain[feat_idx] = float(global_imp[feat_idx]) / count

        # --- Build feature index → name map ----------------------------------
        idx_to_name: Dict[int, str] = {}
        if self.feature_names:
            for i, name in enumerate(self.feature_names):
                idx_to_name[i] = name
        else:
            try:
                for i, name in enumerate(obj._get_feature_names()):
                    idx_to_name[i] = name
            except Exception:
                pass

        # --- Second pass: build TreeLog for each iteration -------------------
        leaf_offset = 0
        for i, log in enumerate(self.logs):
            if i >= n_trees:
                break

            n_leaves = int(leaf_counts[i])
            try:
                raw_splits = obj._get_tree_splits(i, None)
            except Exception:
                raw_splits = []

            depth = len(raw_splits)

            # Build SplitInfo: one entry per depth level (one feature per level
            # in CatBoost's oblivious/symmetric trees)
            splits: List[SplitInfo] = []
            for level, entry in enumerate(raw_splits):
                feat_idx = _parse_split_feat(entry)
                threshold = _parse_split_threshold(entry)
                splits.append(SplitInfo(
                    feature_index=feat_idx,
                    feature_name=idx_to_name.get(feat_idx),
                    threshold=threshold,
                    gain=per_split_gain.get(feat_idx, 0.0),
                    depth=level,
                ))

            # Build LeafInfo with actual sample counts from _get_leaf_weights()
            tree_weights = leaf_weights_flat[leaf_offset:leaf_offset + n_leaves]
            tree_values  = leaf_values_flat[leaf_offset:leaf_offset + n_leaves]
            leaves: List[LeafInfo] = []
            for j in range(n_leaves):
                leaves.append(LeafInfo(
                    leaf_index=j,
                    leaf_value=float(tree_values[j]) if j < len(tree_values) else 0.0,
                    leaf_count=int(round(float(tree_weights[j])))
                    if j < len(tree_weights) else 0,
                ))

            leaf_offset += n_leaves

            self.logs[i] = IterationLog(
                iteration=log.iteration,
                framework=log.framework,
                num_trees=1,
                trees=[TreeLog(
                    tree_index=0,
                    num_leaves=n_leaves,
                    depth=depth,
                    splits=splits,
                    leaves=leaves,
                )],
                metrics=log.metrics,
            )


def _parse_split_feat(entry: str) -> int:
    """Parse feature index from a CatBoost split string like '0, bin=5.74365'."""
    try:
        return int(entry.split(",")[0].strip())
    except (ValueError, IndexError):
        return -1


def _parse_split_threshold(entry: str) -> float:
    """Parse threshold from a CatBoost split string like '0, bin=5.74365'."""
    try:
        return float(entry.split("bin=")[1].strip())
    except (ValueError, IndexError):
        return 0.0


class _CatBoostCallback:
    """CatBoost callback that captures per-iteration metrics."""

    def __init__(self, observer: CatBoostObserver) -> None:
        self._observer = observer

    def after_iteration(self, info: Any) -> bool:
        """Called by CatBoost after each boosting iteration.

        Returns:
            ``True`` to continue training, ``False`` to stop early.
            (CatBoost convention: True=continue, False=stop.)
        """
        metrics: Dict[str, float] = {}
        if hasattr(info, "metrics") and info.metrics:
            for dataset, metric_dict in info.metrics.items():
                for metric_name, values in metric_dict.items():
                    if values:
                        metrics["{}-{}".format(dataset, metric_name)] = float(values[-1])

        self._observer._log_iteration(IterationLog(
            iteration=int(info.iteration),
            framework="catboost",
            num_trees=0,
            trees=[],
            metrics=metrics,
        ))
        return True  # True = continue training (CatBoost: True=continue, False=stop)
