"""CatBoost integration for boostwatch.

CatBoost does not expose full tree structure through training callbacks, so
boostwatch uses a two-tier approach:

1. **Live metrics** — the ``_CatBoostCallback`` captures per-iteration eval
   metrics via ``after_iteration(info)``.
2. **Post-fit tree extraction** — call ``observer.finalize(model)`` after
   ``model.fit()`` to back-fill tree structure into the existing logs using
   CatBoost's post-training model inspection API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.base import BaseObserver
from ..core.log_schema import IterationLog, LeafInfo, SplitInfo, TreeLog


class CatBoostObserver(BaseObserver):
    """Observer for CatBoost training.

    Usage::

        observer = CatBoostObserver(feature_names=list(X.columns))
        model.fit(X, y, callbacks=observer.callbacks())
        # Optionally back-fill tree structure after training:
        observer.finalize(model)

    Or use the :func:`~boostwatch.watch` factory.
    """

    def callbacks(self) -> list:
        """Return a CatBoost-compatible callback list."""
        return [_CatBoostCallback(self)]

    def finalize(self, model: Any) -> None:
        """Back-fill tree structure into logs after model.fit() completes.

        CatBoost's Python API does not expose tree structure during training,
        but ``model.get_tree_leaf_counts()`` and
        ``model.get_feature_importance()`` are available post-fit.

        Args:
            model: A fitted CatBoost model (CatBoostClassifier, CatBoostRegressor, etc.)
        """
        try:
            leaf_counts = model.get_tree_leaf_counts()  # array of shape (n_trees,)
        except Exception:
            return

        try:
            feature_importance = model.get_feature_importance()  # array, one value per feature
        except Exception:
            feature_importance = None

        n_trees = len(leaf_counts)

        for i, log in enumerate(self.logs):
            if i >= n_trees:
                break
            n_leaves = int(leaf_counts[i])
            # Build a minimal TreeLog from available post-fit information.
            # Full split structure is not available without private CatBoost internals.
            splits: List[SplitInfo] = []
            if feature_importance is not None:
                for feat_idx, importance in enumerate(feature_importance):
                    if importance > 0:
                        feat_name = (
                            self.feature_names[feat_idx]
                            if self.feature_names and feat_idx < len(self.feature_names)
                            else None
                        )
                        splits.append(SplitInfo(
                            feature_index=feat_idx,
                            feature_name=feat_name,
                            threshold=0.0,
                            gain=float(importance),
                            depth=0,
                        ))

            tree = TreeLog(
                tree_index=0,
                num_leaves=n_leaves,
                depth=0,  # depth not available without private API
                splits=splits,
                leaves=[
                    LeafInfo(leaf_index=j, leaf_value=0.0, leaf_count=0)
                    for j in range(n_leaves)
                ],
            )
            # Replace the stub trees list with the back-filled one
            self.logs[i] = IterationLog(
                iteration=log.iteration,
                framework=log.framework,
                num_trees=1,
                trees=[tree],
                metrics=log.metrics,
            )


class _CatBoostCallback:
    """CatBoost callback that captures per-iteration metrics."""

    def __init__(self, observer: CatBoostObserver) -> None:
        self._observer = observer

    def after_iteration(self, info: Any) -> bool:
        """Called by CatBoost after each boosting iteration.

        Args:
            info: CatBoost iteration info object with ``.iteration`` and
                  ``.metrics`` attributes.

        Returns:
            ``False`` to continue training, ``True`` to stop early.
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
        return False  # False = continue training
