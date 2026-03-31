"""sklearn GradientBoosting integration for boostwatch.

sklearn's GradientBoostingClassifier / GradientBoostingRegressor have no
native callback mechanism.  This observer uses ``warm_start=True`` to train
one tree at a time, capturing the new estimator after each increment.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..core.base import BaseObserver
from ..core.log_schema import IterationLog, TreeLog
from ..utils.helpers import compute_max_depth, parse_sklearn_tree


class SklearnGBTObserver(BaseObserver):
    """Observer for sklearn GradientBoostingClassifier / GradientBoostingRegressor.

    Because sklearn GBT has no native callback system, training must go
    through :meth:`fit`::

        from sklearn.ensemble import GradientBoostingClassifier
        from boostwatch import watch

        model = GradientBoostingClassifier(n_estimators=100)
        observer = watch(model, feature_names=list(X.columns))
        observer.fit(X, y)
        observer.plot_summary()

    The model is available at ``observer.model`` after training.
    """

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None) -> None:
        super().__init__(feature_names=feature_names)
        self.model = model
        self._n_estimators_target: int = model.n_estimators

    def callbacks(self) -> list:
        raise NotImplementedError(
            "sklearn GradientBoosting does not support training callbacks. "
            "Use observer.fit(X, y) instead."
        )

    def fit(self, X: Any, y: Any, **kwargs: Any) -> "SklearnGBTObserver":
        """Train the model incrementally, capturing per-iteration tree info.

        Internally sets ``warm_start=True`` on the model and trains one tree
        (or one tree per class) at a time.

        Args:
            X: Training features.
            y: Training targets.
            **kwargs: Additional keyword arguments forwarded to ``model.fit()``.

        Returns:
            self, so calls can be chained.
        """
        model = self.model
        n_target = self._n_estimators_target

        original_warm_start = model.warm_start
        model.warm_start = True

        try:
            model.n_estimators = 1
            model.fit(X, y, **kwargs)
            self._capture_iteration(0)

            for i in range(1, n_target):
                model.n_estimators = i + 1
                model.fit(X, y, **kwargs)
                self._capture_iteration(i)
        finally:
            model.n_estimators = n_target
            model.warm_start = original_warm_start

        return self

    def _capture_iteration(self, i: int) -> None:
        """Extract tree structure from the i-th round of estimators."""
        # estimators_[i] has shape (n_classes,) for multi-class or (1,) for binary/regression
        estimator_group = self.model.estimators_[i]

        trees: List[TreeLog] = []
        for idx, base_tree in enumerate(estimator_group):
            tree_ = getattr(base_tree, "tree_", None)
            if tree_ is None:
                continue
            splits, leaves = parse_sklearn_tree(tree_, self.feature_names)
            trees.append(TreeLog(
                tree_index=idx,
                num_leaves=len(leaves),
                depth=compute_max_depth(splits),
                splits=splits,
                leaves=leaves,
            ))

        self._log_iteration(IterationLog(
            iteration=i,
            framework="sklearn",
            num_trees=len(trees),
            trees=trees,
            metrics={},
        ))
