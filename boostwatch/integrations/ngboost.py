"""NGBoost integration for boostwatch.

NGBoost does not have a native callback mechanism.  Instead, use
``observer.fit(model, X, y)`` which calls ``model.fit()`` internally and
then inspects ``model.learners`` to build per-iteration logs.
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..core.base import BaseObserver
from ..core.log_schema import IterationLog, TreeLog
from ..utils.helpers import compute_max_depth, parse_sklearn_tree


class NGBoostObserver(BaseObserver):
    """Observer for NGBoost training.

    Because NGBoost has no native callback system, use :meth:`fit` instead of
    passing callbacks to the model directly::

        observer = NGBoostObserver(feature_names=list(X.columns))
        observer.fit(model, X, y)
        observer.plot_summary()

    Or use the :func:`~boostwatch.watch` factory.
    """

    def callbacks(self) -> list:
        raise NotImplementedError(
            "NGBoost does not support training callbacks. "
            "Use observer.fit(model, X, y) instead."
        )

    def fit(self, model: Any, X: Any, y: Any, **kwargs: Any) -> "NGBoostObserver":
        """Train the NGBoost model and capture per-iteration tree information.

        Args:
            model: An unfitted NGBoost model instance.
            X: Training features.
            y: Training targets.
            **kwargs: Additional keyword arguments forwarded to ``model.fit()``.

        Returns:
            self, so calls can be chained.
        """
        model.fit(X, y, **kwargs)
        self._extract_logs(model)
        return self

    def _extract_logs(self, model: Any) -> None:
        """Extract per-iteration tree structure from a fitted NGBoost model.

        NGBoost stores base learners in ``model.learners``.  Each element of
        ``model.learners`` corresponds to one boosting iteration and is a list
        of base estimators (one per distribution parameter).
        """
        learners = getattr(model, "learners", None)
        if learners is None:
            return

        feature_names = self.feature_names

        for iteration, learner_group in enumerate(learners):
            # learner_group is a list — one base estimator per distribution parameter
            if not isinstance(learner_group, (list, tuple)):
                learner_group = [learner_group]

            trees: List[TreeLog] = []
            for idx, base_estimator in enumerate(learner_group):
                tree_ = getattr(base_estimator, "tree_", None)
                if tree_ is None:
                    continue
                splits, leaves = parse_sklearn_tree(tree_, feature_names)
                trees.append(TreeLog(
                    tree_index=idx,
                    num_leaves=len(leaves),
                    depth=compute_max_depth(splits),
                    splits=splits,
                    leaves=leaves,
                ))

            self._log_iteration(IterationLog(
                iteration=iteration,
                framework="ngboost",
                num_trees=len(trees),
                trees=trees,
                metrics={},
            ))
