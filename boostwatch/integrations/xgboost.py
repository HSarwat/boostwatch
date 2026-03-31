"""XGBoost integration for boostwatch."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..core.base import BaseObserver
from ..core.log_schema import IterationLog, TreeLog
from ..utils.helpers import compute_max_depth, parse_xgb_tree_json


class XGBoostObserver(BaseObserver):
    """Observer for XGBoost training.

    Pass the result of :meth:`callbacks` to XGBoost's ``callbacks``
    parameter::

        observer = XGBoostObserver(feature_names=list(X.columns))

        # With xgb.train():
        bst = xgb.train(params, dtrain, callbacks=observer.callbacks())

        # With the sklearn API:
        model.fit(X, y, callbacks=observer.callbacks())

    Or use the :func:`~boostwatch.watch` factory.
    """

    def __init__(self, feature_names: Optional[List[str]] = None) -> None:
        super().__init__(feature_names=feature_names)
        # We lazily build the callback so that xgboost is only imported when
        # this class is actually used.
        self._callback: Optional[Any] = None

    def callbacks(self) -> list:
        """Return an XGBoost-compatible callback list."""
        if self._callback is None:
            self._callback = _XGBoostCallback(self)
        return [self._callback]


class _XGBoostCallback:
    """XGBoost TrainingCallback wrapper — avoids hard import at module level."""

    def __init__(self, observer: XGBoostObserver) -> None:
        self._observer = observer
        # Dynamically inherit from xgboost.callback.TrainingCallback so that
        # xgboost is only imported when this class is instantiated.
        try:
            import xgboost as xgb
            self.__class__ = type(
                "_XGBoostCallbackImpl",
                (self.__class__, xgb.callback.TrainingCallback),
                {},
            )
            # Call TrainingCallback.__init__ on self
            xgb.callback.TrainingCallback.__init__(self)
        except ImportError:
            pass  # If xgboost isn't installed the user will get an error at fit time

    def after_iteration(self, model: Any, epoch: int, evals_log: Dict) -> bool:
        """Called by XGBoost after each boosting round."""
        dump = model.get_dump(dump_format="json")

        trees: List[TreeLog] = []
        for idx, tree_json in enumerate(dump):
            node = json.loads(tree_json)
            splits, leaves = parse_xgb_tree_json(node)
            trees.append(TreeLog(
                tree_index=idx,
                num_leaves=len(leaves),
                depth=compute_max_depth(splits),
                splits=splits,
                leaves=leaves,
            ))

        metrics: Dict[str, float] = {}
        for dataset, metric_dict in evals_log.items():
            for metric, values in metric_dict.items():
                metrics["{}-{}".format(dataset, metric)] = float(values[-1])

        self._observer._log_iteration(IterationLog(
            iteration=epoch,
            framework="xgboost",
            num_trees=len(trees),
            trees=trees,
            metrics=metrics,
        ))
        return False  # returning False means "do not stop training"
