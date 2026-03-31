"""LightGBM integration for boostwatch."""

from __future__ import annotations

from typing import List, Optional

from ..core.base import BaseObserver
from ..core.log_schema import IterationLog, TreeLog
from ..utils.helpers import compute_max_depth, traverse_lgb_tree


class LightGBMObserver(BaseObserver):
    """Observer for LightGBM training.

    Pass the result of :meth:`callbacks` to LightGBM's ``callbacks``
    parameter::

        observer = LightGBMObserver(feature_names=list(X.columns))
        model.fit(X, y, callbacks=observer.callbacks())

    Or use the :func:`~boostwatch.watch` factory which auto-detects the
    framework::

        from boostwatch import watch
        observer = watch(model, feature_names=list(X.columns))
        model.fit(X, y, callbacks=observer.callbacks())
    """

    def callbacks(self) -> list:
        """Return a LightGBM-compatible callback list."""

        def _callback(env) -> None:
            model = env.model
            iteration = env.iteration

            dump = model.dump_model()
            raw_tree_info = dump.get("tree_info", [])

            # For multi-class models LightGBM emits num_class trees per round.
            # dump_model() returns ALL trees up to the current iteration, so we
            # grab the trees added in this round by looking at the last
            # num_class entries (1 for binary/regression).
            num_class = dump.get("num_class", 1)
            round_trees = raw_tree_info[-(num_class):]

            trees: List[TreeLog] = []
            for idx, t in enumerate(round_trees):
                splits, leaves = traverse_lgb_tree(t["tree_structure"])
                trees.append(TreeLog(
                    tree_index=idx,
                    num_leaves=t.get("num_leaves", len(leaves)),
                    depth=compute_max_depth(splits),
                    splits=splits,
                    leaves=leaves,
                ))

            # evaluation_result_list: [(dataset, metric, value, higher_is_better), ...]
            metrics = {}
            for entry in (env.evaluation_result_list or []):
                dataset, metric, value = entry[0], entry[1], entry[2]
                metrics["{}-{}".format(dataset, metric)] = float(value)

            self._log_iteration(IterationLog(
                iteration=iteration,
                framework="lightgbm",
                num_trees=len(trees),
                trees=trees,
                metrics=metrics,
            ))

        _callback.order = 10  # LightGBM requires the order attribute
        return [_callback]
