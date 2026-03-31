"""Abstract base class for all boostwatch framework observers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .log_schema import IterationLog


class BaseObserver(ABC):
    """Abstract base for all gradient boosting framework observers.

    Subclasses implement :meth:`callbacks` to return framework-specific
    callback objects.  All observers share the same log format
    (:class:`~boostwatch.core.log_schema.IterationLog`) so that analysis and
    visualization functions work identically regardless of the framework.

    Usage (LightGBM / XGBoost / CatBoost)::

        observer = watch(model, feature_names=list(X.columns))
        model.fit(X, y, callbacks=observer.callbacks())
        observer.plot_summary()

    Usage (sklearn GBT / NGBoost)::

        observer = watch(model, feature_names=list(X.columns))
        observer.fit(model, X, y)
        observer.plot_summary()
    """

    def __init__(self, feature_names: Optional[List[str]] = None) -> None:
        self.logs: List[IterationLog] = []
        self.feature_names: Optional[List[str]] = feature_names

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def callbacks(self) -> list:
        """Return framework-specific callback object(s) to pass to model.fit().

        For frameworks without native callback support (sklearn GBT, NGBoost)
        this method raises :class:`NotImplementedError`; use :meth:`fit`
        instead.
        """

    # ------------------------------------------------------------------
    # Log access
    # ------------------------------------------------------------------

    def get_logs(self) -> List[IterationLog]:
        """Return a copy of all logged iteration data."""
        return list(self.logs)

    def _log_iteration(self, log: IterationLog) -> None:
        """Append a new :class:`IterationLog` entry (called by subclasses)."""
        self.logs.append(log)

    # ------------------------------------------------------------------
    # Convenience analysis / visualization delegates
    # ------------------------------------------------------------------

    def feature_stats(self) -> Dict[int, Dict[str, Any]]:
        """Compute per-feature split statistics across all logged iterations.

        Returns:
            Dict mapping feature index to ``{"count", "total_gain",
            "avg_gain", "name"}`` statistics.
        """
        from ..analysis.feature_stats import compute_feature_stats
        return compute_feature_stats(self.logs, self.feature_names)

    def tree_stats(self) -> Dict[str, Any]:
        """Compute per-iteration tree complexity statistics.

        Returns:
            Dict with keys ``iterations``, ``avg_depth``, ``avg_leaves``,
            ``total_splits``.
        """
        from ..analysis.tree_analysis import compute_tree_stats
        return compute_tree_stats(self.logs)

    def plot_summary(self) -> None:
        """Display a multi-panel summary figure for the training run."""
        from ..viz.plotting import plot_summary
        plot_summary(self.logs, self.feature_names)
