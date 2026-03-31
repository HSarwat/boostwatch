"""Backward-compatible GBDTObserver (now delegates to LightGBMObserver)."""

from __future__ import annotations

from typing import List, Optional


class GBDTObserver:
    """LightGBM observer — kept for backward compatibility.

    .. deprecated::
        Prefer :func:`boostwatch.watch` with a LightGBM model, or import
        :class:`boostwatch.integrations.LightGBMObserver` directly.

    The ``callback()`` (singular) method is preserved for code that passes
    ``observer.callback()`` to LightGBM's callbacks list.
    """

    def __init__(self, feature_names: Optional[List[str]] = None) -> None:
        from ..integrations.lightgbm import LightGBMObserver
        self._delegate = LightGBMObserver(feature_names=feature_names)
        # Keep .logs as a direct reference so existing code reading
        # observer.logs still works (both point to the same list object).
        self.logs = self._delegate.logs
        self.feature_names = feature_names

    # ------------------------------------------------------------------
    # Backward-compatible single-callback API
    # ------------------------------------------------------------------

    def callback(self):
        """Return a single LightGBM callback function (legacy API)."""
        return self._delegate.callbacks()[0]

    # ------------------------------------------------------------------
    # New multi-callback API (consistent with BaseObserver)
    # ------------------------------------------------------------------

    def callbacks(self) -> list:
        """Return a LightGBM callback list."""
        return self._delegate.callbacks()

    # ------------------------------------------------------------------
    # Log access and analysis helpers
    # ------------------------------------------------------------------

    def get_logs(self) -> list:
        """Return all logged iteration data."""
        return self._delegate.get_logs()

    def feature_stats(self) -> dict:
        """Compute per-feature split statistics."""
        return self._delegate.feature_stats()

    def tree_stats(self) -> dict:
        """Compute per-iteration tree complexity statistics."""
        return self._delegate.tree_stats()

    def plot_summary(self) -> None:
        """Display a multi-panel summary figure."""
        return self._delegate.plot_summary()
