"""Boostwatch: Training-time observability for gradient boosting models."""

from __future__ import annotations

from typing import List, Optional

from .core.observer import GBDTObserver
from .core.base import BaseObserver
from .analysis.feature_stats import compute_feature_stats
from .viz.plotting import (
    plot_splits_per_iteration,
    plot_feature_usage_over_time,
    plot_feature_stats,
    plot_confidence_distribution,
    plot_confidence_vs_errors,
    plot_summary,
    plot_tree_complexity,
)

__version__ = "0.2.0"
__author__ = "Boostwatch Contributors"


def watch(model, feature_names: Optional[List[str]] = None) -> BaseObserver:
    """Auto-detect the gradient boosting framework and return the correct observer.

    Supported frameworks: LightGBM, XGBoost, CatBoost, NGBoost, and sklearn
    GradientBoostingClassifier / GradientBoostingRegressor.

    For LightGBM, XGBoost, and CatBoost, pass the returned observer's
    callbacks to ``model.fit()``::

        observer = watch(model, feature_names=list(X.columns))
        model.fit(X, y, callbacks=observer.callbacks())
        observer.plot_summary()

    For sklearn GBT and NGBoost, call ``observer.fit()`` instead::

        observer = watch(model, feature_names=list(X.columns))
        observer.fit(X, y)          # sklearn GBT
        # or
        observer.fit(model, X, y)  # NGBoost
        observer.plot_summary()

    Args:
        model: An instantiated (not yet fitted) gradient boosting model.
        feature_names: Optional list of feature names. When provided, split
            info will include human-readable feature names.

    Returns:
        A :class:`~boostwatch.core.base.BaseObserver` subclass appropriate
        for the detected framework.

    Raises:
        ValueError: If the model's framework is not recognised.
    """
    module = getattr(type(model), "__module__", "") or ""
    name = type(model).__name__

    if module.startswith("lightgbm"):
        from .integrations.lightgbm import LightGBMObserver
        return LightGBMObserver(feature_names=feature_names)

    if module.startswith("xgboost"):
        from .integrations.xgboost import XGBoostObserver
        return XGBoostObserver(feature_names=feature_names)

    if module.startswith("catboost"):
        from .integrations.catboost import CatBoostObserver
        return CatBoostObserver(feature_names=feature_names)

    if module.startswith("ngboost"):
        from .integrations.ngboost import NGBoostObserver
        return NGBoostObserver(feature_names=feature_names)

    if module.startswith("sklearn") and "GradientBoosting" in name:
        from .integrations.sklearn_gbt import SklearnGBTObserver
        return SklearnGBTObserver(model, feature_names=feature_names)

    raise ValueError(
        "Unsupported model type: {}.{}.\n"
        "Supported frameworks: LightGBM, XGBoost, CatBoost, NGBoost, "
        "sklearn GradientBoostingClassifier/GradientBoostingRegressor.".format(module, name)
    )


__all__ = [
    # Primary API
    "watch",
    "BaseObserver",
    # Backward compat
    "GBDTObserver",
    # Analysis
    "compute_feature_stats",
    # Visualizations
    "plot_splits_per_iteration",
    "plot_feature_usage_over_time",
    "plot_feature_stats",
    "plot_confidence_distribution",
    "plot_confidence_vs_errors",
    "plot_summary",
    "plot_tree_complexity",
]
