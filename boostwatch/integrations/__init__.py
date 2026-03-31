"""Framework-specific observer integrations for boostwatch.

Each observer is imported lazily so that the corresponding framework package
is only required when the observer is actually used.
"""

from __future__ import annotations


def _lazy_import(module_path: str, class_name: str):
    """Return a callable that imports and returns the named class on first call."""
    def _getter(*args, **kwargs):
        import importlib
        mod = importlib.import_module(module_path, package=__name__)
        cls = getattr(mod, class_name)
        return cls(*args, **kwargs)
    _getter.__name__ = class_name
    return _getter


# Convenience re-exports (import only when the module is accessed)
def __getattr__(name: str):
    _map = {
        "LightGBMObserver": (".lightgbm", "LightGBMObserver"),
        "XGBoostObserver": (".xgboost", "XGBoostObserver"),
        "CatBoostObserver": (".catboost", "CatBoostObserver"),
        "NGBoostObserver": (".ngboost", "NGBoostObserver"),
        "SklearnGBTObserver": (".sklearn_gbt", "SklearnGBTObserver"),
    }
    if name in _map:
        module_path, class_name = _map[name]
        import importlib
        mod = importlib.import_module(module_path, package=__package__)
        return getattr(mod, class_name)
    raise AttributeError("module 'boostwatch.integrations' has no attribute '{}'".format(name))


__all__ = [
    "LightGBMObserver",
    "XGBoostObserver",
    "CatBoostObserver",
    "NGBoostObserver",
    "SklearnGBTObserver",
]
