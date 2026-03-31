# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Boostwatch** is a training-time observability library for gradient boosting models. Compatible with LightGBM, XGBoost, CatBoost, NGBoost, and sklearn GradientBoosting.

## Environment & Commands

This project uses **uv** as the package manager (not pip). Python 3.8+.

```bash
# Install core + a framework extra
uv pip install -e .[lightgbm]
uv pip install -e .[all]        # all frameworks

# Run examples
python examples/basic_usage.py

# Launch Jupyter
uv run jupyter notebook exploration.ipynb
```

No test suite exists yet.

## Architecture

**Primary API**: `watch(model, feature_names=...)` — auto-detects the framework and returns a `BaseObserver` subclass.

### Layer overview

1. **`boostwatch/core/log_schema.py`** — Unified dataclasses: `SplitInfo`, `LeafInfo`, `TreeLog`, `IterationLog`. All frameworks produce logs in this format.

2. **`boostwatch/core/base.py`** — `BaseObserver` ABC. Holds `self.logs: List[IterationLog]`, defines the `callbacks()` abstract method, and delegates to `feature_stats()`, `tree_stats()`, `plot_summary()`.

3. **`boostwatch/integrations/`** — One observer per framework:
   - `lightgbm.py`: `LightGBMObserver` — uses `env.model.dump_model()` inside a LightGBM callback
   - `xgboost.py`: `XGBoostObserver` — subclasses `xgb.callback.TrainingCallback`, uses `model.get_dump(dump_format='json')`
   - `catboost.py`: `CatBoostObserver` — captures metrics live; call `observer.finalize(model)` post-fit for tree structure
   - `ngboost.py`: `NGBoostObserver` — no callbacks; `observer.fit(model, X, y)` wraps fit then inspects `model.learners`
   - `sklearn_gbt.py`: `SklearnGBTObserver` — no callbacks; `observer.fit(X, y)` uses `warm_start=True` to train one tree at a time

4. **`boostwatch/utils/helpers.py`** — Tree traversal/parsing: `traverse_lgb_tree()`, `parse_xgb_tree_json()`, `parse_sklearn_tree()`, `compute_max_depth()`.

5. **`boostwatch/analysis/`** — Pure functions over `List[IterationLog]`:
   - `feature_stats.py`: `compute_feature_stats()` — split count, total/avg gain per feature
   - `tree_analysis.py`: `compute_tree_stats()`, `compute_leaf_distribution()`, `compute_split_depth_distribution()`

6. **`boostwatch/viz/plotting.py`** — All matplotlib plots. Accepts both new `IterationLog` and legacy flat-dict format. Key functions: `plot_summary()`, `plot_tree_complexity()`, `plot_feature_stats()`, `plot_confidence_distribution()`.

7. **`boostwatch/viz/themes.py`** — `apply_theme(name)` — sets matplotlib rcParams. Themes: `"default"`, `"dark"`, `"minimal"`.

### Data flow

```
watch(model) → <FrameworkObserver>
    ↓ training callback / observer.fit()
IterationLog(trees=[TreeLog(splits=[SplitInfo...], leaves=[LeafInfo...])], metrics={})
    ↓
analysis functions → viz functions
```

### Backward compatibility

`GBDTObserver` (in `core/observer.py`) delegates to `LightGBMObserver`. Its `callback()` (singular) method is preserved for code predating v0.2.

## Key dependencies

| Package | Role | Required? |
|---|---|---|
| `numpy`, `pandas` | Numerical computation | Core |
| `matplotlib` | All visualizations | Core |
| `scikit-learn` | sklearn GBT + shared utilities | Core |
| `lightgbm` / `xgboost` / `catboost` / `ngboost` | Framework integrations | Optional extras |
