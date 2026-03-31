# boostwatch

Training-time observability for gradient boosting models.

`boostwatch` captures per-iteration tree structure, split gains, and feature usage during model training — with a single line of code added to your existing workflow.

## Supported frameworks

| Framework | Integration style |
|---|---|
| LightGBM | `model.fit(..., callbacks=observer.callbacks())` |
| XGBoost | `xgb.train(..., callbacks=observer.callbacks())` |
| CatBoost | `model.fit(..., callbacks=observer.callbacks())` |
| NGBoost | `observer.fit(model, X, y)` |
| sklearn GradientBoosting | `observer.fit(X, y)` |

## Installation

Install the core library plus your framework of choice:

```bash
pip install boostwatch[lightgbm]
pip install boostwatch[xgboost]
pip install boostwatch[catboost]
pip install boostwatch[ngboost]
pip install boostwatch[all]      # all frameworks
```

sklearn's `GradientBoostingClassifier` / `GradientBoostingRegressor` work with the core install (no extra needed):

```bash
pip install boostwatch
```

## Quickstart

### LightGBM

```python
import lightgbm as lgb
from boostwatch import watch

model = lgb.LGBMClassifier(n_estimators=100)
observer = watch(model, feature_names=list(X.columns))

model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
          callbacks=observer.callbacks())

observer.plot_summary()
```

### XGBoost

```python
import xgboost as xgb
from boostwatch import watch

model = xgb.XGBClassifier(n_estimators=100)
observer = watch(model, feature_names=list(X.columns))

model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
          callbacks=observer.callbacks())

observer.plot_summary()
```

### CatBoost

```python
from catboost import CatBoostClassifier
from boostwatch import watch

model = CatBoostClassifier(iterations=100, verbose=0)
observer = watch(model, feature_names=list(X.columns))

model.fit(X_train, y_train, callbacks=observer.callbacks())
observer.finalize(model)   # back-fills tree structure post-fit
observer.plot_summary()
```

### sklearn GradientBoosting

```python
from sklearn.ensemble import GradientBoostingClassifier
from boostwatch import watch

model = GradientBoostingClassifier(n_estimators=100)
observer = watch(model, feature_names=list(X.columns))

observer.fit(X_train, y_train)   # trains the model internally
observer.plot_summary()

# fitted model is available at observer.model
preds = observer.model.predict(X_test)
```

### NGBoost

```python
from ngboost import NGBClassifier
from boostwatch import watch

model = NGBClassifier(n_estimators=100)
observer = watch(model, feature_names=list(X.columns))

observer.fit(model, X_train, y_train)
observer.plot_summary()
```

## API reference

### `watch(model, feature_names=None)`

Auto-detects the framework and returns an observer.

### Observer methods

| Method | Description |
|---|---|
| `callbacks()` | Returns framework-specific callbacks to pass to `model.fit()` |
| `fit(X, y)` or `fit(model, X, y)` | For sklearn GBT / NGBoost (no native callbacks) |
| `get_logs()` | Returns list of `IterationLog` objects |
| `feature_stats()` | Per-feature split count, total gain, average gain |
| `tree_stats()` | Per-iteration depth, leaf count, split count |
| `plot_summary()` | Multi-panel training summary figure |

### Visualization themes

```python
from boostwatch.viz.themes import apply_theme

apply_theme("dark")     # or "default", "minimal"
```
