"""
Basic usage examples for boostwatch.

Demonstrates the watch() API with:
  1. LightGBM (callbacks pattern)
  2. sklearn GradientBoostingClassifier (fit() pattern)
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from boostwatch import watch
from boostwatch.viz.plotting import plot_confidence_distribution, plot_confidence_vs_errors

# ---------------------------------------------------------------------------
# Shared dataset
# ---------------------------------------------------------------------------
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Dataset: Iris ({} train, {} test)".format(len(X_train), len(X_test)))
print("Features:", feature_names)


# ---------------------------------------------------------------------------
# Example 1 — LightGBM
# ---------------------------------------------------------------------------
def example_lightgbm():
    try:
        import lightgbm as lgb
    except ImportError:
        print("\n[skip] LightGBM not installed. Run: pip install boostwatch[lightgbm]")
        return

    print("\n=== LightGBM example ===")

    model = lgb.LGBMClassifier(n_estimators=30, learning_rate=0.1, verbose=-1)
    observer = watch(model, feature_names=feature_names)

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="multi_logloss",
        callbacks=observer.callbacks(),
    )

    print("Logged {} iterations".format(len(observer.get_logs())))
    observer.plot_summary()

    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    print("Accuracy: {:.4f}".format(accuracy_score(y_test, preds)))

    confidence = np.max(probs, axis=1)
    errors = preds != y_test
    plot_confidence_distribution(confidence)
    plot_confidence_vs_errors(confidence, errors)


# ---------------------------------------------------------------------------
# Example 2 — sklearn GradientBoostingClassifier
# ---------------------------------------------------------------------------
def example_sklearn():
    from sklearn.ensemble import GradientBoostingClassifier

    print("\n=== sklearn GradientBoostingClassifier example ===")

    model = GradientBoostingClassifier(n_estimators=30, learning_rate=0.1, random_state=42)
    observer = watch(model, feature_names=feature_names)

    # observer.fit() trains the model internally using warm_start
    observer.fit(X_train, y_train)

    print("Logged {} iterations".format(len(observer.get_logs())))
    observer.plot_summary()

    probs = observer.model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    print("Accuracy: {:.4f}".format(accuracy_score(y_test, preds)))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    example_lightgbm()
    example_sklearn()
    print("\nDone.")
