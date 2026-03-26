"""
Basic usage example for boostwatch library.
This replicates the functionality from exploration.ipynb using the boostwatch package.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

# Import boostwatch components
from boostwatch import GBDTObserver, compute_feature_stats
from boostwatch.viz.plotting import (
    plot_splits_per_iteration,
    plot_feature_usage_over_time,
    plot_feature_stats,
    plot_confidence_distribution,
    plot_confidence_vs_errors
)

# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Dataset loaded:")
print(f"Features: {list(X.columns)}")
print(f"Classes: {list(data.target_names)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")

# Create observer and train model
observer = GBDTObserver()

model = lgb.LGBMClassifier(
    n_estimators=30,
    learning_rate=0.1,
    max_depth=-1
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="multi_logloss",
    callbacks=[observer.callback()]
)

# Compute feature statistics
stats = compute_feature_stats(observer.get_logs(), X.columns)

# Generate visualizations
print("\nGenerating visualizations...")
plot_splits_per_iteration(observer.get_logs())
plot_feature_stats(stats, X.columns)
plot_feature_usage_over_time(observer.get_logs(), X.columns)

# Model evaluation
probs = model.predict_proba(X_test)
preds = np.argmax(probs, axis=1)

print(f"\nAccuracy: {accuracy_score(y_test, preds):.4f}")

# Confidence analysis
confidence = np.max(probs, axis=1)
errors = preds != y_test

plot_confidence_distribution(confidence)
plot_confidence_vs_errors(confidence, errors)

print("\nAnalysis complete!")