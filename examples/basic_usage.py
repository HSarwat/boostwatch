"""
Basic usage examples for boostwatch.

Two complementary visualization workflows:
  1. LightGBM      → self-contained HTML report (shareable artifact)
  2. sklearn GBT   → interactive matplotlib plots (live exploration)
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from boostwatch import watch, generate_report

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
# Example 1 — LightGBM → HTML report
# ---------------------------------------------------------------------------
def example_lightgbm_report():
    try:
        import lightgbm as lgb
    except ImportError:
        print("\n[skip] LightGBM not installed. Run: pip install boostwatch[lightgbm]")
        return

    print("\n=== LightGBM → HTML report ===")

    model = lgb.LGBMClassifier(n_estimators=50, learning_rate=0.1, verbose=-1)
    observer = watch(model, feature_names=feature_names)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="multi_logloss",
        callbacks=observer.callbacks(),
    )

    probs = model.predict_proba(X_test)
    preds = np.argmax(probs, axis=1)
    print("Logged {} iterations | Accuracy: {:.4f}".format(
        len(observer.get_logs()), accuracy_score(y_test, preds)
    ))

    output_path = os.path.join(os.path.dirname(__file__), "report.html")
    generate_report(observer.get_logs(), feature_names=feature_names, output_path=output_path)
    print("Report written to: {}".format(os.path.abspath(output_path)))


# ---------------------------------------------------------------------------
# Example 2 — sklearn GBT → interactive matplotlib plots
# ---------------------------------------------------------------------------
def example_sklearn_interactive():
    from sklearn.ensemble import GradientBoostingClassifier

    from boostwatch.viz.data_export import (
        get_tree_stats,
        get_feature_stats,
        get_split_depth_distribution,
    )
    from boostwatch.viz._helpers import _iter_log, _split_feature, _split_gain

    print("\n=== sklearn GBT → training-internals dashboard ===")

    model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
    observer = watch(model, feature_names=feature_names)
    observer.fit(X_train, y_train)

    logs = observer.get_logs()
    preds = np.argmax(observer.model.predict_proba(X_test), axis=1)
    print("Logged {} iterations | Accuracy: {:.4f}".format(
        len(logs), accuracy_score(y_test, preds)
    ))

    # --- Derive training-internal signals from the logs ---
    iterations = [log.iteration for log in logs]
    iter_total_gain = []
    for log in logs:
        _, splits = _iter_log(log)
        iter_total_gain.append(sum(_split_gain(s) for s in splits))

    tree = get_tree_stats(logs)
    depth_dist = get_split_depth_distribution(logs)
    top_feats = get_feature_stats(logs, feature_names).head(5)

    # Per-iteration gain share for the top features
    top_idx = top_feats["feature_index"].tolist()
    top_names = top_feats["name"].tolist()
    gain_share = np.zeros((len(top_idx), len(logs)))
    for j, log in enumerate(logs):
        _, splits = _iter_log(log)
        col_total = sum(_split_gain(s) for s in splits) or 1.0
        for s in splits:
            fidx = _split_feature(s)
            if fidx in top_idx:
                gain_share[top_idx.index(fidx), j] += _split_gain(s) / col_total

    # --- Dashboard ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Boostwatch — sklearn GBT Training Internals",
                 fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(iterations, iter_total_gain, marker="o", color="#4a90d9")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Total split gain")
    ax.set_title("Learning Signal — Total Gain per Iteration")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(tree["iterations"], tree["avg_depth"], marker="o", color="#4a90d9", label="Avg depth")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Avg depth", color="#4a90d9")
    ax.tick_params(axis="y", labelcolor="#4a90d9")
    ax2 = ax.twinx()
    ax2.plot(tree["iterations"], tree["avg_leaves"], marker="s", color="#d9534f", label="Avg leaves")
    ax2.set_ylabel("Avg leaves", color="#d9534f")
    ax2.tick_params(axis="y", labelcolor="#d9534f")
    ax.set_title("Tree Complexity Drift"); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    im = ax.imshow(gain_share, aspect="auto", cmap="YlOrRd",
                   extent=(iterations[0], iterations[-1], len(top_names), 0))
    ax.set_yticks(np.arange(len(top_names)) + 0.5)
    ax.set_yticklabels(top_names)
    ax.set_xlabel("Iteration")
    ax.set_title("Feature Gain Share Over Iterations (top 5)")
    fig.colorbar(im, ax=ax, label="Gain share")

    ax = axes[1, 1]
    ax.bar(depth_dist["depth"], depth_dist["split_count"], color="#4a90d9", edgecolor="white")
    ax.set_xlabel("Split depth"); ax.set_ylabel("Number of splits")
    ax.set_title("Where Splits Happen in Trees"); ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    print("Opening dashboard window — close it to exit.")
    plt.show()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    example_lightgbm_report()
    example_sklearn_interactive()
    print("\nDone.")
