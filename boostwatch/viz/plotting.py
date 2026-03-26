"""Visualization functions for gradient boosting model analysis."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any


def plot_splits_per_iteration(logs: List[Dict[str, Any]]) -> None:
    """Plot the number of splits per iteration.
    
    Args:
        logs: List of dictionaries from GBDTObserver.logs
    """
    iterations = []
    split_counts = []

    for log in logs:
        iterations.append(log["iteration"])
        split_counts.append(len(log["splits"]))

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, split_counts, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Number of splits")
    plt.title("Tree Complexity Over Time")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_feature_usage_over_time(logs: List[Dict[str, Any]], feature_names: List[str]) -> None:
    """Plot feature usage over time.
    
    Args:
        logs: List of dictionaries from GBDTObserver.logs
        feature_names: List of feature names
    """
    n_features = len(feature_names)
    n_iters = len(logs)

    usage_matrix = np.zeros((n_iters, n_features))

    for i, log in enumerate(logs):
        for split in log["splits"]:
            f = split["feature"]
            usage_matrix[i, f] += 1

    plt.figure(figsize=(12, 8))
    for f in range(n_features):
        plt.plot(range(n_iters), usage_matrix[:, f], label=feature_names[f], linewidth=2)

    plt.xlabel("Iteration")
    plt.ylabel("Split Count")
    plt.title("Feature Usage Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_stats(stats: Dict[int, Dict], feature_names: List[str]) -> None:
    """Plot feature statistics.
    
    Args:
        stats: Dictionary of feature statistics from compute_feature_stats
        feature_names: List of feature names
    """
    names = []
    counts = []
    gains = []
    avg_gains = []

    for i, name in enumerate(feature_names):
        names.append(name)
        counts.append(stats[i]["count"])
        gains.append(stats[i]["total_gain"])
        avg_gains.append(stats[i]["avg_gain"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Split Frequency
    axes[0].barh(names, counts)
    axes[0].set_title("Split Frequency")
    axes[0].set_xlabel("Count")

    # Total Gain
    axes[1].barh(names, gains)
    axes[1].set_title("Total Gain")
    axes[1].set_xlabel("Gain")

    # Average Gain per Split
    axes[2].barh(names, avg_gains)
    axes[2].set_title("Average Gain per Split")
    axes[2].set_xlabel("Average Gain")

    plt.tight_layout()
    plt.show()


def plot_confidence_distribution(confidence: np.ndarray, bins: int = 10) -> None:
    """Plot prediction confidence distribution.
    
    Args:
        confidence: Array of prediction confidence values
        bins: Number of bins for histogram
    """
    plt.figure(figsize=(10, 6))
    plt.hist(confidence, bins=bins, edgecolor='black', alpha=0.7)
    plt.title("Prediction Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_confidence_vs_errors(confidence: np.ndarray, errors: np.ndarray) -> None:
    """Plot confidence vs errors.
    
    Args:
        confidence: Array of prediction confidence values
        errors: Boolean array indicating prediction errors
    """
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(range(len(confidence)), confidence, c=errors, cmap='coolwarm', alpha=0.7)
    plt.title("Confidence vs Errors (colored)")
    plt.xlabel("Sample Index")
    plt.ylabel("Confidence")
    plt.colorbar(scatter, label='Error (True=Misclassified)')
    plt.grid(True, alpha=0.3)
    plt.show()