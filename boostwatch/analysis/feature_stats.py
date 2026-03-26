"""Feature statistics computation for gradient boosting models."""

from typing import Dict, List
import numpy as np


def compute_feature_stats(observer_logs: List[Dict], feature_names: List[str]) -> Dict[int, Dict]:
    """Compute feature statistics from observer logs.
    
    Args:
        observer_logs: List of dictionaries from GBDTObserver.logs
        feature_names: List of feature names corresponding to indices
        
    Returns:
        Dictionary mapping feature indices to statistics dictionaries
    """
    # Initialize stats for each feature
    stats = {
        i: {
            "count": 0,
            "total_gain": 0.0
        }
        for i in range(len(feature_names))
    }

    # Aggregate statistics from all iterations
    for log in observer_logs:
        for split in log["splits"]:
            f = split["feature"]
            g = split["gain"]

            stats[f]["count"] += 1
            stats[f]["total_gain"] += g

    # Compute averages
    for f in stats:
        if stats[f]["count"] > 0:
            stats[f]["avg_gain"] = stats[f]["total_gain"] / stats[f]["count"]
        else:
            stats[f]["avg_gain"] = 0.0

    return stats