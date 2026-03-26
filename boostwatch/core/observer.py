"""Core observer for monitoring gradient boosting training iterations."""

from typing import List, Dict, Any, Callable
import numpy as np


class GBDTObserver:
    """Observer to monitor gradient boosting decision tree training.
    
    This observer captures detailed information about splits, features used,
    and tree structure at each boosting iteration.
    """
    
    def __init__(self):
        """Initialize the observer with empty logs."""
        self.logs: List[Dict[str, Any]] = []

    def log_iteration(self, env) -> None:
        """Log information from a single boosting iteration.
        
        Args:
            env: LightGBM callback environment containing model and iteration info
        """
        model = env.model
        iteration = env.iteration
        
        # Get model dump for the current iteration
        dump = model.dump_model()
        tree_info = dump["tree_info"][iteration]
        tree = tree_info["tree_structure"]

        splits = []
        
        def traverse(node: Dict[str, Any]) -> None:
            """Recursively traverse tree nodes to extract split information."""
            if "split_feature" in node:
                splits.append({
                    "feature": node["split_feature"],
                    "threshold": node["threshold"],
                    "gain": node.get("split_gain", 0.0)
                })
                traverse(node["left_child"])
                traverse(node["right_child"])

        traverse(tree)

        self.logs.append({
            "iteration": iteration,
            "num_leaves": tree_info["num_leaves"],
            "splits": splits,
        })

    def callback(self) -> Callable:
        """Create a callback function for LightGBM training.
        
        Returns:
            Callback function that logs iteration information
        """
        def _callback(env):
            self.log_iteration(env)
        return _callback

    def get_logs(self) -> List[Dict[str, Any]]:
        """Get all logged iteration information.
        
        Returns:
            List of dictionaries containing iteration logs
        """
        return self.logs.copy()