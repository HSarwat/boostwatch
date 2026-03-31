"""Unified log schema for all gradient boosting framework observers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SplitInfo:
    """Information about a single split node in a decision tree."""

    feature_index: int
    threshold: float
    gain: float
    depth: int
    feature_name: Optional[str] = None


@dataclass
class LeafInfo:
    """Information about a leaf node in a decision tree."""

    leaf_index: int
    leaf_value: float
    leaf_count: int  # number of training samples reaching this leaf


@dataclass
class TreeLog:
    """Logged information for a single decision tree."""

    tree_index: int
    num_leaves: int
    depth: int  # max depth of the tree
    splits: List[SplitInfo] = field(default_factory=list)
    leaves: List[LeafInfo] = field(default_factory=list)


@dataclass
class IterationLog:
    """Logged information for a single boosting iteration."""

    iteration: int
    framework: str  # "lightgbm", "xgboost", "catboost", "ngboost", "sklearn"
    num_trees: int  # >1 for multi-class (one tree per class per round)
    trees: List[TreeLog] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
