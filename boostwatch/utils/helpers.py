"""Tree traversal and parsing utilities for all supported frameworks."""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..core.log_schema import LeafInfo, SplitInfo


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

def traverse_lgb_tree(
    node: dict,
    depth: int = 0,
    leaf_counter: Optional[List[int]] = None,
) -> Tuple[List[SplitInfo], List[LeafInfo]]:
    """Recursively traverse a LightGBM dump_model() tree node.

    Args:
        node: A node dict from LightGBM's dump_model()["tree_info"][i]["tree_structure"]
        depth: Current recursion depth (used to annotate split depth)
        leaf_counter: Mutable list with a single integer used to assign leaf indices

    Returns:
        Tuple of (splits, leaves) lists
    """
    if leaf_counter is None:
        leaf_counter = [0]

    splits: List[SplitInfo] = []
    leaves: List[LeafInfo] = []

    if "split_feature" in node:
        splits.append(SplitInfo(
            feature_index=node["split_feature"],
            feature_name=node.get("split_feature_name"),
            threshold=float(node.get("threshold", 0.0)),
            gain=float(node.get("split_gain", 0.0)),
            depth=depth,
        ))
        left_splits, left_leaves = traverse_lgb_tree(node["left_child"], depth + 1, leaf_counter)
        right_splits, right_leaves = traverse_lgb_tree(node["right_child"], depth + 1, leaf_counter)
        splits.extend(left_splits)
        splits.extend(right_splits)
        leaves.extend(left_leaves)
        leaves.extend(right_leaves)
    else:
        # Leaf node
        idx = leaf_counter[0]
        leaf_counter[0] += 1
        leaves.append(LeafInfo(
            leaf_index=idx,
            leaf_value=float(node.get("leaf_value", 0.0)),
            leaf_count=int(node.get("leaf_count", 0)),
        ))

    return splits, leaves


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

def parse_xgb_tree_json(node: dict) -> Tuple[List[SplitInfo], List[LeafInfo]]:
    """Parse an XGBoost tree from get_dump(dump_format='json').

    XGBoost node fields for split nodes: nodeid, depth, split, split_condition,
    gain, cover, children.
    XGBoost node fields for leaf nodes: nodeid, leaf.

    Args:
        node: Root node dict parsed from XGBoost JSON dump

    Returns:
        Tuple of (splits, leaves) lists
    """
    splits: List[SplitInfo] = []
    leaves: List[LeafInfo] = []

    def _traverse(n: dict) -> None:
        if "children" in n:
            # Split node — feature is stored as "f0", "f1", or a name string
            feature_str = n.get("split", "")
            try:
                feature_index = int(feature_str.lstrip("f"))
                feature_name = None
            except (ValueError, AttributeError):
                feature_index = -1
                feature_name = feature_str

            splits.append(SplitInfo(
                feature_index=feature_index,
                feature_name=feature_name,
                threshold=float(n.get("split_condition", 0.0)),
                gain=float(n.get("gain", 0.0)),
                depth=int(n.get("depth", 0)),
            ))
            for child in n["children"]:
                _traverse(child)
        else:
            # Leaf node
            leaves.append(LeafInfo(
                leaf_index=int(n.get("nodeid", 0)),
                leaf_value=float(n.get("leaf", 0.0)),
                leaf_count=int(round(float(n.get("cover", 0)))),
            ))

    _traverse(node)
    return splits, leaves


# ---------------------------------------------------------------------------
# sklearn / NGBoost (both use DecisionTreeRegressor internally)
# ---------------------------------------------------------------------------

def parse_sklearn_tree(
    tree_,
    feature_names: Optional[List[str]] = None,
) -> Tuple[List[SplitInfo], List[LeafInfo]]:
    """Parse a fitted sklearn DecisionTreeRegressor.tree_ object.

    Args:
        tree_: The .tree_ attribute of a fitted sklearn DecisionTreeRegressor
        feature_names: Optional list of feature names (indexed by feature column)

    Returns:
        Tuple of (splits, leaves) lists
    """
    TREE_LEAF = -1  # sklearn sentinel value for leaves

    splits: List[SplitInfo] = []
    leaves: List[LeafInfo] = []
    leaf_counter = [0]

    def _traverse(node_id: int, depth: int) -> None:
        left = tree_.children_left[node_id]
        right = tree_.children_right[node_id]

        if left == TREE_LEAF:
            leaves.append(LeafInfo(
                leaf_index=leaf_counter[0],
                leaf_value=float(tree_.value[node_id].flat[0]),
                leaf_count=int(tree_.n_node_samples[node_id]),
            ))
            leaf_counter[0] += 1
        else:
            feat_idx = int(tree_.feature[node_id])
            feat_name = (
                feature_names[feat_idx]
                if feature_names and feat_idx < len(feature_names)
                else None
            )
            splits.append(SplitInfo(
                feature_index=feat_idx,
                feature_name=feat_name,
                threshold=float(tree_.threshold[node_id]),
                gain=float(tree_.impurity[node_id]),
                depth=depth,
            ))
            _traverse(left, depth + 1)
            _traverse(right, depth + 1)

    _traverse(0, 0)
    return splits, leaves


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def compute_max_depth(splits: List[SplitInfo]) -> int:
    """Return the maximum split depth from a list of SplitInfo objects."""
    if not splits:
        return 0
    return max(s.depth for s in splits)
