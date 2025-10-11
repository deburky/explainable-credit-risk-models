#!/usr/bin/env python3
"""
Exact TreeSHAP (tree-path-dependent) for the custom DecisionTree.

For each sample:
1) Get its decision path.
2) Form coalitions over the unique features occurring on that path (not nodes).
3) For a coalition S, all splits on features in S follow the x-consistent branch;
   all other splits are marginalized using empirical left/right frequencies.
4) Compute Shapley values over those unique features and aggregate per feature index.
"""

import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class TreeSHAP:
    """Compute exact SHAP values for decision trees (tree-path-dependent)."""

    def __init__(
        self, tree: Any, X_train: np.ndarray, model_type: str = "classification"
    ) -> None:
        """
        Parameters
        ----------
        tree : your DecisionTree instance (already fitted)
        X_train : array-like, shape (n_samples, n_features)
            Used to compute per-node cover (branch probabilities).
        model_type : {"classification", "regression"}
        """
        self.tree = tree
        self.X_train = np.asarray(X_train)
        self.model_type = model_type
        self.n_features = self.X_train.shape[1]

        # Build an internal copy of the tree annotated with sample counts
        self.root = self._build_tree(tree.tree, np.arange(len(self.X_train)))

    # Explainer interface
    def explain(self, X: np.ndarray, class_index: Optional[int] = None) -> np.ndarray:
        """
        Compute SHAP values for X.

        Returns
        -------
        If classification:
            array shape (n_samples, n_classes, n_features)
        If regression:
            array shape (n_samples, n_features)
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]

        if self.model_type == "classification":
            n_classes = len(self.tree.classes_)
            shap_values = np.zeros((n_samples, n_classes, self.n_features))
            for i, c in itertools.product(range(n_samples), range(n_classes)):
                shap_values[i, c] = self._explain_single(X[i], class_index=c)
        else:
            shap_values = np.zeros((n_samples, self.n_features))
            for i in range(n_samples):
                shap_values[i] = self._explain_single(X[i], class_index=class_index)
        return shap_values

    # Internal functions
    def _build_tree(
        self, node: Union[Dict[str, Any], np.ndarray], indices: np.ndarray
    ) -> Dict[str, Any]:
        """Copy the learned tree while adding per-node covers."""
        if not isinstance(node, dict):
            return {"is_leaf": True, "value": np.asarray(node), "samples": len(indices)}

        f, t = node["feature"], node["threshold"]
        mask = self.X_train[indices, f] <= t
        left_idx = indices[mask]
        right_idx = indices[~mask]

        return {
            "is_leaf": False,
            "feature": f,
            "threshold": t,
            "left": self._build_tree(node["left"], left_idx),
            "right": self._build_tree(node["right"], right_idx),
            "samples": len(indices),
            "left_samples": len(left_idx),
            "right_samples": len(right_idx),
        }

    def _get_path(
        self, node: Dict[str, Any], x: np.ndarray
    ) -> List[Tuple[Dict[str, Any], Optional[bool]]]:
        """Return decision path as [(node, direction_bool), ... , (leaf, None)]."""
        path = []
        cur = node
        while isinstance(cur, dict) and not cur.get("is_leaf", False):
            go_left = x[cur["feature"]] <= cur["threshold"]
            path.append((cur, go_left))
            cur = cur["left"] if go_left else cur["right"]
        path.append((cur, None))  # leaf
        return path

    def _explain_single(self, x: np.ndarray, class_index: int = 0) -> np.ndarray:
        """Exact Shapley over UNIQUE FEATURES on path (tree-path-dependent)."""
        path = self._get_path(self.root, x)
        path_nodes = [n for (n, _) in path[:-1]]  # exclude leaf

        if not path_nodes:
            return np.zeros(self.n_features)

        # Unique features in order of first appearance on the path
        unique_feats = []
        for n in path_nodes:
            f = n["feature"]
            if f not in unique_feats:
                unique_feats.append(f)
        K = len(unique_feats)

        # Precompute f(S) for all coalitions S over unique features
        f_S = {}
        for mask in range(1 << K):
            f_S[mask] = self._evaluate_subset_unique_feats(
                x, mask, unique_feats, class_index
            )

        # Precompute factorials for Shapley weights
        fact = [1.0] * (K + 1)
        for i in range(1, K + 1):
            fact[i] = fact[i - 1] * i

        def weight(s_size: int) -> float:
            return 1.0 if K <= 1 else fact[s_size] * fact[K - s_size - 1] / fact[K]

        # Aggregate contributions per unique feature
        phi = np.zeros(self.n_features)
        for j, feat in enumerate(unique_feats):
            contrib = 0.0
            for mask in range(1 << K):
                if (mask >> j) & 1:
                    continue
                s_size = bin(mask).count("1")
                contrib += weight(s_size) * (f_S[mask | (1 << j)] - f_S[mask])
            phi[feat] += contrib
        return phi

    def _evaluate_subset_unique_feats(
        self, x: np.ndarray, mask: int, unique_feats: List[int], class_index: int
    ) -> float:
        """
        Evaluate E[f | features in S are revealed], where S is a subset of the
        UNIQUE FEATURES on the decision path. Revealed features force *all* their
        splits to the x-consistent branch; others are marginalized by cover.
        """
        # Build revealed feature set from mask
        revealed = set()
        for j, feat in enumerate(unique_feats):
            if (mask >> j) & 1:
                revealed.add(feat)

        def recurse(node: Dict[str, Any]) -> float:
            if node.get("is_leaf", False):
                v = node["value"]
                # Both leaf types are arrays in our tree; guard anyway.
                if isinstance(v, np.ndarray):
                    if v.ndim == 0:
                        return float(v)
                    # For classification return prob of class_index
                    return float(v[class_index if class_index < v.shape[0] else 0])
                return float(v)

            f = node["feature"]
            t = node["threshold"]
            go_left = x[f] <= t

            if f in revealed:
                child = node["left"] if go_left else node["right"]
                return recurse(child)
            else:
                n = node["samples"]
                if n > 0:
                    p_left = node["left_samples"] / n
                    p_right = node["right_samples"] / n
                else:
                    p_left = p_right = 0.5
                left_val = recurse(node["left"])
                right_val = recurse(node["right"])
                return p_left * left_val + p_right * right_val

        return recurse(self.root)


# Main function
if __name__ == "__main__":
    try:
        from .decision_tree import DecisionTree  # type: ignore
    except ImportError:
        from decision_tree import DecisionTree  # type: ignore

    # Data
    rng = np.random.default_rng(42)
    X = rng.random((200, 3)) * 4
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 1.5).astype(int)

    # Train/test split
    split = 150
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Fit tree
    tree = DecisionTree(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)

    # Explain
    explainer = TreeSHAP(tree, X_train, model_type="classification")
    shap_values = explainer.explain(X_test[:5])

    print("=== Exact TreeSHAP (fixed) ===\n")
    print(f"SHAP values shape: {shap_values.shape}\n")

    for i in range(min(3, len(X_test))):
        pred = tree.predict([X_test[i]])[0]
        proba = tree.predict_proba([X_test[i]])[0]
        print(f"Sample {i}: x={X_test[i]}")
        print(f"  Prediction: {pred}, Probabilities: {proba}")
        print(f"  SHAP values (class 1): {shap_values[i, 1]}")
        print(f"  Sum of SHAP (class 1): {shap_values[i, 1].sum():.4f}")
        print()

    # Optional comparison with shap package (if available)
    try:
        import shap
        from sklearn.tree import DecisionTreeClassifier as SKTree

        print("=== Comparison with official SHAP package ===\n")
        sk = SKTree(max_depth=3, random_state=42)
        sk.fit(X_train, y_train)

        expl_shap = shap.TreeExplainer(sk, feature_perturbation="tree_path_dependent")
        pkg_vals = expl_shap.shap_values(X_test[:5])
        print(f"SHAP package output shape: {np.array(pkg_vals).shape}\n")

        if isinstance(pkg_vals, list):
            print("Class 1 SHAP values (first 3 samples):")
            print(f"pkg:\n{pkg_vals[1][:3]}\n")
            print(f"ours:\n{shap_values[:3, 1]}\n")

            our_flat = shap_values[:5, 1].ravel()
            pkg_flat = np.array(pkg_vals[1]).ravel()
            corr = np.corrcoef(our_flat, pkg_flat)[0, 1]
            l1 = np.abs(our_flat - pkg_flat).sum()
            print(f"Correlation: {corr:.6f}")
            print(f"L1 difference: {l1:.6f}")
    except (ImportError, AttributeError, ValueError) as e:
        print(f"(Could not compare with shap: {e})")
