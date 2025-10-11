#!/usr/bin/env python3
"""
Decision Tree Classifier using CART algorithm.

A simple implementation that splits data recursively using Gini impurity
to find the best feature and threshold at each node.

Example:
    >>> tree = DecisionTree(max_depth=3)
    >>> tree.fit(X_train, y_train)
    >>> predictions = tree.predict(X_test)
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np


class DecisionTree:
    """Decision Tree class."""

    def __init__(self, max_depth: int = 3, random_state: Optional[int] = None) -> None:
        self.max_depth = max_depth
        self.tree: Optional[Union[Dict[str, Any], np.ndarray]] = None
        self.classes_: Optional[np.ndarray] = None
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        """Fit the decision tree."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.classes_ = np.unique(y)
        self.tree = self._build_tree(X, y, 0)
        return self

    def predict(self, X: np.ndarray) -> List[int]:
        """Predict the class for the given data."""
        return [self._predict_single(x) for x in X]

    def predict_proba(self, X: np.ndarray) -> List[np.ndarray]:
        """Predict class probabilities for the given data."""
        return [self._predict_proba_single(x) for x in X]

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, depth: int
    ) -> Union[Dict[str, Any], np.ndarray]:
        """Build the decision tree."""
        # Stop if max depth or pure node
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return self._get_probs(y)

        # Find best split
        best_gain, best_split = 0, None

        for feature in range(X.shape[1]):
            for threshold in np.unique(X[:, feature]):
                left = X[:, feature] <= threshold

                if not (left.any() and (~left).any()):  # Skip if empty split
                    continue

                # Calculate information gain
                gain = self._gini(y) - sum(
                    mask.sum() / len(y) * self._gini(y[mask]) for mask in [left, ~left]
                )

                if gain > best_gain:
                    best_gain, best_split = gain, (feature, threshold)

        if best_split is None:
            return self._get_probs(y)

        # Split and recurse
        feature, threshold = best_split
        left = X[:, feature] <= threshold
        right = ~left

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(X[left], y[left], depth + 1),
            "right": self._build_tree(X[right], y[right], depth + 1),
        }

    def _gini(self, y: np.ndarray) -> float:
        """Calculate the Gini impurity of the given data."""
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs**2)

    def _get_probs(self, y: np.ndarray) -> np.ndarray:
        """Get class probabilities for a set of labels."""
        if len(y) == 0:
            return np.ones(len(self.classes_)) / len(self.classes_)

        probs = np.zeros(len(self.classes_))
        for i, class_label in enumerate(self.classes_):
            probs[i] = np.sum(y == class_label) / len(y)
        return probs

    def _predict_single(self, x: np.ndarray) -> int:
        """Predict the class for the given data."""
        probs = self._predict_proba_single(x)
        return int(np.argmax(probs))

    def _predict_proba_single(self, x: np.ndarray) -> np.ndarray:
        """Predict class probabilities for a single sample."""
        node = self.tree
        while isinstance(node, dict):
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node

    def print_tree(
        self, node: Optional[Union[Dict[str, Any], np.ndarray]] = None, depth: int = 0
    ) -> None:
        """Print tree structure for understanding."""
        if node is None:
            node = self.tree
        if isinstance(node, dict):
            print(
                "  " * depth + f"Feature {node['feature']} <= {node['threshold']:.2f}"
            )
            self.print_tree(node["left"], depth + 1)
            self.print_tree(node["right"], depth + 1)
        else:
            probs = [f"{p:.2f}" for p in node]
            print("  " * depth + f"Leaf: [{', '.join(probs)}]")


# Demo
if __name__ == "__main__":
    # Test with multivariate data (3 features for teaching)
    np.random.seed(42)
    X = np.random.rand(200, 3) * 4  # 3 features
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 1.5).astype(int)

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Class distribution: {np.bincount(y)}")
    print()

    # Our implementation
    tree = DecisionTree(max_depth=3, random_state=42)
    tree.fit(X, y)
    predictions = tree.predict(X)
    probabilities = tree.predict_proba(X)
    accuracy = np.mean(predictions == y)

    print("=== DECISION TREE ===")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Predictions: {predictions[:10]}")
    print(
        f"Probabilities: {[p.tolist() if hasattr(p, 'tolist') else p for p in probabilities[:5]]}"
    )
    # print the tree
    tree.print_tree()

    # Compare with scikit-learn
    try:
        from sklearn.tree import DecisionTreeClassifier as SKTree

        sk_tree = SKTree(max_depth=3, random_state=42)
        sk_tree.fit(X, y)
        sk_pred = sk_tree.predict(X)
        sk_proba = sk_tree.predict_proba(X)
        sk_acc = np.mean(sk_pred == y)

        print("\n=== SCIKIT-LEARN COMPARISON ===")
        print(f"SK Accuracy: {sk_acc:.3f}")
        print(f"SK Predictions: {sk_pred[:10].tolist()}")
        print(f"SK Probabilities: {sk_proba[:5].tolist()}")
        print(f"Prediction agreement: {np.mean(predictions == sk_pred):.1%}")
    except ImportError:
        print("\n(Install scikit-learn to see comparison)")
