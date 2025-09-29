"""
Real AdaBoost in Python.
Author: https://github.com/deburky
Source: https://github.com/pedwardsada/real_adaboost
"""

import math
import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import logistic
from sklearn.datasets import make_classification
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class RealAdaBoost:
    """Real-valued AdaBoost classifier using decision trees as weak learners.

    This implementation learns per-leaf real-valued contributions and aggregates
    them additively. Probabilities are obtained by passing the decision scores
    through the logistic CDF.

    Parameters
    ----------
    n_estimators : int, default=25
        Number of boosting rounds (weak learners).
    max_depth : int, default=3
        Maximum depth of each decision tree learner.
    learning_rate : float, default=0.2
        Shrinkage applied to each learner's leaf contribution.
    sample_prop : float, default=1.0
        Proportion of samples drawn (with replacement) at each round using the
        current weights.
    interaction : bool, default=True
        If False, restricts trees to consider one feature at a time via
        `max_features=1`.
    random_state : int, default=1234
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators=25,
        max_depth=3,
        learning_rate=0.2,
        sample_prop=1.0,
        interaction=True,
        random_state=1234,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.sample_prop = sample_prop
        self.interaction = interaction
        self.random_seed = int(random_state)
        self.rng = np.random.default_rng(self.random_seed)
        self.models = []
        self.leaf_scores = []  # stores fm per leaf

    def _fit_tree(self, X, y, sample_weight):
        """Fit a single decision tree with sample weights.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training features for this round.
        y : ndarray of shape (n_samples,)
            Binary labels {0,1}.
        sample_weight : ndarray of shape (n_samples,)
            Weights to emphasize currently hard examples.
        """
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            max_features=None if self.interaction else 1,
            random_state=self.random_seed,
        )
        tree.fit(X, y, sample_weight=sample_weight)
        return tree

    def fit(self, X, y):
        """Fit the AdaBoost ensemble.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training features.
        y : ndarray of shape (n_samples,)
            Binary labels {0,1}.
        """
        n_samples = X.shape[0]
        # initialize weights
        w = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            # bootstrap / weighted sample
            sample_size = int(self.sample_prop * n_samples)
            sample_idx = self.rng.choice(
                n_samples, size=sample_size, replace=True, p=w / w.sum()
            )
            X_sample, y_sample, w_sample = X[sample_idx], y[sample_idx], w[sample_idx]

            tree = self._fit_tree(X_sample, y_sample, w_sample)
            leaf_ids = tree.apply(X)
            unique_leaves = np.unique(leaf_ids)

            fm_leaf = {}
            for leaf in unique_leaves:
                mask = leaf_ids == leaf
                w_leaf = w[mask]
                y_leaf = y[mask]
                if w_leaf.sum() == 0:
                    continue
                p1 = (w_leaf[y_leaf == 1].sum()) / w_leaf.sum()
                # avoid perfect probabilities
                p1 = np.clip(p1, 1e-6, 1 - 1e-6)
                fm = 0.5 * np.log(p1 / (1 - p1)) * self.learning_rate
                fm_leaf[leaf] = fm

            # update weights
            f = np.array([fm_leaf.get(lid, 0.0) for lid in leaf_ids])
            w *= np.exp(-(2 * y - 1) * f)  # y∈{0,1} -> {-1,+1}
            w /= w.sum()

            self.models.append(tree)
            self.leaf_scores.append(fm_leaf)

        return self

    def decision_function(self, X):
        """Compute raw decision scores f(x) for samples X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        ndarray of shape (n_samples,)
            Additive scores before link function.
        """
        f_sum = np.zeros(X.shape[0])
        for tree, leaf_map in zip(self.models, self.leaf_scores, strict=False):
            leaf_ids = tree.apply(X)
            f_sum += np.array([leaf_map.get(lid, 0.0) for lid in leaf_ids])
        return f_sum

    def predict_proba(self, X):
        """Predict class probabilities for samples X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Columns correspond to P(class=0), P(class=1).
        """
        scores = self.decision_function(X)
        proba_pos = logistic.cdf(scores)
        return np.vstack([1 - proba_pos, proba_pos]).T

    def predict(self, X):
        """Predict class labels {0,1} for samples X using f(x) >= 0.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted labels {0,1}.
        """
        return (self.decision_function(X) >= 0).astype(int)

    def decision_function_partial(self, X, n_rounds=None):
        """Compute decision function using only the first n_rounds learners.

        If n_rounds is None, all learners are used.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.
        n_rounds : int or None, optional
            Number of initial learners to include.

        Returns
        -------
        ndarray of shape (n_samples,)
            Partial additive scores.
        """
        if n_rounds is None:
            n_rounds = len(self.models)
        n_rounds = max(0, min(n_rounds, len(self.models)))
        f_sum = np.zeros(X.shape[0])
        for tree, leaf_map in list(zip(self.models, self.leaf_scores, strict=False))[
            :n_rounds
        ]:
            leaf_ids = tree.apply(X)
            f_sum += np.array([leaf_map.get(lid, 0.0) for lid in leaf_ids])
        return f_sum


def _plot_background(ax, xx, yy, zz, cmap):
    """Draw filled contour background for decision surface.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    xx, yy : ndarray
        Meshgrid coordinates.
    zz : ndarray
        Decision scores over the grid, shaped like xx.
    cmap : str or Colormap
        Colormap name or object.
    """
    return ax.contourf(xx, yy, zz, levels=20, cmap=cmap, alpha=0.5, antialiased=True)


def plot_ensemble_boundary(
    model, X, y, title="Ensemble decision boundary", n_rounds=None, cmap="bwr"
):
    """Plot the ensemble decision boundary for 2D X.

    Parameters
    ----------
    model : RealAdaBoost
        Trained RealAdaBoost model.
    X : array-like of shape (n_samples, 2)
        Training data with exactly two features.
    y : array-like of shape (n_samples,)
        Binary labels {0,1}.
    title : str
        Plot title.
    n_rounds : int or None
        If provided, use only the first n_rounds weak learners.
    cmap : str or Colormap, default="bwr"
        Matplotlib colormap for background.
    """
    if X.shape[1] != 2:
        raise ValueError("plot_ensemble_boundary requires X to have exactly 2 features")

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.decision_function_partial(grid, n_rounds=n_rounds).reshape(xx.shape)

    _fig, ax = plt.subplots(figsize=(6, 5))
    _plot_background(ax, xx, yy, zz, cmap)
    sns.scatterplot(
        x=X[:, 0],
        y=X[:, 1],
        hue=y,
        palette=["tab:blue", "tab:red"],
        edgecolor="k",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(title="class", loc="best")
    plt.tight_layout()
    return ax


def plot_tree_boundaries(
    model, X, y, cmap="bwr", n_cols=3, titles_prefix="Tree", include_ensemble_panel=True
):
    """Plot decision boundaries of each tree in the ensemble for 2D X.

    Uses sklearn's DecisionBoundaryDisplay for each weak learner. If
    `include_ensemble_panel=True`, an extra last subplot shows the full ensemble
    decision surface in addition to all individual trees.

    Parameters
    ----------
    model : RealAdaBoost
        Trained RealAdaBoost model.
    X : array-like of shape (n_samples, 2)
        Feature matrix with exactly two columns.
    y : array-like of shape (n_samples,)
        Binary labels {0,1}.
    cmap : str or Colormap, default="bwr"
        Colormap for background decision regions.
    n_cols : int, default=3
        Number of subplot columns.
    titles_prefix : str, default="Tree"
        Prefix used for each subplot title.
    include_ensemble_panel : bool, default=True
        If True, add an extra subplot showing the ensemble prediction.
    """
    if X.shape[1] != 2:
        raise ValueError("plot_tree_boundaries requires X to have exactly 2 features")

    n_trees = len(model.models)
    n_cols = max(1, n_cols)
    n_panels = n_trees + (1 if include_ensemble_panel else 0)
    n_rows = math.ceil(n_panels / n_cols)
    _fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(-1)

    # Plot each individual tree
    for idx in range(n_trees):
        ax = axes[idx]
        tree = model.models[idx]
        DecisionBoundaryDisplay.from_estimator(
            tree,
            X,
            response_method="predict",
            cmap=cmap,
            alpha=0.5,
            ax=ax,
        )
        sns.scatterplot(
            x=X[:, 0],
            y=X[:, 1],
            hue=y,
            palette=["tab:blue", "tab:red"],
            edgecolor="k",
            ax=ax,
            legend=False,
        )
        ax.set_title(f"{titles_prefix} {idx + 1}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    # Add ensemble panel at the end
    if include_ensemble_panel:
        ax = axes[n_trees]
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300),
            np.linspace(y_min, y_max, 300),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = model.decision_function(grid).reshape(xx.shape)
        _plot_background(ax, xx, yy, zz, cmap)
        sns.scatterplot(
            x=X[:, 0],
            y=X[:, 1],
            hue=y,
            palette=["tab:blue", "tab:red"],
            edgecolor="k",
            ax=ax,
            legend=False,
        )
        ax.set_title("Ensemble")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

    # Hide any unused axes
    for ax in axes[n_panels:]:
        ax.axis("off")

    plt.tight_layout()
    return axes[:n_panels]


# ============================
# Example usage with fake data
# ============================
if __name__ == "__main__":
    # Create images directory if it doesn't exist
    os.makedirs("images", exist_ok=True)

    X_data, y_data = make_classification(
        n_samples=1500,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.2,
        random_state=4321,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_data, y_data, test_size=0.3, random_state=1234
    )

    # Train model
    model = RealAdaBoost(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.3,
        interaction=True,
        random_state=1234,
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    print("Validation Gini:", auc * 2 - 1)

    # Per-tree boundaries
    # Set font to Avenir
    plt.rcParams["font.family"] = "Avenir"

    plot_tree_boundaries(
        model,
        X_train,
        y_train,
        n_cols=3,
        titles_prefix="Tree",
        cmap="bwr",
        include_ensemble_panel=True,
    )
    plt.savefig("images/adaboost_trees.png", dpi=300, bbox_inches="tight")

    # Full ensemble boundary
    plot_ensemble_boundary(
        model,
        X_train,
        y_train,
        title="Ensemble decision boundary (all rounds)",
        cmap="bwr",
    )
    plt.savefig("images/adaboost_ensemble_all.png", dpi=300, bbox_inches="tight")

    # Partial ensemble boundary (first 3 rounds)
    plot_ensemble_boundary(
        model,
        X_train,
        y_train,
        title="Ensemble decision boundary (first 3 rounds)",
        n_rounds=3,
        cmap="bwr",
    )
    plt.savefig("images/adaboost_ensemble_first3.png", dpi=300, bbox_inches="tight")
