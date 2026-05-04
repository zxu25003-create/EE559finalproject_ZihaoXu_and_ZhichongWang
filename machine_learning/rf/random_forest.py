# random_forest.py

import numpy as np
from decision_tree import DecisionTreeClassifierScratch


class RandomForestClassifierScratch:
    def __init__(self, n_estimators=10, max_depth=8, min_samples_split=10,
                 max_features="sqrt", random_state=None):
        """
        Args:
            n_estimators: number of trees in the forest.
            max_depth: maximum depth of each decision tree.
            min_samples_split: minimum number of samples required to split a node.
            max_features: number of features considered at each split.
                - "sqrt": use sqrt(num_features).
                - None: use all features.
                - int: use the specified number of features.
            random_state: random seed.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state

        self.trees = []

        if self.random_state is not None:
            np.random.seed(self.random_state)

    def fit(self, X, y):
        """
        Train the random forest.

        Each tree is trained on a bootstrap sample of the training set.
        """
        self.trees = []
        num_samples, num_features = X.shape

        # Determine how many features each tree split should consider.
        if self.max_features == "sqrt":
            tree_max_features = max(1, int(np.sqrt(num_features)))
        elif self.max_features is None:
            tree_max_features = num_features
        elif isinstance(self.max_features, int):
            tree_max_features = self.max_features
        else:
            raise ValueError("max_features must be 'sqrt', None, or int")

        for i in range(self.n_estimators):
            # Bootstrap sampling with replacement.
            sample_indices = np.random.choice(num_samples, size=num_samples, replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            # Train one decision tree.
            tree = DecisionTreeClassifierScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=tree_max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

            print(f"Tree {i + 1}/{self.n_estimators} trained.")

    def _get_trees(self, num_trees=None):
        """
        Return either all trees or only the first num_trees trees.
        This is used to create a curve over the number of trees.
        """
        if num_trees is None:
            return self.trees

        if num_trees < 1 or num_trees > len(self.trees):
            raise ValueError("num_trees must be between 1 and the number of trained trees.")

        return self.trees[:num_trees]

    def predict(self, X, num_trees=None):
        """
        Predict labels using majority voting.

        Args:
            X: feature matrix.
            num_trees: if provided, use only the first num_trees trees.
        """
        trees = self._get_trees(num_trees)

        # Store the prediction of each tree for all samples.
        tree_predictions = []

        for tree in trees:
            preds = tree.predict(X)
            tree_predictions.append(preds)

        # shape: (n_trees, n_samples)
        tree_predictions = np.array(tree_predictions)

        # Convert to shape: (n_samples, n_trees)
        tree_predictions = tree_predictions.T

        final_predictions = []
        for sample_preds in tree_predictions:
            values, counts = np.unique(sample_preds, return_counts=True)
            majority_vote = values[np.argmax(counts)]
            final_predictions.append(majority_vote)

        return np.array(final_predictions, dtype=np.int32)

    def predict_proba(self, X, num_trees=None):
        """
        Estimate class probabilities using tree votes.

        Returns:
            probs: shape = (n_samples, 2).
                Column 0: estimated probability of NORMAL class.
                Column 1: estimated probability of PNEUMONIA class.

        The positive-class probability is approximated by the fraction of trees
        voting for class 1.
        """
        trees = self._get_trees(num_trees)

        tree_predictions = []

        for tree in trees:
            preds = tree.predict(X)
            tree_predictions.append(preds)

        # shape: (n_trees, n_samples)
        tree_predictions = np.array(tree_predictions, dtype=np.float32)

        # For each sample, compute the fraction of trees voting for class 1.
        prob_class_1 = np.mean(tree_predictions, axis=0)
        prob_class_0 = 1.0 - prob_class_1

        probs = np.stack([prob_class_0, prob_class_1], axis=1)
        return probs
