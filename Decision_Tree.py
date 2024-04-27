import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold  # Threshold value for the feature
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Prediction value if the node is a leaf

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        predicted_class = np.argmax(n_samples_per_class)
        node = Node(value=predicted_class)

        if depth < self.max_depth:
            best_gini = float('inf')
            best_criteria = None
            best_sets = None

            for feature_index in range(self.n_features):
                thresholds = np.unique(X[:, feature_index])

                for threshold in thresholds:
                    left_indices = np.where(X[:, feature_index] <= threshold)[0]
                    right_indices = np.where(X[:, feature_index] > threshold)[0]

                    gini = self._gini_impurity(y[left_indices], y[right_indices])
                    if gini < best_gini:
                        best_gini = gini
                        best_criteria = (feature_index, threshold)
                        best_sets = (left_indices, right_indices)

            if best_gini != 0:
                left = self._grow_tree(X[best_sets[0]], y[best_sets[0]], depth + 1)
                right = self._grow_tree(X[best_sets[1]], y[best_sets[1]], depth + 1)
                node = Node(feature_index=best_criteria[0], threshold=best_criteria[1], left=left, right=right)

        return node

    def _gini_impurity(self, *groups):
        total_samples = sum(len(group) for group in groups)
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            for class_val in range(self.n_classes):
                p = [np.sum(group == class_val) / size for group in groups]
                score += p[class_val] * p[class_val]
            gini += (1.0 - score) * (size / total_samples)
        return gini

    def predict(self, X):
        return np.array([self._predict_instance(x, self.tree) for x in X])

    def _predict_instance(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_value = x[tree.feature_index]
        if feature_value <= tree.threshold:
            return self._predict_instance(x, tree.left)
        else:
            return self._predict_instance(x, tree.right)

# Example usage:
X = np.array([[2, 4], [4, 2], [1, 3], [3, 1]])
y = np.array([0, 1, 0, 1])

tree = DecisionTreeClassifier(max_depth=2)
tree.fit(X, y)

# Predictions for new instances
print(tree.predict(np.array([[1.5, 3.5], [3.5, 1.5]])))
