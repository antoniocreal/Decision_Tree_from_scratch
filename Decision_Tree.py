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
        # print('fit function: ')
        self.n_classes = len(np.unique(y))
        # print('self.n_classes: '+str(self.n_classes)) # All good. It identifies how many features there are 
        self.n_features = X.shape[1] # All good
        # print('self.n_features: '+str(self.n_features))
        self.tree = self.build_tree(X, y) 
        # print('self.tree: '+str(self.tree))

    def best_split(self, X, y):

        for feature_index in range(self.n_features):
                thresholds = np.unique(X[:, feature_index])# For each feature, it finds the unique values along that feature's column. These unique values represent potential split points (thresholds) for that feature.
                # By considering all unique values within each feature, the algorithm evaluates various potential splits and selects the one that minimizes impurity (e.g., Gini impurity or entropy) the most.

                # print(f'thresholds: {thresholds}')

                for threshold in thresholds:
                    # It goes through every threshold, meaning that it goes through every unique value of each feature. and for each threshold, it calculates the gini_impurity
                    # print(f'threshold: {threshold}')
                    left_indices = np.where(X[:, feature_index] <= threshold)[0]
                    right_indices = np.where(X[:, feature_index] > threshold)[0]

                    gini = self._gini_impurity(y[left_indices], y[right_indices])
                    if gini < best_gini:
                        print(f'New gini best: {gini}')
                        best_gini = gini
                        best_criteria = (feature_index, threshold)
                        best_sets = (left_indices, right_indices)
                        print(f'best_sets: {best_sets}')

        return best_gini, best_criteria, best_sets


    def build_tree(self, X, y, depth=0):
        # print('build_tree: ')
        n_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        # print(f'n_samples_per_class: {n_samples_per_class}')
        predicted_class = np.argmax(n_samples_per_class)
        # print(f'predicted_class: {predicted_class}')
        node = Node(value=predicted_class)

        # print(f'X: {X}')
        # print(f'depth: {depth}') # As the function calls itself, it is always adding up 1 after defining a new threshold
        if depth < self.max_depth:
            best_gini = float('inf')
            best_criteria = None
            best_sets = None

            best_gini, best_criteria, best_sets = self.best_split(self, X, y)

            if best_gini != 0: # If best_gini is not equal to 0, it indicates that the current node is not perfectly pure and further splitting is necessary to improve the model's predictive power.
                # If the condition is met, the code recursively calls the build_tree method to construct the left and right child nodes of the current node
                if best_sets[0].size != 0:
                    left = self.build_tree(X[best_sets[0]], y[best_sets[0]], depth + 1)
                else:
                    left = None

                if best_sets[1].size != 0:
                    right = self.build_tree(X[best_sets[1]], y[best_sets[1]], depth + 1)
                else:
                    right = None

                # After constructing the left and right child nodes, the code creates a new Node object (node) representing the current node in the decision tree.
                # It assigns the feature index and threshold that resulted in the best split (best_criteria) to the current node.
                print(f'best_criteria: {best_criteria}')
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
