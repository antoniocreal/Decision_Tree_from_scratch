import numpy as np
from Decision_Tree import DecisionTreeClassifier

# Define the file path relative to the current directory
file_path = "banknote_data/Dataset.txt"

# Load the data from the text file
data = np.loadtxt(file_path, delimiter=',')

print(data)

# # Example usage:
# X = np.array([[2, 4], [4, 2], [1, 3], [3, 1]])
# y = np.array([0, 1, 0, 1])

# tree = DecisionTreeClassifier(max_depth=2)
# tree.fit(X, y)

# # Predictions for new instances
# print(tree.predict(np.array([[1.5, 3.5], [3.5, 1.5]])))
