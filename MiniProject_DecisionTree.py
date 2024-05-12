import numpy as np
from Decision_Tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define the file path relative to the current directory
file_path = "banknote_data/Dataset.txt"

# Load the data from the text file
data = np.loadtxt(file_path, delimiter=',')
# print(data.size)
# print(data)

Input_data = data[:,0:4]
Output_data = data[:,4]
print(Input_data)
print(Output_data)

X_train, X_test, y_train, y_test = train_test_split(Input_data, Output_data, test_size=0.5, random_state=42)

tree = DecisionTreeClassifier(max_depth=3)
tree.fit(X_train, y_train)

# Predictions for new instances
y_pred = tree.predict(X_test)
print('Predictions: ', y_pred)

acc_score = accuracy_score(y_test, y_pred)
print('Accuracy score: ', acc_score)
