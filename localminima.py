#******* ASSIGNMENT 4 **********
# Dataset : diabetes.csv
# Implement Gradient Descent Algorithm to find the local minima of a function

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import preprocessing

df = pd.read_csv("C:\\Users\\vaishnavi\\\OneDrive\\Desktop\\diabetes.csv")

df.info()

df.head()

df.corr().style.background_gradient(cmap='BuGn')

df.drop(['BloodPressure', 'SkinThickness'], axis=1, inplace=True)

df.isna().sum()

df.describe()

hist = df.hist(figsize=(20,16))

X=df.iloc[:, :df.shape[1]-1] #Independent Variables
y=df.iloc[:, -1] #Dependent Variable
X.shape, y.shape


# Split the data into training and testing sets with a test size of 0.2 and a random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def knn(X_train, X_test, y_train, y_test, neighbors, power):
    model = KNeighborsClassifier(n_neighbors=neighbors, p=power)
    
    # Fit the model using the training set and make predictions on the test set
    y_pred = model.fit(X_train, y_train).predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for K-Nearest Neighbors model: {accuracy:.2f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f'''Confusion matrix :
 | Positive Prediction | Negative Prediction
 ---------------+------------------------+----------------------
 Positive Class | True Positive (TP) {cm[0, 0]} | False Negative (FN) {cm[0, 1]}
 Negative Class | False Positive (FP) {cm[1, 0]} | True Negative (TN) {cm[1, 1]}
 ---------------+------------------------+----------------------
''')
    
    cr = classification_report(y_test, y_pred)
    print('Classification report : \n', cr)

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_neighbors': range(1, 51),
    'p': range(1, 4)
}

grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

best_estimator = grid.best_estimator_
best_params = grid.best_params_
best_score = grid.best_score_

print("Best Estimator:", best_estimator)
print("Best Parameters:", best_params)
print("Best Score:", best_score)

# Assuming you already have the best parameters from the grid search
best_n_neighbors = grid.best_params_['n_neighbors']
best_p = grid.best_params_['p']

knn(X_train, X_test, y_train, y_test, best_n_neighbors, best_p)


