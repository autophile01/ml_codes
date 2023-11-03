#****** ASSIGNMENT 5 *******
# Dataset : sales_data_sample.csv
# Dataset : diabetes.csv (Curicullam)
# Implement K-Nearest Neighbors algorithm. 
# Implement K-Nearest Neighbors algorithm on diabetes.csv dataset. Compute confusion matrix, accuracy, error rate, precision and recall on the given dataset.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import preprocessing

df = pd.read_csv("P:\\ML PRACTICAL\\datasets\\diabetes.csv")
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

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a StandardScaler and scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def knn(X_train, X_test, y_train, y_test, neighbors, power):
    model = KNeighborsClassifier(n_neighbors=neighbors, p=power)
    
    # Fit and predict with the model
    y_pred = model.fit(X_train, y_train).predict(X_test)
    
    # Calculate accuracy and print it
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for K-Nearest Neighbors model: {accuracy}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Print confusion matrix in a readable format
    print(f'''Confusion matrix:
 | Positive Prediction\t| Negative Prediction
 ---------------+------------------------+----------------------
 Positive Class | True Positive (TP): {cm[0, 0]}\t| False Negative (FN): {cm[0, 1]}
 ---------------+------------------------+----------------------
 Negative Class | False Positive (FP): {cm[1, 0]}\t| True Negative (TN): {cm[1, 1]}\n''')
    
    # Generate and print the classification report
    cr = classification_report(y_test, y_pred)
    print('Classification report:\n', cr)

param_grid = {
 'n_neighbors': range(1, 51),
 'p': range(1, 4)
}
grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
grid.best_estimator_, grid.best_params_, grid.best_score_

knn(X_train, X_test, y_train, y_test, grid.best_params_['n_neighbors'], grid.best_params_['p'])





