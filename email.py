#******* ASSIGNMENT 2 **********
# Dataset : emails.csv
# Classify the email using the binary classification method. Email Spam detection has two
# states: a) Normal State – Not Spam, b) Abnormal State – Spam. Use K-Nearest Neighbors and
# Support Vector Machine for classification. Analyze their performance.


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import preprocessing

df = pd.read_csv("C:\\Users\\vaishnavi\\\OneDrive\\Desktop\\emails.csv")

df.info()

df.head()

df.dtypes()

df.drop(columns=['Email No.'], inplace=True)

df.isna().sum()

df.describe()

X=df.iloc[:, :df.shape[1]-1] #Independent Variables
y=df.iloc[:, -1] #Dependent Variable
X.shape, y.shape

# Split the data into training and testing sets with a test size of 15% and a random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

models = {
 "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=2),
 "Linear SVM":LinearSVC(random_state=8, max_iter=900000),
 "Polynomical SVM":SVC(kernel="poly", degree=2, random_state=8),
 "RBF SVM":SVC(kernel="rbf", random_state=8),
 "Sigmoid SVM":SVC(kernel="sigmoid", random_state=8)
}

from sklearn.metrics import accuracy_score

for model_name, model in models.items():
    y_pred = model.fit(X_train, y_train).predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {model_name} model: {accuracy}")

