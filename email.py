#******* ASSIGNMENT 2 **********
# Dataset : emails.csv
# Classify the email using the binary classification method. Email Spam detection has two
# states: a) Normal State – Not Spam, b) Abnormal State – Spam. Use K-Nearest Neighbors and
# Support Vector Machine for classification. Analyze their performance.

# IMPORTING PYTHON MODULES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import preprocessing

# LOADING THE DATASET : First we load the dataset and find out the number of columns, rows, NULL values etc.
df = pd.read_csv("P:\\ML PRACTICAL\\datasets\\emails.csv")
df.info()

df.head()

df.dtypes()

# CLEANING
df.drop(columns=['Email No.'], inplace=True)
df.isna().sum()

df.describe()

# SEPERATING THE FEATURES AND THE LABELS
X=df.iloc[:, :df.shape[1]-1] #Independent Variables
y=df.iloc[:, -1] #Dependent Variable
X.shape, y.shape

# SPLITTING THE DATASET
# Split the data into training and testing sets with a test size of 15% and a random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

models = {
"K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=2),
"Linear SVM":LinearSVC(random_state=8, max_iter=900000),
"Polynomical SVM":SVC(kernel="poly", degree=2, random_state=8),
"RBF SVM":SVC(kernel="rbf", random_state=8),
"Sigmoid SVM":SVC(kernel="sigmoid", random_state=8)
}

# FIT AND PREDICT ON EACH MODEL : Each model is trained using the train set and predictions are made based on the test set. Accuracy scores are calculated for each model.
from sklearn.metrics import accuracy_score

for model_name, model in models.items():
    y_pred = model.fit(X_train, y_train).predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {model_name} model: {accuracy}")

