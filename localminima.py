#******* ASSIGNMENT 4 **********
# Dataset : diabetes.csv
# Implement Gradient Descent Algorithm to find the local minima of a function

# IMPORTING PYTHON MODULES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import preprocessing

# LOADING THE DATASET
df = pd.read_csv("P:\ML PRACTICAL\datasets\diabetes.csv")
df.info()

df.head()

# CLEANING
df.corr().style.background_gradient(cmap='BuGn')

df.drop(['BloodPressure', 'SkinThickness'], axis=1, inplace=True)
df.isna().sum()

df.describe()

# VISUALISATION
hist = df.hist(figsize=(20,16))

# SEPEARTING THE FEATURES AND THE LABELS
X=df.iloc[:, :df.shape[1]-1] #Independent Variables
y=df.iloc[:, -1] #Dependent Variable
X.shape, y.shape

# SPLITTING THE DATASET : Training & Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MACHINE LEARNING MODEL
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


# HYPERPARAMETER TUNING
param_grid = {
 'n_neighbors': range(1, 51),
 'p': range(1, 4)
}
grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
grid.best_estimator_, grid.best_params_, grid.best_score_


knn(X_train, X_test, y_train, y_test, grid.best_params_['n_neighbors'], grid.best_params_['p'])

# GRADIENT DESCENT BEGINS
from sympy import Symbol, lambdify
import matplotlib.pyplot as plt
import numpy as np

x = Symbol('x')

def gradient_descent(
 function, start, learn_rate, n_iter=10000, tolerance=1e-06, step_size=1
):
 gradient = lambdify(x, function.diff(x))
 function = lambdify(x, function)
 points = [start]
 iters = 0 #iteration counter

 while step_size > tolerance and iters < n_iter:
  prev_x = start #Store current x value in prev_x
 start = start - learn_rate * gradient(prev_x) #Grad descent
 step_size = abs(start - prev_x) #Change in x
 iters = iters+1 #iteration count
 points.append(start)
 print("The local minimum occurs at", start)

 # Create plotting array
 x_ = np.linspace(-7,5,100)
 y = function(x_)
 # setting the axes at the centre
 fig = plt.figure(figsize = (10, 10))
 ax = fig.add_subplot(1, 1, 1)
 ax.spines['left'].set_position('center')
 ax.spines['bottom'].set_position('zero')
 ax.spines['right'].set_color('none')
 ax.spines['top'].set_color('none')
 ax.xaxis.set_ticks_position('bottom')
 ax.yaxis.set_ticks_position('left')
 # plot the function
 plt.plot(x_,y, 'r')
 plt.plot(points, function(np.array(points)), '-o')
 # show the plot
 plt.show()

def gradient_descent(function, start, learn_rate, n_iter):
    # Implementation of gradient descent goes here
    pass

# Example usage
function = lambda x: (x + 5) ** 2
start = 3.0
learn_rate = 0.2
n_iter = 50

gradient_descent(function, start, learn_rate, n_iter)




