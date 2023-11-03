#******** ASSIGNMENT 3 *********
# Dataset : Churn_Modelling.csv
# Given a bank customer, build a neural networkbased classifier that can determine whether they will leave or not in the next 6 months.

# IMPORTING PYTHON MODULES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import preprocessing

# LOADING THE DATASET : First we load the dataset and find out the number of columns, rows, NULL values etc.
df = pd.read_csv("P:\ML PRACTICAL\datasets\Churn_Modelling.csv")
df.info()

df.head()

# CLEANING
df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True)
df.isna().sum()

df.describe()

# SEPERATING THE FEATURES & LABELS
X=df.iloc[:, :df.shape[1]-1].values #Independent Variables
y=df.iloc[:, -1].values #Dependent Variable
X.shape, y.shape

# ENCODING CATEGORICAL (STRING BASED) DATA
print(X[:8,1], '... will now become: ')
label_X_country_encoder = LabelEncoder()
X[:,1] = label_X_country_encoder.fit_transform(X[:,1])
print(X[:8,1])

print(X[:6,2], '... will now become: ')
label_X_gender_encoder = LabelEncoder()
X[:,2] = label_X_gender_encoder.fit_transform(X[:,2])
print(X[:6,2])

# SPLIT THE COUNTRIES INTO RESPECTIVE DIMENSIONS. CONVERTING THE STRING FEATURES INTO THEIR OWN DIMENSIONS.
transform = ColumnTransformer([("countries", OneHotEncoder(), [1])], remainder="passthrough") #
X = transform.fit_transform(X)
X

# DIMENSIONALITY REDUCTION : A 0 ON TWO COUNTRIES MEANS THAT THE COUNTRY HAS TO BE ONE VARIABLE WHICH WASNT INCLUDED
X = X[:,1:]
X.shape

# SPLITTING THE DATASET : Training & Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# NORMALISE THE TRAIN AND TEST DATA 
# ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
sc=StandardScaler()
X_train[:,np.array([2,4,5,6,7,10])] = sc.fit_transform(X_train[:,np.array([2,4,5,6,7,10])])
X_test[:,np.array([2,4,5,6,7,10])] = sc.transform(X_test[:,np.array([2,4,5,6,7,10])])

sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train

# INITIALISE AND BUILD THE MODEL
from tensorflow.keras.models import Sequential
# Initializing the ANN
classifier = Sequential()

from tensorflow.keras.layers import Dense
# Create a Sequential model
classifier = Sequential()
# Add a Dense layer to the model
classifier.add(Dense(units=256, activation='relu', input_dim=11, kernel_initializer='uniform'))

# Adding the hidden layer
classifier.add(Dense(activation = 'relu', units=512, kernel_initializer='uniform'))
classifier.add(Dense(activation = 'relu', units=256, kernel_initializer='uniform'))
classifier.add(Dense(activation = 'relu', units=128, kernel_initializer='uniform'))

# Adding the output layer
# Notice that we do not need to specify input dim.
# we have an output of 1 node, which is the the desired dimensions of our output (stay with the
# We use the sigmoid because we want probability outcomes
classifier.add(Dense(activation = 'sigmoid', units=1, kernel_initializer='uniform'))

# Create optimizer with default learning rate
# sgd_optimizer = tf.keras.optimizers.SGD()
# Compile the model
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.summary()

classifier.fit(
 X_train, y_train,
 validation_data=(X_test,y_test),
 epochs=20,
 batch_size=32
)

# PREDICT THE RESULTS USING 0.5 AS A THRESHOLD
y_pred = classifier.predict(X_test)
y_pred


# To use the confusion Matrix, we need to convert the probabilities that a customer will leave t
# So we will use the cutoff value 0.5 to indicate whether they are likely to exit or not.
y_pred = (y_pred > 0.5)
y_pred


# PRINT THE ACCURACY SCORE AND CONFUSION MATRIX
from sklearn.metrics import confusion_matrix,classification_report
cm1 = confusion_matrix(y_test, y_pred)
cm1

print(classification_report(y_test, y_pred))

accuracy_model1 = ((cm1[0][0]+cm1[1][1])*100)/(cm1[0][0]+cm1[1][1]+cm1[0][1]+cm1[1][0])
print (accuracy_model1, '% of testing data was classified correctly')


