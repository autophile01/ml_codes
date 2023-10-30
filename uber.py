#******** ASSIGNMENT 1 **********
# Dataset : uber.csv
# Predict the price of the Uber ride from a given pickup
# point to the agreed drop-off location. Perform
# following tasks:
# 1. Pre-process the dataset.
# 2. Identify outliers.
# 3. Check the correlation.
# 4. Implement linear regression and random forest
# regression models.
# Evaluate the models and compare their respective
# scores like R2, RMSE, etc. Dataset link:
# https://www.kaggle.com/datasets/yasserh/uber-faresdataset

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pylab
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import preprocessing

# Load the dataset
df = pd.read_csv("C:\\Users\\vaishnavi\\OneDrive\\Desktop\\uber.csv")

# Display basic information about the dataset
df.info()

# Display the first few rows of the dataset
df.head()

# Remove unnecessary columns
df = df.drop(['Unnamed: 0', 'key'], axis=1)

# Check and remove rows with missing values
df.dropna(axis=0, inplace=True)

# Define a function to calculate the distance between two sets of coordinates
def distance_transform(longitude1, latitude1, longitude2, latitude2):
    long1, lati1, long2, lati2 = map(np.radians, [longitude1, latitude1, longitude2, latitude2])
    dist_long = long2 - long1
    dist_lati = lati2 - lati1
    a = np.sin(dist_lati/2)**2 + np.cos(lati1) * np.cos(lati2) * np.sin(dist_long/2)
    c = 2 * np.arcsin(np.sqrt(a)) * 6371
    return c

# Calculate the distance and add it as a new column in the DataFrame
df['Distance'] = distance_transform(
    df['pickup_longitude'],
    df['pickup_latitude'],
    df['dropoff_longitude'],
    df['dropoff_latitude']
)

# Remove outliers in the 'Distance' and 'fare_amount' columns
df.drop(df[df['Distance'] >= 60].index, inplace=True)
df.drop(df[df['fare_amount'] <= 0].index, inplace=True)
df.drop(df[(df['fare_amount'] > 100) & (df['Distance'] < 1)].index, inplace=True)
df.drop(df[(df['fare_amount'] < 100) & (df['Distance'] > 100)].index, inplace=True)

# Split the data into independent (X) and dependent (y) variables
X = df['Distance'].values.reshape(-1, 1)  # Independent Variable
y = df['fare_amount'].values.reshape(-1, 1)  # Dependent Variable

# Standardize the data
std = StandardScaler()
y_std = std.fit_transform(y)
x_std = std.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x_std, y_std, test_size=0.2)

# Train a Linear Regression model
from sklearn.linear_model import LinearRegression
l_reg = LinearRegression()
l_reg.fit(X_train, y_train)

# Evaluate the Linear Regression model
print("Training set score: {:.2f}".format(l_reg.score(X_train, y_train)))
print("Test set score: {:.7f".format(l_reg.score(X_test, y_test)))

y_pred = l_reg.predict(X_test)

# Calculate and print various regression metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Absolute % Error:', metrics.mean_absolute_percentage_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R Squared (R²):', metrics.r2_score(y_test, y_pred))

# Plot the regression results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, l_reg.predict(X_train), color="blue")
plt.title("Fare vs Distance (Training Set)")
plt.ylabel("fare_amount")
plt.xlabel("Distance")
plt.subplot(2, 2, 2)
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, l_reg.predict(X_train), color="blue")
plt.ylabel("fare_amount")
plt.xlabel("Distance")
plt.title("Fare vs Distance (Test Set)")
plt.tight_layout()
plt.show()

# Create a DataFrame to store the regression results
cols = ['Model', 'RMSE', 'R-Squared']
result_tabulation = pd.DataFrame(columns=cols)

# Compile the results for Linear Regression
linreg_metrics = pd.DataFrame([[
    "Linear Regression model",
    np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
    np.sqrt(metrics.r2_score(y_test, y_pred))
]], columns=cols)
result_tabulation = pd.concat([result_tabulation, linreg_metrics], ignore_index=True)

# Train a Random Forest Regressor model
rf_reg = RandomForestRegressor(n_estimators=100, random_state=10)
rf_reg.fit(X_train, y_train)

# Predict with the Random Forest model
y_pred_RF = rf_reg.predict(X_test)

# Calculate and print various regression metrics for Random Forest
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_RF))
print('Mean Absolute % Error:', metrics.mean_absolute_percentage_error(y_test, y_pred_RF))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_RF))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_RF)))
print('R Squared (R²):', metrics.r2_score(y_test, y_pred_RF))

# Create a scatter plot to visualize the Random Forest results
plt.scatter(X_test, y_test, c='b', alpha=0.5, marker='.', label='Real')
plt.scatter(X_test, y_pred_RF, c='r', alpha=0.5, marker='.', label='Predicted')
plt.xlabel('Distance')
plt.ylabel('fare_amount')
plt.grid(color='#D3D3D3', linestyle='solid')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Compile the results for Random Forest
random_forest_metrics = pd.DataFrame([[
    "Random Forest Regressor model",
    np.sqrt(metrics.mean_squared_error(y_test, y_pred_RF)),
    np.sqrt(metrics.r2_score(y_test, y_pred_RF))
]], columns=cols)
result_tabulation = pd.concat([result_tabulation, random_forest_metrics], ignore_index=True)

# Display the tabulated results
print(result_tabulation)
