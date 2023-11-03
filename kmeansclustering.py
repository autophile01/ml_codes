#******* ASSIGNMENT 6 **********
# Dataset : sales_data_sample.csv
# Implement K-Means clustering/ hierarchical clustering on sales_data_sample.csv dataset. Determine the number of clusters using the elbow method.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

# Try reading the CSV file with multiple encodings
encodings = ['utf-8', 'latin1', 'ISO-8859-1']

for encoding in encodings:
    try:
        df = pd.read_csv("P:\\ML PRACTICAL\\datasets\\sales_data_sample.csv", encoding=encoding)
        break  # Break the loop if successful
    except UnicodeDecodeError:
        continue  # Try the next encoding if there's an error

# Display the head of the DataFrame
df.head()

df.info()

df.describe()

fig = plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, fmt='.2f')
plt.show()

df= df[['PRICEEACH', 'MSRP']]
df.head()

df.isna().any()

df.describe().T

df.shape

from sklearn.cluster import KMeans
inertia = []
for i in range(1, 11):
 clusters = KMeans(n_clusters=i, init='k-means++', random_state=42)
 clusters.fit(df)
 inertia.append(clusters.inertia_)

plt.figure(figsize=(6, 6))
sns.lineplot(x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], y = inertia)

kmeans = KMeans(n_clusters = 3, random_state = 42)
y_kmeans = kmeans.fit_predict(df)
y_kmeans

plt.figure(figsize=(8, 8))
sns.scatterplot(x=df['PRICEEACH'], y=df['MSRP'], hue=y_kmeans)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red')  # Add missing closing parenthesis
plt.legend()
plt.show()  # Add this line to display the plot

kmeans.cluster_centers_


