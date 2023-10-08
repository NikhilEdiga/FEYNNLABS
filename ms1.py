import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load your EV market dataset
data = pd.read_csv('Electric_Vehicle_Population_Data.csv')

# Select relevant columns for segmentation
selected_columns = ['Electric Range', 'Base MSRP']

# Preprocess your data (handle missing values if needed)
# You may also need to convert some categorical data to numerical (e.g., Clean Alternative Fuel Vehicle Eligibility).

# Select the features for clustering
X = data[selected_columns].values

# Determine the number of clusters (K) using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Choose the optimal K based on the Elbow Method (e.g., K = 3)

# Apply K-Means clustering with the selected K
k = 3
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
data['Cluster'] = kmeans.fit_predict(X)

# Scatterplot for Electric Range vs. Base MSRP
plt.figure(figsize=(8, 6))
for i in range(k):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data[selected_columns[0]], cluster_data[selected_columns[1]], label=f'Cluster {i}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroids')
plt.title('Scatterplot of Clusters')
plt.xlabel(selected_columns[0])
plt.ylabel(selected_columns[1])
plt.legend()
plt.show()
