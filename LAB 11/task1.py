# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Loading the blobs data set
data, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Performing K-means++ clustering with three clusters
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(data)

# Making a figure of distortion vs number of clusters using the elbow method
distortions = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(data)
    distortions.append(kmeans.inertia_)

# Plotting the elbow curve
plt.plot(range(1, 11), distortions, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.show()

# Step 4: Using the elbow method to find the optimal number of clusters
# Getting the elbow point from the the plot (when the distortion starts to decrease at a slower rate)
# optimal number of clusters (this is where the elbow occurs)

# using a more programmatic approach to find the elbow point
distances = pairwise_distances_argmin_min(np.array(range(1, 11)).reshape(-1, 1), np.array(distortions).reshape(-1, 1))
optimal_clusters = distances[0][np.argmin(distances[1])]

print(f"The optimal number of clusters is {optimal_clusters}")
