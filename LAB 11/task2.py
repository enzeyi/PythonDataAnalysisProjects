import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import DBSCAN, KMeans

# clustering by using DBSCAN with the blobs data set
blobs_data, _ = make_blobs(n_samples=300, centers=3, random_state=42)

dbscan_blobs = DBSCAN(eps=0.5, min_samples=5)
blobs_labels = dbscan_blobs.fit_predict(blobs_data)

plt.scatter(blobs_data[:, 0], blobs_data[:, 1], c=blobs_labels, cmap='viridis')
plt.title('DBSCAN Clustering - Blobs Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# clustering by using DBSCAN with the moon data set
moon_data, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

dbscan_moon = DBSCAN(eps=0.3, min_samples=5)
moon_labels = dbscan_moon.fit_predict(moon_data)

plt.scatter(moon_data[:, 0], moon_data[:, 1], c=moon_labels, cmap='viridis')
plt.title('DBSCAN Clustering - Moon Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# K-means++ clustering with the moon data set
kmeans_moon = KMeans(n_clusters=2, init='k-means++', random_state=42)
moon_kmeans_labels = kmeans_moon.fit_predict(moon_data)

plt.scatter(moon_data[:, 0], moon_data[:, 1], c=moon_kmeans_labels, cmap='viridis')
plt.title('K-means++ Clustering - Moon Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
