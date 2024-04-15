import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.decomposition import KernelPCA

# Generating the moon dataset
moon_data, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

# clustering with DBSCAN
dbscan_moon = DBSCAN(eps=0.3, min_samples=5)
moon_labels = dbscan_moon.fit_predict(moon_data)

# Visualization using Kernel PCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
moon_transformed = kpca.fit_transform(moon_data)

# clusters plotting after Kernel PCA
plt.scatter(moon_transformed[:, 0], moon_transformed[:, 1], c=moon_labels, cmap='viridis')
plt.title('DBSCAN Clustering with Kernel PCA - Moon Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
