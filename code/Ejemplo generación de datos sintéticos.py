from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_hastie_10_2
import numpy as np
import matplotlib.pyplot as plt

# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

######################################################################
# Generate synthetic data of blobs (clusters)
centers = [[0, 0], [1, 1]]
std_devs = [0.3, 0.3]
X, y = make_blobs(n_samples=1000, centers=centers, cluster_std=std_devs, n_features=2, random_state=42)
ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
ax.set_title("Blobs")
ax.set_xlabel("$X_1$")
ax.set_ylabel("$X_2$")
ax.set_aspect('equal', 'box')
ax.set_xlim(-1.1, 2.1)
ax.set_ylim(-1.1, 2.1)

######################################################################
# Generate synthetic data of concentric circles
X, y = make_circles(n_samples=1000, noise=0.15, factor=0.5, random_state=42)
ax = axes[1]
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
ax.set_title("Circles")
ax.set_xlabel("$X_1$")
ax.set_ylabel("$X_2$")
ax.set_aspect('equal', 'box')
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.6, 1.6)

######################################################################
# Generate synthetic data of moons
X, y = make_moons(n_samples=1000, noise=0.15, random_state=42)
ax = axes[2]
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
ax.set_title("Moons")
ax.set_xlabel("$X_1$")
ax.set_ylabel("$X_2$")
ax.set_aspect('equal', 'box')
ax.set_xlim(-1.6, 2.6)
ax.set_ylim(-1.6, 2.6)

plt.show()
