# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 13:06:42 2015

@author: mcnear1
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.feature_extraction import DictVectorizer

"""Exercise 1 - K Means"""

print("========================Exercise 1 - K Means========================")

# Creating artificial data
X, y = make_blobs(n_samples=300, centers=3,
                  random_state=0, cluster_std=0.60)

# Displaying what the artificial data looks like
plt.title("Artificial data")
plt.scatter(X[:, 0], X[:, 1], s=50);
plt.show()

# Creating k means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=8)
# Creating k means that fits the data and getting the labels for each point
y_hat = kmeans.fit(X).labels_

plt.title("Clustering after using k-Means")
plt.scatter(X[:, 0], X[:, 1], c=y_hat)
plt.show()

"""Exercise 2 - K-Nearest neighbors"""