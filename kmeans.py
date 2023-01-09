#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
    Practical example of a k-means algorithm.

    Inspired by
    https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#%matplotlib inline

import logging

logging.basicConfig(level=logging.INFO)

# Generate random data (2D)
num_points = 100
n_clusters = 2
logging.info('Number of data points: '+str(n_clusters)+' * '+str(num_points))
X = -2 * np.random.rand(num_points,2)
X1 = 1 + 2 * np.random.rand(int(num_points/2),2)
X[50:100, :] = X1

# Perform k-means
Kmean = KMeans(n_clusters=2)
Kmean.fit(X) # Compute k-means clustering.
kmean_params      = Kmean.get_params(deep=True)
a_cluster_centers = Kmean.cluster_centers_
a_labels          = Kmean.labels_
logging.info('Kmeans parameters: '+str(kmean_params))
logging.info('Kmeans cluster centers: '+str(a_cluster_centers))
logging.info('Kmeans cluster labels : #'+str(len(a_labels))+' '+str(a_labels))

# plot it
#print(X.shape)
#print(X)
plt.scatter(X[ : , 0], X[ : , 1], s = 50, c = 'b')
plt.scatter(a_cluster_centers[0, 0], a_cluster_centers[0, 1], s=200, c='g', marker='s')
plt.scatter(a_cluster_centers[1, 0], a_cluster_centers[1, 1], s=200, c='r', marker='s')
plt.show()