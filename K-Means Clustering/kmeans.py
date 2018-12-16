# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 14:36:36 2018

@author: Deathrow77
"""
# Importing the Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values

# Finding the optimal number of clusters using wcss values
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    km = KMeans(n_clusters=i, init='k-means++', random_state=0)
    km.fit(X)
    # print(km.inertia_)
    wcss.append(km.inertia_)
plt.plot(range(1,11), wcss)
plt.show()

# Using the optimal number of clusters, predicting values
km = KMeans(n_clusters=5, init='k-means++', random_state=0)
y_pred = km.fit_predict(X)

# Scatter Plot for the Clusters
plt.scatter(X[y_pred==0,0], X[y_pred==0,1], color='red', label='Cluster 1')
plt.scatter(X[y_pred==1,0], X[y_pred==1,1], color='blue', label='Cluster 2')
plt.scatter(X[y_pred==2,0], X[y_pred==2,1], color='black', label='Cluster 3')
plt.scatter(X[y_pred==3,0], X[y_pred==3,1], color='green', label='Cluster 4')
plt.scatter(X[y_pred==4,0], X[y_pred==4,1], color='magenta', label='Cluster 5')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:,1], s=100, color='yellow', label='Centroid')
plt.title('KMeans Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.show()