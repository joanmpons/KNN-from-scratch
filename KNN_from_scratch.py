#!/usr/bin/env python
# coding: utf-8

# In[237]:


import numpy as np

class KNN:
    def __init__(self, k, max_iterations = 100):
        self.og_data = data
        self.n_clusters = k
        self.max_iter = max_iterations
        self.centroids = None
        self.cluster_label = None
        
    def fit_transform(self, data):
        #Initialization
        self.centroids = data[np.random.choice(data.shape[0], self.n_clusters)]
        #Iterative step
        for i in range(self.max_iter):
            centroid_dist = np.zeros([data.shape[0], self.n_clusters])
            #Calculating centroid distances
            for i,j in enumerate(data):
                centroid_dist[i] = np.linalg.norm(j - self.centroids, axis=1)
            #Assigning clusters to data   
            self.cluster_label = np.argmin(centroid_dist, axis = 1)
            #Updating centroids
            self.centroids = np.zeros([self.n_clusters, data.shape[1]])
            for i in range(self.n_clusters):
                if data[cluster_label == i].size == 0:
                    self.centroids[i] = np.array(0)
                else:
                    self.centroids[i] = np.nanmean(data[cluster_label == i],axis=0)
        
        


# In[238]:


from sklearn.datasets import make_blobs
import pandas as pd

data, _ = make_blobs(n_samples=300, centers=3, random_state=41)

knn = KNN(3)
knn.fit_transform(data)

df = pd.DataFrame(knn.og_data,columns = ["a","b"])
df["clusts"] = knn.cluster_label
df.plot.scatter(x="a",y="b",c="clusts",colormap='viridis')

