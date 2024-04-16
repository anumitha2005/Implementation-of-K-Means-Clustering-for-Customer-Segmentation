# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary packages using import statement.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Import KMeans and use for loop to cluster the data.

4.Predict the cluster and plot data graphs.

5.Print the outputs and end the program

## Program:
```
``
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: M. R. ANUMITHA
RegisterNumber:  212223040018
*/
```
```
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
data = pd.read_csv('/content/Mall_Customers_EX8.csv')
data
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
X
plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
#Number of clusters
k = 5

#Initialize KMeans
kmeans = KMeans(n_clusters=k)

#Fit the data
kmeans.fit(X)
centroids = kmeans.cluster_centers_

labels = kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors = ['r', 'g', 'b', 'c', 'm']
for i in range(k):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points['Annual Income (k$)'], cluster_points['Spending Score (1-100)'], color=colors[i], label=f'Cluster {i+1}')
    distances = euclidean_distances(cluster_points, [centroids[i]])
    radius = np.max(distances)
    circle = plt.Circle(centroids[i], radius, color=colors[i], fill=False)
    plt.gca().add_patch(circle)

plt.scatter(centroids[:, 0], centroids[:,1], marker='*', s=200, color='k', label='Centroids')

plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
```
``
## Output:
![K Means Clustering for Customer Segmentation](sam.png)


![Screenshot 2024-04-16 105605](https://github.com/anumitha2005/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/155522855/4d30ccc8-e2d7-4dd3-9c6b-900d063d568a)

![Screenshot 2024-04-16 113741](https://github.com/anumitha2005/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/155522855/0c14db24-1d89-4451-b17a-ff421a3ec6d1)

![Screenshot 2024-04-16 113752](https://github.com/anumitha2005/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/155522855/cfe30b05-da63-4603-bad8-0f19e6c8ccc5)

![Screenshot 2024-04-16 113801](https://github.com/anumitha2005/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/155522855/ba87a526-a6ad-4118-8644-dae387d7ab08)

![Screenshot 2024-04-16 113815](https://github.com/anumitha2005/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/155522855/1f89cd0e-975a-4ee7-9238-6c7386b725a0)

![Screenshot 2024-04-16 113837](https://github.com/anumitha2005/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/155522855/651a2561-8770-43cb-883b-69f14a2c4c26)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
