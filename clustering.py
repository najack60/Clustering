#-------------------------------------------------------------------------
# AUTHOR: Nate COlbert
# FILENAME: clustering.py
# SPECIFICATION: Implements the clustering algorithm with a test data set.
# FOR: CS 4210- Assignment #5
# TIME SPENT: 3-4 hours
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = np.array(df.values)[:,:64]


#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code
sill = np.array([])
sillA = []
bestK = 0
bestS = 0
for k in range(2, 21):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)


   # sill[int(x)] > bestS:
     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
    sill = np.append(sill, silhouette_score(X_training, kmeans.labels_))

sillA = sill
for x in range(19):
    if x == 0:
        bestS = sill[0]
    if sillA[x] > bestS:
        bestS = sillA[x]
        bestK = x+2

#print(bestS, "----", bestK)

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here
x = np.array([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = sill
 
# plotting
plt.title("Best k")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.plot(x, y, color ="black")
#plt.show()

#reading the validation data (clusters) by using Pandas library
#--> add your Python code here
df = pd.read_csv('testing_data.csv', sep=',', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
labels = np.array(df.values).reshape(1,len(df))[0]

#Calculate and print the Homogeneity of this kmeans clustering
#^^print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())

#run agglomerative clustering now by using the best value o k calculated before by kmeans
#Do it:
agg = AgglomerativeClustering(n_clusters=bestK, linkage='ward')
agg.fit(X_training)

#Calculate and print the Homogeneity of this agglomerative clustering
print("Agglomerative Clustering Homogeneity Score = " + metrics.homogeneity_score(labels, agg.labels_).__str__())

#plot graph
plt.show()