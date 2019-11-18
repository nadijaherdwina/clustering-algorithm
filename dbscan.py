import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import completeness_score

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

class DBSCANClustering:
    def __init__(self, epsilon = 0.6, minPts = 4):
        self.epsilon = epsilon
        self.minPts = minPts
        self.distanceMatrix = []
        self.distanceMatrixMember = []
        self.clusterList = []
        self.X = []
        self.Xchecked = []
        self.Xcluster = []

    def euclidean(self, a, b):
        a = np.array(a)
        b = np.array(b)
        dist = np.linalg.norm(a-b, ord=2)
        return dist
    
    def initDistanceMatrix(self, X):
        for i in range (0, len(X)):
            distanceRow = []
            for j in range (0, len(X)):
                dist = self.euclidean(np.array(X[i]), np.array(X[j]))
                distanceRow.append(dist)
            self.distanceMatrix.append(distanceRow)
            self.distanceMatrixMember.append([i])
        self.clusterList.append(self.distanceMatrixMember[:])
    
    def pointsCompleted(self, X): 
        for i in range(135):
            if (self.Xchecked[i] == 0):
                return False
        return True
    
    def findNeighborsRecursive(self, i, X):
        neighbors = []
        for j in range(135):
            if (self.distanceMatrix[i][j] < self.epsilon) and (self.Xcluster[j] == -1) and (self.Xchecked[j] == 0):
                neighbors.append(X[j])
        return neighbors
    
    def findNeighbors(self, i, X):
        neighbors = []
        for j in range(135):
            if (self.distanceMatrix[i][j] < 0.5):
                neighbors.append(X[j])
        return neighbors

    def fit(self, X, i, neighbors, clusterCount):
        self.Xchecked[i] = 1
        if (len(neighbors) == 0):
            self.Xcluster[i] = -1
        else:
            if (len(self.findNeighbors(i, X)) >= self.minPts):
                self.Xcluster[i] = clusterCount
            for j in range (len(neighbors)):
                if (self.Xcluster[j] == -1) and (self.Xchecked[j] == 0):
                    self.fit(X, j, self.findNeighborsRecursive(i, X), clusterCount)        

def readData():
    dataset = pd.read_csv('iris.data', names=["1", "2", "3", "4", "label"])
    df = pd.DataFrame(dataset)
    X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=0.1, shuffle = False, stratify = None)


    dicti = {"Iris-setosa": 0, "Iris-versicolor":1, "Iris-virginica":2}
    y_train = y_train.apply(lambda y: dicti[y])
    y_test = y_test.apply(lambda y: dicti[y])

    return np.array(X_train), np.array(X_test), y_train, y_test
 
# ------- DBSCAN NO LIBRARY -------
def dbscanNoLibrary():
    X_train, X_test, y_train, y_test = readData()

    dbscan = DBSCANClustering(1, 4)
    dbscan.initDistanceMatrix(X_train)
    
    for k in range(135):
        dbscan.Xchecked.append(0)
        dbscan.Xcluster.append(-1)
    
    completed = False
    clusterCount = 0

    # Iterate through points
    for i in range(135):
        if (dbscan.Xchecked[i] == 0):
            dbscan.Xchecked[i] = 1

            # Find neighbors
            neighbors = dbscan.findNeighbors(i, X_train)

            # Cluster it (running recursively)
            dbscan.fit(X_train, i, neighbors, clusterCount)
            clusterCount += 1
            
            p = dbscan.pointsCompleted(X_train)
    
    # ---- RESULT ----
    labelsCluster = []
    for i in range(135):
        labelsCluster.append(dbscan.Xcluster[i])
    
    labels = np.array(labelsCluster)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[134] = True

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % homogeneity_score(y_train, labels))
    print("Completeness: %0.3f" % completeness_score(y_train, labels))

    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 1, 0, 1]

        class_member_mask = (labels == k)

        xy = X_train[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=5)

        xy = X_train[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=5)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

# ------- DBSCAN CLUSTERING WITH SKLEARN LIBRARY -------
def dbscanLibrarySklearn():
    X_train, X_test, labels_true, y_test = readData()

    # Compute DBSCAN
    db = DBSCAN(eps=0.6, min_samples=4).fit(X_train)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % completeness_score(labels_true, labels))

    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X_train[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=5)

        xy = X_train[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=5)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


if __name__ == "__main__":
    print("Train Data with Created DBSCAN Algorithm")
    dbscanNoLibrary()

    print("Train Data with Sklearn DBSCAN Algorithm")
    dbscanLibrarySklearn()