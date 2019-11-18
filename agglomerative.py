import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import AgglomerativeClustering

class AggloClustering:
	def __init__(self, nb_cluster=5, linkage="average-group"):
		self.nb_cluster = nb_cluster
		self.linkage = linkage
		self.distanceMatrix = []
		self.distanceMatrixChanged = []
		self.distanceMatrixMember = []
		self.clusterList = []
		self.selectedClusterList = []
		self.allInOneCluster = False
		self.dataLength = 0
		self.X = []

	def euclidean(self, a, b):
		a = np.array(a)
		b = np.array(b)
		dist = np.linalg.norm(a-b, ord=2)
		return dist

	def manhattan(self, a, b):
		dist = 0
		for i in range(0, len(a)):
			dist += abs(b[i] - a[i])
		return dist

	def initDistanceMatrix(self, X):
		for i in range (0, len(X)):
			distanceRow = []
			for j in range (0, len(X)):
				dist = self.euclidean(np.array(X[i]), np.array(X[j]))
				# dist = self.manhattan(np.array(X[i]), np.array(X[j]))
				distanceRow.append(dist)
			self.distanceMatrix.append(distanceRow)
			self.distanceMatrixMember.append([i])
		self.clusterList.append(self.distanceMatrixMember[:])

	def getPossibleValues(self, dataList1, dataList2):
		arr = []
		for data1 in dataList1:
			for data2 in dataList2:
				if isinstance(data1, int) and isinstance(data2, int):
					arr.append(self.distanceMatrix[data1][data2])
				else:
					if isinstance(data1, int) and not isinstance(data2, int):
						arr.append(self.distanceMatrix[data1][data2[0]])
					elif not isinstance(data1, int) and isinstance(data2, int):
						arr.append(self.distanceMatrix[data1[0]][data2])
					else:
						arr.append(self.distanceMatrix[data1[0]][data2[0]])
		return arr
	
	def singleLinkage(self, dataList1, dataList2):
		minValue = 0
		if (dataList1 != dataList2):
			arr = self.getPossibleValues(dataList1, dataList2)
			minValue = np.amin(arr)
		return minValue

	def completeLinkage(self, dataList1, dataList2):
		maxValue = 0
		if (dataList1 != dataList2):
			arr = self.getPossibleValues(dataList1, dataList2)
			maxValue = np.amax(arr)
		return maxValue

	def averageLinkage(self, dataList1, dataList2):
		avgValue = 0
		if (dataList1 != dataList2):
			arr = self.getPossibleValues(dataList1, dataList2)
			avgValue = np.average(arr)
		return avgValue
	
	def averageGroupLinkage(self, X, dataList1, dataList2):
		avgValue = 0
		centroid1 = []
		centroid2 = []
		if len(dataList1) > 1:
			centroid1 = self.createCentroid(dataList1)
		else:
			centroid1 = X[dataList1[0]]

		if (len(dataList2) > 1):
			centroid2 = self.createCentroid(dataList2)
		else:
			centroid2 = X[dataList2[0]]
		# avgValue = self.manhattan(centroid1, centroid2)
		avgValue = self.euclidean(centroid1, centroid2)

		return avgValue

	def createCentroid(self, dataList):
		centroid = []
		for i in range(0, len(self.X[0])):
			centroid.append(0)
		for data in dataList:
			for i in range(0, len(self.X[data])):
				centroid[i] += self.X[data][i]
		for i in range (len(centroid)):
			centroid[i] = float(centroid[i])/len(dataList)
		return centroid

	def isAllInOneCluster(self):
		for cluster in self.clusterList:
			arr = np.array(cluster)
			if len(cluster[0]) == 	self.dataLength:
				self.allInOneCluster = True
				break
	
	def printCluster(self):
		print ("CLUSTER LIST")
		for cluster in agglo.clusterList:
			print(cluster)

	def getClusterList(self):
		clust = self.clusterList[0]
		if self.nb_cluster < self.dataLength:
			#find cluster with length = nb_cluster
			for cluster in self.clusterList:
				if len(cluster) == self.nb_cluster:
					clust = cluster[:] 
		return clust

	def generateLabel(self):
		labelList = []
		self.selectedClusterList = self.getClusterList()
		for i in range(0, self.dataLength):
			labelList.append(1)

		for i in range(0, len(self.selectedClusterList)):
			for data in self.selectedClusterList[i]:
				labelList[data] = i

		return np.array(labelList)

	def fit(self, X):
		self.dataLength = len(X)
		self.X = X[:]
		self.initDistanceMatrix(X)
		self.distanceMatrixChanged = self.distanceMatrix[:]
		z = 0
		while not (self.allInOneCluster):
			#get min distance value
			arr = np.array(self.distanceMatrixChanged)
			minValue = 0
			try:
				minValue = np.min(arr[np.nonzero(arr)])
			except ValueError:
				pass
			if (minValue == 0):
				minIndexTemp = np.where(arr == np.min(arr))
				minIndexList = list(zip(minIndexTemp[0], minIndexTemp[1]))
				minIndex = []
				for min in minIndexList:
					if (min[0] != min[1]):
						minIndex = min
						break
			else:	
				minIndexTemp = np.where(arr == np.min(arr[np.nonzero(arr)]))
				minIndexList = list(zip(minIndexTemp[0], minIndexTemp[1]))
				minIndex = minIndexList[0]

			#update distance member list
			distanceMatrixMemberTemp = self.distanceMatrixMember[:]
			newCluster = []
			for data in minIndex:
				self.distanceMatrixMember.remove(distanceMatrixMemberTemp[data])
				newCluster.append(distanceMatrixMemberTemp[data])
			self.distanceMatrixMember.append(np.concatenate(newCluster).ravel().tolist())

			# save cluster
			self.clusterList.append(self.distanceMatrixMember[:])

			#create new distance matrix
			temp = []
			for i in range(0, len(self.distanceMatrixMember)):
				data = self.distanceMatrixMember[i]
				distanceRow = []
				for j in range(0, len(self.distanceMatrixMember)):
					dist = 0
					if (self.linkage == "single"):
						dist = self.singleLinkage(self.distanceMatrixMember[i], self.distanceMatrixMember[j])
					elif (self.linkage == "complete"):
						dist = self.completeLinkage(self.distanceMatrixMember[i], self.distanceMatrixMember[j])
					elif(self.linkage == "average"):
						dist = self.averageLinkage(self.distanceMatrixMember[i], self.distanceMatrixMember[j])
					else:
						dist = self.averageGroupLinkage(X, self.distanceMatrixMember[i], self.distanceMatrixMember[j])
					distanceRow.append(dist)	
				temp.append(distanceRow[:])

			self.distanceMatrixChanged = temp[:]	
			self.isAllInOneCluster()	
			z+=1
		labels = self.generateLabel()
		return labels

	def predict(self, X_test, y_train_pred):
		centroidList = []
		centroidLabelList = []
		#create centroid from cluster
		for cluster in self.selectedClusterList:
			centroid = self.createCentroid(cluster)
			centroidLabelList.append(y_train_pred[cluster[0]])
			centroidList.append(centroid)
		
		#predict
		y_pred = []
		for data in X_test:
			minDist = self.euclidean(data, centroidList[0])
			clusterPred = 0
			for i in range(len(centroidList)):
				dist = self.euclidean(data, centroidList[i])
				if (dist < minDist):
					minDist = dist
					clusterPred = centroidLabelList[i]
			y_pred.append(clusterPred)
		return np.array(y_pred)

def readData():
	dataset = pd.read_csv('iris.data', names=["1", "2", "3", "4", "label"])
	df = pd.DataFrame(dataset)
	X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=0.1, random_state=42)

	dicti = {"Iris-setosa": 0, "Iris-versicolor":1, "Iris-virginica":2}
	y_train = y_train.apply(lambda y: dicti[y])
	y_test = y_test.apply(lambda y: dicti[y])
	
	return np.array(X_train), np.array(X_test), y_train, y_test

def plot(X, labels):
	plt.scatter(X[labels==0, 0], X[labels==0, 1], s=5, marker='o', color='purple')
	plt.scatter(X[labels==1, 0], X[labels==1, 1], s=5, marker='o', color='green')
	plt.scatter(X[labels==2, 0], X[labels==2, 1], s=5, marker='o', color='red')
	plt.scatter(X[labels==3, 0], X[labels==3, 1], s=5, marker='o', color='orange')
	plt.scatter(X[labels==4, 0], X[labels==4, 1], s=5, marker='o', color='blue')
	plt.show()

def convertLabel(X, y, y_pred, nb_cluster):
    target = {}
    for i in range(nb_cluster):
        id = np.where(y_pred == i)
        target[i] = y.iloc[id].mode()[0]
    y_conv = np.vectorize(target.get)(y_pred)
    return y_conv

if __name__ == "__main__":
	X_train, X_test, y_train, y_test = readData()

	#AGGLOMERATIVE CLUSTERING ------------------------
	model = AggloClustering(3, "average") #linkage: single, complete, average, average-group
	
	#fit
	y_train_pred = model.fit(X_train)
	y_train_pred = convertLabel(X_train, y_train, y_train_pred, 3)
	plot(np.array(X_train), y_train_pred)
	
	#predict
	y_test_pred = model.predict(X_test, y_train_pred)
	plot(np.array(X_test), y_test_pred)
	
	#get accuracy
	accuracy = accuracy_score(y_test, y_test_pred)
	print(accuracy)

	#AGGLOMERATIVE CLUSTERING WITH SCIKIT LEARN ------------
	skmodel = AgglomerativeClustering(n_clusters=3, linkage='average')
	
	#fit
	skmodel.fit(X_train)
	sk_y_train_pred = convertLabel(X_train, y_train, skmodel.labels_, 3)
	plot(np.array(X_train), sk_y_train_pred)

	#predict
	sk_y_test_pred = skmodel.fit_predict(X_test)
	plot(np.array(X_test), sk_y_test_pred)

	#accuracy
	accuracy = accuracy_score(y_test, sk_y_test_pred)
	print(accuracy)