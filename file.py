import numpy as np
from scipy.spatial.distance import cdist

class AgglomerativeClustering:
	def __init__(self, nb_cluster=2, linkage="single"):
		self.nb_cluster = nb_cluster
		self.linkage = linkage
		self.distanceMatrix = []
		self.distanceMatrixChanged = []
		self.distanceMatrixMember = []
		self.clusterList = []
		self.allInOneCluster = False
		self.dataLength = 0

	def euclidean(self, a, b):
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
				# dist = self.euclidean(np.array(X[i]), np.array(X[j]))
				dist = self.manhattan(np.array(X[i]), np.array(X[j]))
				distanceRow.append(dist)
			self.distanceMatrix.append(distanceRow)
			self.distanceMatrixMember.append([i])
			self.clusterList.append([i])
	
	def singleLinkage(self, dataList1, dataList2):
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
		return np.amin(arr)

	def completeLinkage(self, dataList1, dataList2):
		arr = []
		maxValue = 0
		if (dataList1 != dataList2):
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
			maxValue = np.amax(arr)
		return maxValue

	def averageLinkage(self, dataList1, dataList2):
		arr = []
		maxValue = 0
		if (dataList1 != dataList2):
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
			maxValue = np.average(arr)
		return maxValue

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
		avgValue = self.manhattan(centroid1, centroid2)

		return avgValue

	def createCentroid(self, dataList):
		centroid = []
		for i in range(0, len(X[0])):
			centroid.append(0)
		for data in dataList:
			for i in range(0, len(X[data])):
				centroid[i] += X[data][i]
		for i in range (len(centroid)):
			centroid[i] = float(centroid[i])/len(dataList)
		return centroid

	def isAllInOneCluster(self):
		for cluster in self.clusterList:
			arr = np.array(cluster)
			if arr.ndim > 1:
				if len(cluster[0]) == 	self.dataLength:
					self.allInOneCluster = True
					break
	
	def printCluster(self):
		print ("CLUSTER LIST")
		for cluster in agglo.clusterList:
			print(cluster)

	def fit(self, X):
		self.dataLength = len(X)
		self.initDistanceMatrix(X)
		self.distanceMatrixChanged = self.distanceMatrix[:]
		while not (self.allInOneCluster):
			#get min distance value
			arr = np.array(self.distanceMatrixChanged)
			minValue = np.min(arr[np.nonzero(arr)])
			nb_minValue = (arr == minValue).sum()
			minIndex=[]
			if (nb_minValue == 1):
				minIndex = np.where(arr == np.min(arr[np.nonzero(arr)]))[0]
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
					elif(self.linkage == "group-average"):
						dist = self.averageGroupLinkage(X, self.distanceMatrixMember[i], self.distanceMatrixMember[j])
					distanceRow.append(dist)	
				temp.append(distanceRow[:])

			self.distanceMatrixChanged = temp[:]	
			self.isAllInOneCluster()	



		

agglo = AgglomerativeClustering(2, "group-average")
X = [[1,1], [4,1], [1,2], [3,4], [5,4]]
# X = [[0.4, 0.53], [0.22, 0.38], [0.35,0.32], [0.26, 0.19], [0.08,0.41], [0.45,0.3]]
agglo.fit(X)
agglo.printCluster()
