import numpy as np

class AgglomerativeClustering:
	def __init__(self, nb_cluster=2, linkage="single"):
		self.nb_cluster = nb_cluster
		self.linkage = linkage
		self.distanceMatrix = []


	def euclidean(self, a, b):
		dist = np.linalg.norm(a-b, ord=2)
		return dist

	def initDistanceMatrix(self, X):
		for i in range (0, len(X)):
			distanceRow = []
			for j in range (0, len(X)):
				dist = self.euclidean(np.array(X[i]), np.array(X[j]))
				distanceRow.append(dist)
			self.distanceMatrix.append(distanceRow)
		print(self.distanceMatrix)
	
	def singleLinkage(self):
		arr = np.array(self.distanceMatrix)
		minValue = np.min(arr[np.nonzero(arr)])
		minIndex = np.where(arr == np.min(arr[np.nonzero(arr)]))[0]
		distanceMatrixTemp = self.distanceMatrix
		self.distanceMatrix

	def completeLinkage(self):
		return 0

	def averageLinkage(self):
		return 0
		
	def fit(self, X):
		self.initDistanceMatrix(X)
		if (self.linkage == "single"):
			self.singleLinkage()
		elif (self.linkage == "complete"):
			self.completeLinkage()
		elif(self.linkage == "average"):
			self.averageLinkage()


print("Halo")
agglo = AgglomerativeClustering(2, "single")
X = [[1,2,3,4,5], [3,5,6,7,8], [2,2,2,2,2]]
agglo.fit(X)