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
		
		# or 
		# dist = cdist(a,b, metric='euclidean')
		return dist

	def manhattan(self, a, b):
		# dist = cdist(a,b, metric='cityblock')
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
		print("datalist1", dataList1)
		print("datalist2", dataList2)
		for data1 in dataList1:
			for data2 in dataList2:
				if isinstance(data1, int) and isinstance(data2, int):
					print("a",data1, data2, self.distanceMatrix[data1][data2])
					arr.append(self.distanceMatrix[data1][data2])
				else:
					if isinstance(data1, int) and not isinstance(data2, int):
						print("b",data1, data2[0], self.distanceMatrix[data1][data2[0]])
						arr.append(self.distanceMatrix[data1][data2[0]])
					elif not isinstance(data1, int) and isinstance(data2, int):
						print("c",data1[0], data2, self.distanceMatrix[data1[0]][data2])
						arr.append(self.distanceMatrix[data1[0]][data2])
					else:
						print("d",data1[0], data2[0])
						print(self.distanceMatrix[data1[0]][data2[0]])
						arr.append(self.distanceMatrix[data1[0]][data2[0]])
						print("ok")
		print("\n")
		return np.amin(arr)

	def completeLinkage(self, dataList1, dataList2):
		arr = []
		print("datalist1", dataList1)
		print("datalist2", dataList2)
		maxValue = 0
		if (dataList1 != dataList2):
			for data1 in dataList1:
				for data2 in dataList2:
					if isinstance(data1, int) and isinstance(data2, int):
						print("a",data1, data2, self.distanceMatrix[data1][data2])
						arr.append(self.distanceMatrix[data1][data2])
					else:
						if isinstance(data1, int) and not isinstance(data2, int):
							print("b",data1, data2[0], self.distanceMatrix[data1][data2[0]])
							arr.append(self.distanceMatrix[data1][data2[0]])
						elif not isinstance(data1, int) and isinstance(data2, int):
							print("c",data1[0], data2, self.distanceMatrix[data1[0]][data2])
							arr.append(self.distanceMatrix[data1[0]][data2])
						else:
							print("d",data1[0], data2[0])
							print(self.distanceMatrix[data1[0]][data2[0]])
							arr.append(self.distanceMatrix[data1[0]][data2[0]])
							print("ok")
			print("\n")
			maxValue = np.amax(arr)
		return maxValue

	def averageLinkage(self, dataList1, dataList2):
		arr = []
		print("datalist1", dataList1)
		print("datalist2", dataList2)
		maxValue = 0
		if (dataList1 != dataList2):
			for data1 in dataList1:
				for data2 in dataList2:
					if isinstance(data1, int) and isinstance(data2, int):
						print("a",data1, data2, self.distanceMatrix[data1][data2])
						arr.append(self.distanceMatrix[data1][data2])
					else:
						if isinstance(data1, int) and not isinstance(data2, int):
							print("b",data1, data2[0], self.distanceMatrix[data1][data2[0]])
							arr.append(self.distanceMatrix[data1][data2[0]])
						elif not isinstance(data1, int) and isinstance(data2, int):
							print("c",data1[0], data2, self.distanceMatrix[data1[0]][data2])
							arr.append(self.distanceMatrix[data1[0]][data2])
						else:
							print("d",data1[0], data2[0])
							print(self.distanceMatrix[data1[0]][data2[0]])
							arr.append(self.distanceMatrix[data1[0]][data2[0]])
							print("ok")
			maxValue = np.average(arr)
			print("average", maxValue)
			print("\n")
		return maxValue

	def isAllInOneCluster(self):
		for cluster in self.clusterList:
			arr = np.array(cluster)
			if arr.ndim > 1:
				if len(cluster[0]) == 	self.dataLength:
					self.allInOneCluster = True
					break
	
	def fit(self, X):
		self.dataLength = len(X)
		self.initDistanceMatrix(X)
		print("----------------distance member---------------")
		print(self.distanceMatrixMember)
		print("----------------distance matrix---------------")
		for i in range(len(self.distanceMatrix)):
			print(self.distanceMatrix[i])
		print("---------------------------------------------")
		self.distanceMatrixChanged = self.distanceMatrix[:]
		while not (self.allInOneCluster):
		# for i in range(0,3):
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
			distanceMatrixMemberTemp = self.distanceMatrixMember[:]
			newCluster = []
			print("minIndex", minIndex, "minvalue", minValue)
			for data in minIndex:
				print("data to be removed: ", data)
				self.distanceMatrixMember.remove(distanceMatrixMemberTemp[data])
				newCluster.append(distanceMatrixMemberTemp[data])
			self.distanceMatrixMember.append(np.concatenate(newCluster).ravel().tolist())
			# save cluster
			self.clusterList.append(self.distanceMatrixMember[:])
			print("cluster list", self.clusterList)

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
					distanceRow.append(dist)	
				temp.append(distanceRow[:])
			self.distanceMatrixChanged = temp[:]	
			print("----------------distance member---------------")
			print(self.distanceMatrixMember)
			print("----------------distance matrix---------------")
			for i in range(len(self.distanceMatrixChanged)):
				print(self.distanceMatrixChanged[i])
			print("---------------------------------------------")
			self.isAllInOneCluster()	



		

print("Halo")
agglo = AgglomerativeClustering(2, "average")
X = [[1,1], [4,1], [1,2], [3,4], [5,4]]
# X = [[0.4, 0.53], [0.22, 0.38], [0.35,0.32], [0.26, 0.19], [0.08,0.41], [0.45,0.3]]
agglo.fit(X)
print ("CLUSTER LIST")
for cluster in agglo.clusterList:
	print(cluster)