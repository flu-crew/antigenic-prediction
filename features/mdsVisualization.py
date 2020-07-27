# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:41:52 2017

@author: mazeller
"""

#Imports
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits import mplot3d
from matplotlib import cm
import matplotlib.colors as mc
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from mpl_toolkits.mplot3d import Axes3D

class mdsVisualization:

	def __init__(self):
		"""Creates features from alignments and antigenic distances"""
		self.antigenicDistances = None
		self.colors = None
		self.coordinates = None
		
	#Create a symmetric matrix
	def mirrorUpperTriangle(self):
		self.antigenicDistances = np.transpose(np.triu(self.antigenicDistances.values)) + np.triu(self.antigenicDistances.values)
	
	#Create a symmetric matrix
	def mirrorLowerTriangle(self):
		self.antigenicDistances = np.transpose(np.tril(self.antigenicDistances.values)) + np.tril(self.antigenicDistances.values)
	
	def loadAntigenicDistanceLongColumns(self, fileAntigenicDistanceTable):
		df = pd.read_csv(fileAntigenicDistanceTable)
		df = df.pivot(index = "Name1", columns = "Name2", values = "distance") #Long to wide
		df = df.fillna(0)
		self.antigenicDistances = df

	def loadAntigenicDistanceTable(self, fileAntigenicDistanceTable):
		df = pd.read_csv(fileAntigenicDistanceTable, index_col = [0], header = 0)
		df = df.fillna(0)
		self.antigenicDistances = df
		
	def appendPredictionData(self, filePredictionData, valueColumn):
		testData = pd.read_csv(filePredictionData, header = 0)
		testColumns = testData.antigen.unique()
		testRFDf = testData.pivot(index = 'antigen', columns = 'antiserum', values = valueColumn) #For adding to the bottom
		testRFDf.index = [str(col) + '_' + valueColumn for col in testRFDf.index]
		FullMergedDf = self.antigenicDistances .append(testRFDf, sort = False)
		if(FullMergedDf.shape[0] == FullMergedDf.shape[1]):
			FullMergedDf.drop(FullMergedDf.columns[list(range(-1,-1 * (testRFDf.shape[0] +1),-1))], axis=1, inplace=True)
			FullMergedDf = FullMergedDf.transpose()
			testRFDf.columns = FullMergedDf.columns
		else:
			FullMergedDf = FullMergedDf.transpose()
		FullMergedDf = FullMergedDf.append(testRFDf, sort = False)
		FullMergedDf.fillna(0, inplace = True)
		self.antigenicDistances = FullMergedDf
	
	def compute3dMDS(self):
		mdsObj = MDS(n_components = 3, metric = True, verbose = 1, max_iter=3000, eps=1e-9,  dissimilarity = "precomputed", random_state  = 777) #, dissimilarity='precomputed')
		self.coordinates = mdsObj.fit_transform(self.antigenicDistances)
		
	def plot3dResult(self, figureNumber = 1, angle1 = 0, angle2 = 0):
		plt.figure(figureNumber)
		ax = plt.axes(projection='3d')
		if self.colors is not None:
			ax.scatter3D(self.coordinates[:,0], self.coordinates[:,1], self.coordinates[:,2], color = self.colors)
		else:
			ax.scatter3D(self.coordinates[:,0], self.coordinates[:,1], self.coordinates[:,2])

		ax.view_init(angle1, angle2)
			
	def fillColors(self, fillValue = "#000000"):
		if self.colors is None:
			self.colors = []
		self.colors = list(self.colors)
		newColors = [fillValue] * (len(self.coordinates) - len(self.colors)) #['#ffcc00']
		colorSet = list(self.colors) + newColors
		self.colors = colorSet

	def fillMap(self, map = "Wistia"):
		if self.colors is None:
			self.colors = []
		self.colors = list(self.colors)
		cmap = cm.get_cmap(map, (self.antigenicDistances.shape[0] - len(self.colors)))    # PiYG
		newColors = []
		for i in range(cmap.N):
		    rgb = cmap(i) 
		    newColors.append(mc.to_hex(rgb))
		colorSet = list(self.colors) + newColors
		self.colors = colorSet

	def orthogonalProcrustesCentered(self, mtx1, mtx2):
		"""
		modified scipy procrustes to avoid scaling
		
		Returns:
			mtx1 translated to center of mass
			mtx2 rotated to minimize distance to mtx1, translated to center of mass
		"""
		mtx1 = np.array(mtx1, dtype=np.double, copy=True)
		mtx2 = np.array(mtx2, dtype=np.double, copy=True)
		
		# translate all the data to the origin
		mtx1 -= np.mean(mtx1, 0)
		mtx2 -= np.mean(mtx2, 0)
		
		R, s = orthogonal_procrustes(mtx1, mtx2)
		
		mtx2 = np.dot(mtx2, R.T) #* s # Avoid scaling

		disparity = np.sum(np.square(mtx1 - mtx2))

		return mtx1, mtx2, disparity
	
	def rescaledProcrustes(self, mtx1, mtx2):
		#Move all data to origin
		newMtx1, newMtx2, disparity1 = procrustes(mtx1, mtx2)
		
		#Rescale by multiplying the normals
		newMtx1 *= np.linalg.norm(mtx1)
		newMtx2 *= np.linalg.norm(mtx2)
			
		return newMtx1, newMtx2

	def getDistArr(self, coord1, coord2):
		return np.sqrt(np.add(np.subtract(coord1[:,0],coord2[:,0])**2,np.subtract(coord1[:,1],coord2[:,1])**2,np.subtract(coord1[:,2],coord2[:,2])**2))
	
	def iterativeMDSwithProcrustes(self, filePredictionData, valueColumn):
		self.compute3dMDS()
		baseMap = self.coordinates
		baseMap, finalMap, disparity = self.orthogonalProcrustesCentered(baseMap, baseMap) #Centering
	
		#Load in all prediction data
		testData = pd.read_csv(filePredictionData, header = 0)
		testColumns = testData.antigen.unique()
		testRFDf = testData.pivot(index = 'antigen', columns = 'antiserum', values = valueColumn) #For adding to the bottom

		#Iterate through each prediction, do MDS, solve procrustes, add coords
		for i in range(0,len(testRFDf)):
			row = testRFDf.iloc[i]
	
			#Assemble an appropriate matrix for MDS, 
			FullMergedDf = self.antigenicDistances.append(row, sort = False)
			FullMergedDf = FullMergedDf.transpose()
			FullMergedDf = FullMergedDf.append(row, sort = False)
			FullMergedDf.fillna(0, inplace = True)
		
			#Perform MDS
			mdsObj = MDS(n_components = 3, metric = True, verbose = 0, max_iter=3000, eps=1e-9,  dissimilarity = "precomputed", random_state  = 777) 
			newMap = mdsObj.fit_transform(FullMergedDf)
			
			#Procrustes to orientate 
			mtx1 = np.array(baseMap, dtype=np.double, copy=True)
			mtx2 = np.array(newMap[0:len(newMap) - 1], dtype=np.double, copy=True)
			
			
			mtx1 -= np.mean(mtx1, 0)
			mtx2 -= np.mean(mtx2, 0)
			
			R, s = orthogonal_procrustes(mtx1, mtx2)
			newMap = np.dot(newMap, R.T) #* s # Avoid scaling; Add in the one added point
	
			#Tack last value of the newMap onto the finalMap
			finalMap = np.vstack([finalMap, newMap[-1]])
			self.coordinates = finalMap
			
			print("Completion {}/{}".format(i+1, len(testRFDf)))
		
	#https://stackoverflow.com/questions/24123659/scatter-plot-3d-with-labels-and-spheres
	def drawSphere(self, xCenter, yCenter, zCenter, r):
	    #draw sphere
	    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
	    x=np.cos(u)*np.sin(v)
	    y=np.sin(u)*np.sin(v)
	    z=np.cos(v)
	    # shift and scale sphere
	    x = r*x + xCenter
	    y = r*y + yCenter
	    z = r*z + zCenter
	    return (x,y,z)
	
	def plotSpheres(self, coords, sizes, wireFrame = True):
		plt.figure()
		ax = plt.axes(projection='3d')
		
		i = 0
		for (xi,yi,zi,ri) in zip(coords[:,0], coords[:,1], coords[:,2], sizes):
			(xs,ys,zs) = self.drawSphere(xi,yi,zi,ri)
			if wireFrame:
				ax.plot_wireframe(xs, ys, zs, color=self.colors[i],rstride=4, cstride=4)
			else:
				ax.plot_surface(xs, ys, zs, color=self.colors[i],rstride=4, cstride=4)
			i += 1
			
		return ax
		
	
#def main():
#	visualizer = mdsVisualization()
##	visualizer.loadAntigenicDistanceLongColumns("../data/carine_2018_raw.csv")
##	visualizer.mirrorUpperTriangle()
##	visualizer.compute3dMDS()
##	visualizer.plot3dResult()
#	
#	h3Colors = pd.read_csv("data/motiflist.csv", header = None)
#	visualizer.loadAntigenicDistanceTable("data/h3matrixmodified_orig.csv")
#	visualizer.colors = h3Colors[3]
#	#visualizer.appendPredictionData("unknownset.csv", "Random Forest")
#	test = visualizer.antigenicDistances
#	
#	########################
#	#Generate base 3D mapping
#	filePredictionData = "unknownset.csv"
#	valueColumn = "MLP Network"
#	visualizer.compute3dMDS()
#	baseMap = visualizer.coordinates
#	baseMap, finalMap, disparity = visualizer.orthogonalProcrustesCentered(baseMap, baseMap) #Centering
#
#	
#	#Load in all prediction data
#	testData = pd.read_csv(filePredictionData, header = 0)
#	testColumns = testData.antigen.unique()
#	testRFDf = testData.pivot(index = 'antigen', columns = 'antiserum', values = valueColumn) #For adding to the bottom
#
#	#Iterate through each prediction, do MDS, solve procrustes, add coords
#	for i in range(0,len(testRFDf)):
#		row = testRFDf.iloc[i]
#
#		#Assemble an appropriate matrix for MDS, 
#		FullMergedDf = visualizer.antigenicDistances.append(row, sort = False)
#		FullMergedDf = FullMergedDf.transpose()
#		FullMergedDf = FullMergedDf.append(row, sort = False)
#		FullMergedDf.fillna(0, inplace = True)
#	
#		#Perform MDS
#		mdsObj = MDS(n_components = 3, metric = True, verbose = 1, max_iter=3000, eps=1e-9,  dissimilarity = "precomputed", random_state  = 777) 
#		newMap = mdsObj.fit_transform(FullMergedDf)
#		
#		#Procrustes to orientate 
#		mtx1 = np.array(baseMap, dtype=np.double, copy=True)
#		mtx2 = np.array(newMap[0:len(newMap) - 1], dtype=np.double, copy=True)
#		
#		
#		mtx1 -= np.mean(mtx1, 0)
#		mtx2 -= np.mean(mtx2, 0)
#		
#		R, s = orthogonal_procrustes(mtx1, mtx2)
#		newMap = np.dot(newMap, R.T) #* s # Avoid scaling; Add in the one added point
#
#		#Tack last value of the newMap onto the finalMap
#		finalMap = np.vstack([finalMap, newMap[-1]])
#
#		plt.figure()
#		ax = plt.axes(projection='3d')
#		ax.scatter3D(finalMap[:,0], finalMap[:,1], finalMap[:,2])
#	#######################
#	
#	
#	visualizer.fillColors("#ffcc00")
#	visualizer.compute3dMDS()
#	visualizer.plot3dResult(1)
#	return
#
#	visualizer.loadAntigenicDistanceTable("../data/h3matrixmodified_orig.csv")
#	visualizer.colors = h3Colors[3]
#	visualizer.appendPredictionData("../data/cluster_iva_2019_translation.fasta.features.csv.estimates.csv", "AdaBoosted")
#	visualizer.fillColors("#edaa0c")
#	visualizer.compute3dMDS()
#	visualizer.plot3dResult(2)
#	
#	visualizer.loadAntigenicDistanceTable("../data/h3matrixmodified_orig.csv")
#	visualizer.colors = h3Colors[3]
#	visualizer.appendPredictionData("../data/cluster_iva_2019_translation.fasta.features.csv.estimates.csv", "MLP Network")
#	visualizer.fillColors("#faff73")
#	visualizer.compute3dMDS()
#	visualizer.plot3dResult(3)
#	
#	visualizer.loadAntigenicDistanceTable("../data/h3matrixmodified_orig.csv")
#	visualizer.colors = h3Colors[3]
#	visualizer.appendPredictionData("../data/cluster_iva_2019_translation.fasta.features.csv.estimates.csv", "Ensemble")
#	visualizer.fillColors("#ffb88c")
#	visualizer.compute3dMDS()
#	visualizer.plot3dResult(4)
#	
#if __name__ == "__main__": main()
