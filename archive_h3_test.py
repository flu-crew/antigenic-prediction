import features.featureMaker as fm
import features.featureRegression as fr
import features.mdsVisualization as mv
import pandas as pd
import numpy as np

#Training set
h3Set = fm.featureMaker("data/translated.fasta", "data/h3matrixmodified_orig.csv")
h3Set.gatherAminoAcidDifferences()
#h3Set.gatherGlycosylationDifferences()
h3Set.defineFeatures()
h3Set.exportFeatures("training.csv")
h3Set.createTestSet("test_cases/civ_a.fasta", selfCompare = False)
h3Set.exportTestSetFeatures("testFeatures.csv")

regressor = fr.featureRegression("training.csv")
regressor.trainRandomForestRegressor()
regressor.randomForestRegr
regressor.trainAdaBoostedRegressor()
regressor.trainMultilayerPerceptron()
#regressor.tuneRandomForestRegressor()
#regressor.tuneAdaBoostedRegressor()
#regressor.tuneMultilayerPerceptron()
regressor.printEvaluationMetrics()
regressor.visualizePerformance()
regressor.predictUnknownSet("testFeatures.csv")
regressor.exportUnknownSet("unknownset.csv")
#regressor.getCV()
regressor.visualizePredictions()
#regressor.randomForestRegr.feature_importances_

#https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html	
visualizer = mv.mdsVisualization()
h3Colors = pd.read_csv("data/motiflist.csv", header = None)
visualizer.loadAntigenicDistanceTable("data/h3matrixmodified_orig.csv")
#visualizer.colors = ['#AAAAAA'] * (len(h3Colors[3])-1) 
visualizer.colors = h3Colors[3]
visualizer.appendPredictionData("unknownset.csv", "Random Forest")
visualizer.fillColors("#ffda05")
#visualizer.fillMap("rainbow")
visualizer.compute3dMDS()
visualizer.plot3dResult(1)
coordRF = visualizer.coordinates

visualizer.loadAntigenicDistanceTable("data/h3matrixmodified_orig.csv")
visualizer.colors = h3Colors[3]
visualizer.appendPredictionData("unknownset.csv", "AdaBoosted")
visualizer.fillColors("#ffda05")
visualizer.compute3dMDS()
visualizer.plot3dResult(2, angle2 = 180)
coordAda = visualizer.coordinates

visualizer.loadAntigenicDistanceTable("data/h3matrixmodified_orig.csv")
visualizer.colors = h3Colors[3]
visualizer.appendPredictionData("unknownset.csv", "MLP Network")
visualizer.fillColors("#ffda05")
visualizer.compute3dMDS()
visualizer.plot3dResult(3)
coordMLP = visualizer.coordinates

visualizer.loadAntigenicDistanceTable("data/h3matrixmodified_orig.csv")
visualizer.colors = h3Colors[3]
visualizer.appendPredictionData("unknownset.csv", "Ensemble")
visualizer.fillColors("#ffda05")
visualizer.compute3dMDS()
visualizer.plot3dResult(4)
coordEns = visualizer.coordinates

overlayMLP, overlayRF, disparity1 = visualizer.orthogonalProcrustesCentered(coordMLP, coordRF)
overlayMLP, overlayAda, disparity2 = visualizer.orthogonalProcrustesCentered(coordMLP, coordAda)
overlayMLP, overlayEns, disparity3 = visualizer.orthogonalProcrustesCentered(coordMLP, coordEns)

import matplotlib.pyplot as plt
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(overlayEns[:,0],overlayEns[:,1],overlayEns[:,2], color = visualizer.colors)
ax.scatter3D(overlayRF[:,0],overlayRF[:,1],overlayRF[:,2], color = visualizer.colors)
ax.scatter3D(overlayAda[:,0],overlayAda[:,1],overlayAda[:,2], color = visualizer.colors)
ax.scatter3D(overlayMLP[:,0],overlayMLP[:,1],overlayMLP[:,2], color = visualizer.colors)

np.std([overlayRF,overlayAda,overlayMLP], axis = 0)

rf = visualizer.getDistArr(overlayEns,overlayRF)
ada = visualizer.getDistArr(overlayEns,overlayAda)
mlp = visualizer.getDistArr(overlayEns,overlayMLP)

stdDevs = np.std([rf, ada, mlp], axis = 0)

sizes = np.divide(np.add(np.add(rf,ada),mlp),3)
ax = visualizer.plotSpheres(overlayEns, mlp, wireFrame = True)
ax = visualizer.plotSpheres(overlayEns, stdDevs, wireFrame = True)

