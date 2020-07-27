#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:45:42 2019

@author: michael.zeller
"""

import features.asymDataProcess as fm
import features.featureRegression as fr
import features.mdsVisualization as mv
import pandas as pd
import numpy as np
from io import StringIO

#Training set
h3Set = fm.asymDataProcess()
h3Set.loadFoldChangeTable("raw_data/master_short.fasta", "raw_data/carine_final.csv")
h3Set.defineFeatures()
h3Set.exportFeatures("raw_data/carine_feature_set.csv")



#Load in sets, train, and predict missing
antigenIndex = list(h3Set.distances.index)
baseFrame = pd.read_csv("raw_data/carine_feature_set.csv", header=0, index_col=[0])
regressor = fr.featureRegression("raw_data/carine_feature_set.csv")
i = 0
for row in antigenIndex:
	#remove data in copy
	print(row)
	modFrame = baseFrame.copy()
	indexNames = modFrame[ (modFrame['antigen'] == row) | (modFrame['antiserum'] == row) ].index
	holdFrame = modFrame[ (modFrame['antigen'] == row) | (modFrame['antiserum'] == row) ].copy()
	modFrame.drop(indexNames , inplace=True)
	
	#Inject trainx and trainy
	regressor.trainY = modFrame.iloc[:,2].values
	regressor.trainX = modFrame.iloc[:,3:]
	
	#Train
	regressor.trainRandomForestRegressor()
	regressor.randomForestRegr
	regressor.trainAdaBoostedRegressor()
	regressor.trainMultilayerPerceptron()
	regressor.printEvaluationMetrics()
	
	#Test bypass
	unknownX = holdFrame.iloc[:,3:]
	
	#Predict
	rfPred = regressor.randomForestRegr.predict(unknownX)
	adaPred = regressor.adaBoostedRegr.predict(unknownX)
	mlpPred = regressor.multilayerPerceptronRegr.predict(unknownX)
	
	#Push to dataframe
	holdFrame.reset_index(inplace = True)
	unknownDf = pd.DataFrame(np.zeros((len(holdFrame), 2)),columns = ['antigen','antiserum'])
	unknownDf['antigen'] = holdFrame['antigen']
	unknownDf['antiserum'] = holdFrame['antiserum']
	unknownDf['Random Forest'] = rfPred
	unknownDf['AdaBoosted'] = adaPred
	unknownDf['MLP Network'] = mlpPred
	unknownDf['Ensemble'] = (rfPred + adaPred + mlpPred) / 3
	
	#Export to a file
	unknownDf.to_csv("raw_data/loo2/" + str(i) + ".csv" )
	holdFrame.to_csv("raw_data/loo2/" + str(i) + "_real.csv")
	
	#Increment counter
	i += 1	

#Iterate through and calculate averages and variances
outFile = open("raw_data/loo2/summary.tsv","w+")
outFile.write("seq	rf	ada	mlp	ens	rfstd	adastd	mlpstd	ensstd\n")
summaryFile = open("raw_data/loo2/dist_summary.tsv","w+")
summaryFile.write("seq	rf	ada	mlp	ens\n")
summaryFile.close()
i = 0
for row in antigenIndex:
	print(row)
	unknownDf = pd.read_csv("raw_data/loo2/" + str(i) + ".csv", header = 0, index_col=[0])
	holdFrame = pd.read_csv("raw_data/loo2/" + str(i) + "_real.csv", header = 0, index_col=[0])
	rf_avg = np.around(np.average(np.absolute((holdFrame['dist'] - unknownDf['Random Forest']))),4)
	ada_avg = np.around(np.average(np.absolute((holdFrame['dist'] - unknownDf['AdaBoosted']))),4)
	mlp_avg = np.around(np.average(np.absolute((holdFrame['dist'] - unknownDf['MLP Network']))),4)
	ens_avg = np.around(np.average(np.absolute((holdFrame['dist'] - unknownDf['Ensemble']))),4)
	rf_std = np.around(np.std(holdFrame['dist'] - unknownDf['Random Forest']),4)
	ada_std = np.around(np.std(holdFrame['dist'] - unknownDf['AdaBoosted']),4)
	mlp_std  =np.around(np.std(holdFrame['dist'] - unknownDf['MLP Network']),4)
	ens_std = np.around(np.std(holdFrame['dist'] - unknownDf['Ensemble']),4)
	outFile.write("{}	{}	{}	{}	{}	{}	{}	{}	{}\n".format(row,rf_avg,ada_avg,mlp_avg,ens_avg,rf_std,ada_std,mlp_std,ens_std))

	#Extra line to get error distances and generate names, non-symmetrical
	antigenIndeces = holdFrame.index[holdFrame['antigen'] == row].tolist();
	origDist = holdFrame['dist'].iloc[antigenIndeces]
	rfPredDist = unknownDf['Random Forest'].iloc[antigenIndeces]
	adaPredDist = unknownDf['AdaBoosted'].iloc[antigenIndeces]
	mlpPredDist = unknownDf['MLP Network'].iloc[antigenIndeces]
	ensPredDist = unknownDf['Ensemble'].iloc[antigenIndeces]
	rfDist = np.around(np.absolute(origDist - rfPredDist),1)
	adaDist = np.around(np.absolute(origDist - adaPredDist),1)
	mlpDist = np.around(np.absolute(origDist - mlpPredDist),1)
	ensDist = np.around(np.absolute(origDist - ensPredDist),1)
	names = holdFrame['antigen'].iloc[antigenIndeces] + "." + holdFrame['antiserum'].iloc[antigenIndeces]
	summaryDf = pd.concat([names, rfDist, adaDist, mlpDist, ensDist], axis=1)
	summaryDf.to_csv('raw_data/loo2/dist_summary.tsv', mode='a', header=False)
	i += 1
outFile.close()


#Correct broken ref files
i = 0
for row in antigenIndex:
	print(row)
	predFrame = pd.read_csv("raw_data/loo2/" + str(i) + ".csv", header = 0, index_col = 0)
	#realFrame = pd.read_csv("leave_one_out/" + str(i) + "_real.csv", header = 0, index_col = 0)
	indeces = predFrame[ (predFrame['antigen'] != row) ].index
	predFrame.iloc[indeces, 1] = predFrame.iloc[indeces, 0]
	predFrame.iloc[indeces, 0] = row
	#Remove self reference
	#predFrame[ (predFrame['antigen'] == predFrame['antiserum']) ].drop(inplace = True)
	predFrame.to_csv("raw_data/loo2/" + str(i) + ".csv" )
	i += 1

#I AM HERE	
#Calculate differences in terms of distances on the map overlayed
outFile = open("raw_data/loo2/dist_summary.tsv","w+")
outFile.write("seq	rf	ada	mlp	ens\n")
#h3Colors = pd.read_csv("data/mergedH3Carine/motiflist.csv", header = None)
i = 0
for row in antigenIndex:
	visualizer = mv.mdsVisualization()
	visualizer.loadAntigenicDistanceTable("data/mergedH3Carine/h3merged.csv")
	#visualizer.loadAntigenicDistanceTable("data/h3matrixmodified.csv")
	visualizer.coordinates = None
	visualizer.iterativeMDSwithProcrustes("raw_data/loo2/" + str(i) + ".csv", "Random Forest")
	test = visualizer.antigenicDistances
	rfDist = np.sqrt(np.subtract(visualizer.coordinates[i][0],visualizer.coordinates[-1][0])**2 + np.subtract(visualizer.coordinates[i][1],visualizer.coordinates[-1][1])**2 + np.subtract(visualizer.coordinates[i][2],visualizer.coordinates[-1][2])**2)
	visualizer.iterativeMDSwithProcrustes("raw_data/loo2/" + str(i) + ".csv", "AdaBoosted")
	adaDist = np.sqrt(np.subtract(visualizer.coordinates[i][0],visualizer.coordinates[-1][0])**2 + np.subtract(visualizer.coordinates[i][1],visualizer.coordinates[-1][1])**2 + np.subtract(visualizer.coordinates[i][2],visualizer.coordinates[-1][2])**2)
	visualizer.iterativeMDSwithProcrustes("raw_data/loo2/" + str(i) + ".csv", "MLP Network")
	mlpDist = np.sqrt(np.subtract(visualizer.coordinates[i][0],visualizer.coordinates[-1][0])**2 + np.subtract(visualizer.coordinates[i][1],visualizer.coordinates[-1][1])**2 + np.subtract(visualizer.coordinates[i][2],visualizer.coordinates[-1][2])**2)
	visualizer.iterativeMDSwithProcrustes("raw_data/loo2/" + str(i) + ".csv", "Ensemble")
	ensDist = np.sqrt(np.subtract(visualizer.coordinates[i][0],visualizer.coordinates[-1][0])**2 + np.subtract(visualizer.coordinates[i][1],visualizer.coordinates[-1][1])**2 + np.subtract(visualizer.coordinates[i][2],visualizer.coordinates[-1][2])**2)
	outFile.write("{}	{}	{}	{}	{}\n".format(row, rfDist, adaDist, mlpDist, ensDist))

	i += 1
	#coordAda = visualizer.coordinates
outFile.close()