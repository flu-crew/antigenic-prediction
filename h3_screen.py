#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 10:45:42 2019

@author: michael.zeller
"""

import features.rawDataProcess as fm
import features.featureRegression as fr
import features.mdsVisualization as mv
import pandas as pd
import numpy as np
from features import rawDataProcess

#Training set
dp = fm.rawDataProcess()
dp.loadFoldChangeTable("raw_data/master_short.fasta", "raw_data/carine_final.csv")
dp.defineFeatures()
dp.exportFeatures("raw_data/carine_feature_set.csv")
dp.createTestSet("raw_data/screen/h3clusterI.fasta", selfCompare = False)
dp.exportTestSetFeatures("raw_data/screen/screen_h3clusterI.csv")
dp.createTestSet("raw_data/screen/h3clusterIV.fasta", selfCompare = False)
dp.exportTestSetFeatures("raw_data/screen/screen_h3clusterIV.csv")
dp.createTestSet("raw_data/screen/h3clusterIVA.fasta", selfCompare = False)
dp.exportTestSetFeatures("raw_data/screen/screen_h3clusterIVA.csv")
dp.createTestSet("raw_data/screen/h3clusterIVB.fasta", selfCompare = False)
dp.exportTestSetFeatures("raw_data/screen/screen_h3clusterIVB.csv")
dp.createTestSet("raw_data/screen/h32010.1.fasta", selfCompare = False)
dp.exportTestSetFeatures("raw_data/screen/screen_h32010.1.csv")
dp.createTestSet("raw_data/screen/h32010.2.fasta", selfCompare = False)
dp.exportTestSetFeatures("raw_data/screen/screen_h32010.2.csv")
dp.createTestSet("raw_data/screen/hutoswine.fasta", selfCompare = False)
dp.exportTestSetFeatures("raw_data/screen/screen_hutoswine.csv")

regressor = fr.featureRegression("raw_data/carine_feature_set.csv")
regressor.trainRandomForestRegressor()
regressor.trainAdaBoostedRegressor()
regressor.trainMultilayerPerceptron()
regressor.printEvaluationMetrics()
regressor.visualizePerformance()
#regressor.exportFeatureImportance("importances.csv")
regressor.predictUnknownSet("raw_data/screen/screen_h3clusterI.csv")
regressor.exportUnknownSet("raw_data/screen/screen_h3clusterI_pred.csv")
regressor.predictUnknownSet("raw_data/screen/screen_h3clusterIV.csv")
regressor.exportUnknownSet("raw_data/screen/screen_h3clusterIV_pred.csv")
regressor.predictUnknownSet("raw_data/screen/screen_h3clusterIVA.csv")
regressor.exportUnknownSet("raw_data/screen/screen_h3clusterIVA_pred.csv")
regressor.predictUnknownSet("raw_data/screen/screen_h3clusterIVB.csv")
regressor.exportUnknownSet("raw_data/screen/screen_h3clusterIVB_pred.csv")
regressor.predictUnknownSet("raw_data/screen/screen_h32010.1.csv")
regressor.exportUnknownSet("raw_data/screen/screen_h32010.1_pred.csv")
regressor.predictUnknownSet("raw_data/screen/screen_h32010.2.csv")
regressor.exportUnknownSet("raw_data/screen/screen_h32010.2_pred.csv")
regressor.predictUnknownSet("raw_data/screen/screen_hutoswine.csv")
regressor.exportUnknownSet("raw_data/screen/screen_hutoswine_pred.csv")
regressor.printSplitTest(seed=777)
regressor.visualizePredictions()