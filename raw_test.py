import features.rawDataProcess as fm
import features.featureRegression as fr
import features.mdsVisualization as mv
import pandas as pd
import numpy as np
from features import rawDataProcess

dp = fm.rawDataProcess()
dp.loadFoldChangeTable("raw_data/master_short.fasta", "raw_data/carine_corrected_titers.csv")
dp.defineFeatures()
dp.exportFeatures("raw_data/carine_feature_set.csv")
dp.createUnknownComparison("raw_data/retest_final.fasta")
dp.exportTestSetFeatures("raw_data/retestset_final.csv")

regressor = fr.featureRegression("raw_data/carine_feature_set.csv")
regressor.trainRandomForestRegressor()
regressor.trainAdaBoostedRegressor()
regressor.trainMultilayerPerceptron()
#regressor.printEvaluationMetrics()
#regressor.getEnsembleCV()
#regressor.getCV()
regressor.printSplitTest()
regressor.visualizePerformance()
regressor.predictUnknownSet("raw_data/retestset_final.csv")
regressor.exportUnknownSet("raw_data/pred_retest_final.csv")
regressor.exportFeatureImportance("rawImportance.csv")
#regressor.getCV()
regressor.visualizePredictions()



"""
***
"""

dp = fm.rawDataProcess()
#dp.loadFoldChangeTable("raw_data/master.fasta", "raw_data/marcus_final.csv")
dp.loadFoldChangeTable("raw_data/master_short.fasta", "raw_data/carine_corrected_titers.csv")
dp.defineFeatures()
dp.exportFeatures("raw_data/carine_feature_set.csv")
dp.createUnknownComparison("raw_data/screen/h3clusterIVA.fasta")
dp.exportTestSetFeatures("raw_data/civascreen.csv")

regressor = fr.featureRegression("raw_data/carine_feature_set.csv")
regressor.trainRandomForestRegressor()
regressor.trainAdaBoostedRegressor()
regressor.trainMultilayerPerceptron()
regressor.printEvaluationMetrics()
regressor.visualizePerformance()
regressor.predictUnknownSet("raw_data/civascreen.csv")
regressor.exportUnknownSet("raw_data/pred_civascreen.csv")
#regressor.exportFeatureImportance("rawImportance.csv")
#regressor.getCV()
regressor.visualizePredictions()




