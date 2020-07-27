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
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

class featureRegression:

	def __init__(self, fileTrainingSet):
		"""Creates features from alignments and antigenic distances"""
		self.__fileTrainingSet = fileTrainingSet
		self.__loadTrainingSet()
	
	def __loadTrainingSet(self):
		df = pd.read_csv(self.__fileTrainingSet, header=0, index_col=[0])

		self.trainY = df.iloc[:,2].values
		self.trainX = df.iloc[:,3:]
		
		
	def plotCorrelation(self, y_true, y_pred, subtitle = ""):
		"""Draw graph showing ytrue vs ypred"""
		fig = plt.figure()
		fig.suptitle('Real vs Pred Y ' + subtitle)#, fontsize=28, fontweight='bold')
		ax = fig.add_subplot(111)
		ax.set_xlabel('Real Y')
		ax.set_ylabel('Predicted Y')
		plt.scatter(y_true, y_pred, color='red', s=1)
		plt.show()
		return
	
	def plotSortedError(self, y_true, y_pred, subtitle = ""):
		"""Draw graph showing total error over training set"""
		error = list(abs(y_true - y_pred))
		error.sort()
		fig = plt.figure()
		fig.suptitle('Real vs Pred Y ' + subtitle)#, fontsize=28, fontweight='bold')
		ax = fig.add_subplot(111)
		ax.set_xlabel('Real Y')
		ax.set_ylabel('Predicted Y')
		plt.scatter(range(0,len(error)), error, color='red', s=1)
		plt.show()
		return
	
	def trainRandomForestRegressor(self, estimators = 319, depth = 182, features = 311):
		self.randomForestRegr = RandomForestRegressor(n_estimators = estimators, max_depth = depth, max_features = features, random_state = 777) #Locking seed for reproducibility
		self.randomForestRegr.fit(self.trainX, self.trainY)		

	def trainAdaBoostedRegressor(self, depth = 442, features = 476, estimators = 221):
		self.adaBoostedRegr = AdaBoostRegressor(DecisionTreeRegressor(max_depth= depth, max_features = features), n_estimators=estimators, random_state = 777)
		self.adaBoostedRegr.fit(self.trainX, self.trainY)
		
	def trainMultilayerPerceptron(self, hidden_layers = 364, iterations = 300):
		self.multilayerPerceptronRegr = MLPRegressor(hidden_layer_sizes = hidden_layers, max_iter = iterations, random_state = 777)
		self.multilayerPerceptronRegr.fit(self.trainX, self.trainY)
	
	def getCV(self, folds = 10):
		scores = cross_val_score(self.randomForestRegr, self.trainX, self.trainY, scoring="neg_root_mean_squared_error", cv = folds)
		self.rfCV = scores
		print("RF RMSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
		scores = cross_val_score(self.adaBoostedRegr, self.trainX, self.trainY, scoring="neg_root_mean_squared_error", cv = folds)
		self.adaCV = scores
		print("Ada RMSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
		scores = cross_val_score(self.multilayerPerceptronRegr, self.trainX, self.trainY, scoring="neg_root_mean_squared_error", cv = folds)
		self.mlpCV = scores
		print("MLP RMSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

	def getEnsembleCV(self, folds = 10):
		#CV for ensemble
		ensembleRegr = VotingRegressor([('rf', self.randomForestRegr), ('ada', self.adaBoostedRegr), ('mlp', self.multilayerPerceptronRegr)])
		scores = cross_val_score(ensembleRegr, self.trainX, self.trainY, scoring="neg_root_mean_squared_error", cv = folds)
		self.enseCV = scores
		print("Ensemble RMSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	
	def tuneRandomForestRegressor(self):
		"""experiment specific function"""
		parameters = {"n_estimators": [500]}
		tuneRf = GridSearchCV(self.randomForestRegr, parameters, cv = 5, n_jobs = 6)
		tuneRf.fit(self.trainX, self.trainY)
		#print(tuneRf.best_params_)
		print(tuneRf.best_score_)
		self.randomForestRegr = tuneRf.best_estimator_

	def tuneAdaBoostedRegressor(self):
		"""experiment specific function"""
		parameters = {"n_estimators": [240]}
		tuneAda = GridSearchCV(self.adaBoostedRegr, parameters, cv = 5, n_jobs = 6)
		tuneAda.fit(self.trainX, self.trainY)
		#print(tuneAda.best_params_)
		print(tuneAda.best_score_)
		self.adaBoostedRegr = tuneAda.best_estimator_
		
	def tuneMultilayerPerceptron(self):
		"""experiment specific function"""
		parameters = {"hidden_layer_sizes": [200]}
		tuneMLP = GridSearchCV(self.multilayerPerceptronRegr, parameters, cv = 5, n_jobs = 6)
		tuneMLP.fit(self.trainX, self.trainY) 
		#print(tuneMLP.best_params_)
		print(tuneMLP.best_score_)
		self.multilayerPerceptronRegr = tuneMLP.best_estimator_
		
	def printEvaluationMetrics(self):
		y_predRf = self.randomForestRegr.predict(self.trainX)
		y_predAda = self.adaBoostedRegr.predict(self.trainX)
		y_predMlp = self.multilayerPerceptronRegr.predict(self.trainX)
		y_predEnsemble = (y_predRf + y_predAda + y_predMlp)/3
		
		print("\tRF\tAda\tMLP\tEnsemble")
		print("Pearson Corr\t"+ str(r2_score(self.trainY, y_predRf)) + "\t" + str(r2_score(self.trainY, y_predAda)) + "\t" + str(r2_score(self.trainY, y_predMlp)) + "\t" + str(r2_score(self.trainY, y_predEnsemble)))
		print("MAE\t" + str(mean_absolute_error(self.trainY, y_predRf)) + "\t" + str(mean_absolute_error(self.trainY, y_predAda)) + "\t" + str(mean_absolute_error(self.trainY, y_predMlp)) + "\t" +  str(mean_absolute_error(self.trainY, y_predEnsemble)))
		print("MSE\t" + str(mean_squared_error(self.trainY, y_predRf)) + "\t" + str(mean_squared_error(self.trainY, y_predAda)) + "\t" + str(mean_squared_error(self.trainY, y_predMlp)) + "\t" +  str(mean_squared_error(self.trainY, y_predEnsemble)))

	def printSplitTest(self, seed = 777):
		#https://www.blopig.com/blog/2017/07/using-random-forests-in-python-with-scikit-learn/
		from sklearn.model_selection import train_test_split
		X_train, X_test, y_train, y_test = train_test_split(self.trainX, self.trainY, train_size=0.8, random_state=seed)
		rf = RandomForestRegressor(n_estimators=319, oob_score=True, max_depth = 182, max_features = 311, random_state=seed)
		rf.fit(X_train, y_train)
		ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth= 442, max_features = 476), n_estimators=221, random_state = seed)
		ada.fit(X_train, y_train)
		mlp = MLPRegressor(hidden_layer_sizes = 364, max_iter = 300, random_state = seed)
		mlp.fit(X_train, y_train)
		from scipy.stats import spearmanr, pearsonr
		predicted_train = rf.predict(X_train)
		predicted_test = rf.predict(X_test)
		test_score = r2_score(y_test, predicted_test)
		spearman = spearmanr(y_test, predicted_test)
		pearson = pearsonr(y_test, predicted_test)
		mse = mean_squared_error(y_test, predicted_test)
		print('############# RF #############')
		print("Out-of-bag R-2 score estimate: " + str(rf.oob_score_))
		print("Test data R-2 score: " + str(test_score))
		print("Test data Spearman correlation: " + str(spearman[0]))
		print("Test data Pearson correlation: " + str(pearson[0]))
		print("RMSE: " + str(mse))
				
		predicted_test = ada.predict(X_test)
		test_score = r2_score(y_test, predicted_test)
		spearman = spearmanr(y_test, predicted_test)
		pearson = pearsonr(y_test, predicted_test)
		mse = mean_squared_error(y_test, predicted_test, squared = False) #grabs RMSE, allowing us to stay in the original AU
		print('############# ADA ############')
		print("Test data R-2 score: " + str(test_score))
		print("Test data Spearman correlation: " + str(spearman[0]))
		print("Test data Pearson correlation: " + str(pearson[0]))
		print("RMSE: " + str(mse))
		
		predicted_test = mlp.predict(X_test)
		test_score = r2_score(y_test, predicted_test)
		spearman = spearmanr(y_test, predicted_test)
		pearson = pearsonr(y_test, predicted_test)
		mse = mean_squared_error(y_test, predicted_test, squared = False) #grabs RMSE, allowing us to stay in the original AU
		print('############# MLP ############')
		print("Test data R-2 score: " + str(test_score))
		print("Test data Spearman correlation: " + str(spearman[0]))
		print("Test data Pearson correlation: " + str(pearson[0]))
		print("RMSE: " + str(mse))
		
		#Add in silly overhead
		predicted1 = rf.predict(X_test)
		predicted2 = ada.predict(X_test)
		predicted3 = mlp.predict(X_test)
		predicted_test = (predicted1 + predicted2 + predicted3)/3
		test_score = r2_score(y_test, predicted_test)
		spearman = spearmanr(y_test, predicted_test)
		pearson = pearsonr(y_test, predicted_test)
		mse = mean_squared_error(y_test, predicted_test, squared = False) #grabs RMSE, allowing us to stay in the original AU
		print('############# ENSEMBLE ############')
		print("Test data R-2 score: " + str(test_score))
		print("Test data Spearman correlation: " + str(spearman[0]))
		print("Test data Pearson correlation: " + str(pearson[0]))
		print("RMSE: " + str(mse))
				
		
	def visualizeFit(self):
		y_predRf = self.randomForestRegr.predict(self.trainX)
		y_predAda = self.adaBoostedRegr.predict(self.trainX)
		y_predMlp = self.multilayerPerceptronRegr.predict(self.trainX)
		y_predEnsemble = (y_predRf + y_predAda + y_predMlp)/3
		
		plt.figure()
		plt.plot(self.trainY, y_predRf, "g+", label='RandomForestRegressor')
		plt.ylabel('predicted')
		plt.xlabel('real')
		plt.legend(loc="best")
		plt.title('RF')
		plt.show()
		print(np.corrcoef(y_predRf, self.trainY)[0, 1])
		
		plt.figure()
		plt.plot(self.trainY, y_predAda, "g+", label='RandomForestRegressor')
		plt.ylabel('predicted')
		plt.xlabel('real')
		plt.legend(loc="best")
		plt.title('Ada')
		plt.show()
		print(np.corrcoef(y_predAda, self.trainY)[0, 1])
		
		plt.figure()
		plt.plot(self.trainY, y_predMlp, "g+", label='RandomForestRegressor')
		plt.ylabel('predicted')
		plt.xlabel('real')
		plt.legend(loc="best")
		plt.title('MLP')
		plt.show()
		print(np.corrcoef(y_predMlp, self.trainY)[0, 1])
		
		plt.figure()
		plt.plot(self.trainY, y_predEnsemble, "g+", label='RandomForestRegressor')
		plt.ylabel('predicted')
		plt.xlabel('real')
		plt.legend(loc="best")
		plt.title('Ensemble')
		plt.show()
		print(np.corrcoef(y_predEnsemble, self.trainY)[0, 1])

	def visualizePerformance(self):
		y_predRf = self.randomForestRegr.predict(self.trainX)
		y_predAda = self.adaBoostedRegr.predict(self.trainX)
		y_predMlp = self.multilayerPerceptronRegr.predict(self.trainX)
		y_predEnsemble = (y_predRf + y_predAda + y_predMlp)/3
		
		sortedPred = pd.DataFrame(np.zeros((len(y_predRf), 4)),columns = ['Random Forest','AdaBoosted','Multilayer Perceptron','Ensemble'])
		sortedPred['Random Forest'] = y_predRf
		sortedPred['AdaBoosted'] = y_predAda
		sortedPred['MLP Network'] = y_predMlp
		sortedPred['Ensemble'] = y_predEnsemble
		sortedPred.sort_values('AdaBoosted', inplace = True)
		sortedPred.reset_index(inplace = True)
		
		plt.figure()
		plt.plot(sortedPred['Random Forest'], 'gd', label='RandomForestRegressor')
		plt.plot(sortedPred['AdaBoosted'], 'b^', label='AdaBoostedRegressor')
		plt.plot(sortedPred['MLP Network'], 'o', label='MultilayerPerceptronRegressor')
		plt.plot(sortedPred['Ensemble'], 'r*', label='MultilayerPerceptronRegressor')
		plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
		plt.ylabel('predicted')
		plt.xlabel('training samples')
		plt.legend(loc="best")
		plt.title('Comparison of individual predictions with averaged')
		plt.show()
		
	def predictUnknownSet(self, fileTestSet, predictAgainstTest = False):
		unknownSet = pd.read_csv(fileTestSet, header=0, index_col=[0])
		unknownX = unknownSet.iloc[:,3:]
		
		#Predict
		rfPred = self.randomForestRegr.predict(unknownX)
		adaPred = self.adaBoostedRegr.predict(unknownX)
		mlpPred = self.multilayerPerceptronRegr.predict(unknownX)
		
		#Push to dataframe
		unknownDf = pd.DataFrame(np.zeros((len(unknownSet), 2)),columns = ['antigen','antiserum'])
		unknownDf['antigen'] = unknownSet['antigen']
		unknownDf['antiserum'] = unknownSet['antiserum']
		unknownDf['Random Forest'] = rfPred
		unknownDf['AdaBoosted'] = adaPred
		unknownDf['MLP Network'] = mlpPred
		unknownDf['Ensemble'] = (rfPred + adaPred + mlpPred) / 3
		
		self.unknownSet = unknownDf
		
	def exportUnknownSet(self, outputFile):
		#Write out to file
		self.unknownSet.to_csv(outputFile)
		
	def visualizePredictions(self):
		"""Visualize consistancy of predictions"""
		self.unknownSet.sort_values('Ensemble', inplace = True)
		self.unknownSet.reset_index(inplace = True)
		plt.figure()
		plt.plot(self.unknownSet['Random Forest'], 'gd', label='RandomForestRegressor')
		plt.plot(self.unknownSet['AdaBoosted'], 'b^', label='AdaBoostedRegressor')
		plt.plot(self.unknownSet['MLP Network'], 'o', label='MultilayerPerceptronRegressor')
		plt.plot(self.unknownSet['Ensemble'], 'o', label='Ensemble')
		plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
		plt.ylabel('predicted')
		plt.xlabel('training samples')
		plt.legend(loc="best")
		plt.title('Comparison of individual predictions with averaged')
		plt.show()
		
	def exportFeatureImportance(self, outFile):
		importance = list(self.randomForestRegr.feature_importances_)
		features = list(self.trainX.columns)
		dictFeatures = {'feature' : features, 'importance' : importance}
		df = pd.DataFrame(dictFeatures)
		df.to_csv(outFile)
		return