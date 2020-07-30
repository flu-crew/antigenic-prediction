#Imports
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn import svm
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score

#Load training data
df = pd.read_csv("raw_data/carine_feature_set.csv", header=0, index_col=[0])

trainY = df.iloc[:,2].values
trainX = df.iloc[:,3:]

#CV check
def getCV(regr, folds = 5):
	scores = cross_val_score(regr, trainX, trainY, scoring="neg_root_mean_squared_error", cv = folds)
	print("Regressor MSE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        
#identity to antigenic dist base case
baseReg = LinearRegression()
baseReg.fit(np.reshape(trainX.identity.values, (-1,1)), trainY)
getCV(baseReg)
print(baseReg.score(np.reshape(trainX.identity.values, (-1,1)), trainY))

#Linear Reg
lineReg = LinearRegression()
lineReg.fit(trainX, trainY)
getCV(lineReg)
print(lineReg.score(trainX, trainY))

#Linear w/ Lasso
lassoReg = linear_model.Lasso(alpha=0.006)
lassoReg.fit(trainX, trainY)
getCV(lassoReg)
print(lassoReg.score(trainX, trainY))

#Linear w/ Ridge
ridgeReg = Ridge(alpha=11.0)
ridgeReg.fit(trainX, trainY)
getCV(ridgeReg)
print(ridgeReg.score(trainX, trainY))

#Linear w/ Elastic
elasticReg = ElasticNet(random_state=0)
elasticReg.fit(trainX, trainY)
getCV(elasticReg)
print(elasticReg.score(trainX, trainY))

#SVM Regression linear kernel - R2 better than RBF kernel, but lower CV5
svmReg = svm.SVR(kernel='linear')
svmReg.fit(trainX, trainY)
getCV(svmReg)
print(svmReg.score(trainX, trainY))

#SVM regression rbf kernel
svmReg = svm.SVR(kernel='rbf')
svmReg.fit(trainX, trainY)
getCV(svmReg)
print(svmReg.score(trainX, trainY))

#Ada boosted decisions tree
from sklearn.tree import DecisionTreeRegressor
adaReg = AdaBoostRegressor(DecisionTreeRegressor(max_depth= 442, max_features = 476), n_estimators=221)	#optimized
adaReg.fit(trainX, trainY)
getCV(adaReg)
print(adaReg.score(trainX, trainY))

#Random Forest
rfReg = RandomForestRegressor()
rfReg.fit(trainX, trainY)
getCV(rfReg)
print(rfReg.score(trainX, trainY))

#Multilayer Perceptron
mlpReg = MLPRegressor(max_iter=500)
mlpReg.fit(trainX, trainY)
getCV(mlpReg)
print(mlpReg.score(trainX, trainY))



#Test
from yellowbrick.regressor import ResidualsPlot
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.2, random_state=42)


visualizer = ResidualsPlot(baseReg, size=(1080, 1000))
visualizer.fit(np.reshape(X_train.identity.values, (-1,1)), y_train)
visualizer.score(np.reshape(X_test.identity.values, (-1,1)), y_test)
visualizer.show()

visualizer = ResidualsPlot(lineReg)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

visualizer = ResidualsPlot(lassoReg)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

visualizer = ResidualsPlot(ridgeReg)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

visualizer = ResidualsPlot(elasticReg)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

visualizer = ResidualsPlot(svmReg,size=(1080, 720))
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

visualizer = ResidualsPlot(adaReg,size=(1080, 720))
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

visualizer = ResidualsPlot(rfReg,size=(1080, 720))
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

visualizer = ResidualsPlot(mlpReg,size=(1080, 720))
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()


from yellowbrick.regressor import AlphaSelection
from sklearn.linear_model import RidgeCV
model = AlphaSelection(RidgeCV())
model.fit(X_train, y_train)
model.show()

