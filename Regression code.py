import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingRegressor
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor

#Loading the Data
df_shear_train = pd.read_excel('X_train.xlsx', engine='openpyxl')
df_shear_test = pd.read_excel('X_test.xlsx', engine='openpyxl')

#Defining Y-data
Y_shear_train = df_shear_train["ael_shear_modulus_vrh"]
Y_shear_test = df_shear_test["ael_shear_modulus_vrh"]
Y_bulk_train = df_shear_train["ael_bulk_modulus_vrh"]
Y_bulk_test = df_shear_test["ael_bulk_modulus_vrh"]

#Defining X-data and dropping Y-values and categroical variabels
df_shear_train_X = df_shear_train.drop(["Counter","ael_shear_modulus_reuss","ael_shear_modulus_voigt","ael_shear_modulus_vrh","compound","ael_shear_modulus_reuss","ael_bulk_modulus_reuss","ael_bulk_modulus_voigt","ael_bulk_modulus_vrh","auid","aurl","spacegroup_relax","Pearson_symbol_relax"], axis = "columns")
df_shear_test_X = df_shear_test.drop(["Counter","ael_shear_modulus_reuss","ael_shear_modulus_voigt","ael_shear_modulus_vrh","compound","ael_shear_modulus_reuss","ael_bulk_modulus_reuss","ael_bulk_modulus_voigt","ael_bulk_modulus_vrh","auid","aurl","spacegroup_relax","Pearson_symbol_relax"], axis = "columns")

#Filtering features through VarianceThreshold-filtering
var_thres = VarianceThreshold(threshold=0.01)
var_thres.fit(df_shear_train_X)
var_thres.get_support()
constant_columns = [column for column in df_shear_train_X.columns if column not in df_shear_train_X.columns[var_thres.get_support()]]
After_Variance = df_shear_train_X.drop(constant_columns,axis=1)

#performing pearson correlation and dropping features with a correlation factor < 0,75
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[j]  # getting the name of column
                col_corr.add(colname)
    af_corr = dataset.drop(col_corr,axis=1)
    return af_corr

af_both = correlation(After_Variance, .75)

af_both = af_both.drop(columns=['Unnamed: 289', 'Unnamed: 290'])#Dropping unidexed features

#Defining the Xs
df_shear_train_X = df_shear_train_X[af_both.columns]
df_shear_test_X = df_shear_test_X[af_both.columns]

#Crossval data
df_shear_data = pd.concat([df_shear_test_X, df_shear_train_X])
Y_shear_data = pd.concat([Y_shear_test, Y_shear_train])
Y_bulk_data = pd.concat([Y_bulk_test, Y_bulk_train])

#######################################################################################################################################################################################################
#df_shear_train_X and Y_shear/bulk_train used for training a model. This was used for predicting values based on the test set df _shear_test_X. Predicted Values were plotted against the real values.#
#For Validation the whole data set was used to get the best possible result                                                                                                                           #
#######################################################################################################################################################################################################

##############
#Linear model#
##############

#model for shear regression
LinMod_shear = LinearRegression()
LinMod_shear = BaggingRegressor(estimator=LinMod_shear, n_estimators=10, random_state=0)
LinMod_shear.fit(df_shear_train_X, Y_shear_train)

#CV
LinMod_shear_c = LinearRegression()
LinMod_shear_c = BaggingRegressor(estimator=LinMod_shear_c, n_estimators=10, random_state=0)
R_shear_lin = cross_val_score(LinMod_shear_c, df_shear_data, Y_shear_data, cv=5)
Lin_score_shear = R_shear_lin.mean()

#model for bulk regression

LinMod_bulk = LinearRegression()
LinMod_bulk = BaggingRegressor(estimator=LinMod_bulk, n_estimators=10, random_state=0)
LinMod_bulk.fit(df_shear_train_X, Y_bulk_train)

#CV
LinMod_bulk_c = LinearRegression()
LinMod_bulk_c = BaggingRegressor(estimator=LinMod_bulk_c, n_estimators=10, random_state=0)
R_bulk_lin = cross_val_score(LinMod_bulk_c, df_shear_data, Y_bulk_data, cv=5)
Lin_score_bulk = R_bulk_lin.mean()

#################
#Nonlinear model#
#################

#model for shear

#grid search for best hyperparameters
SVR_shear = SVR(kernel="poly")
param = {"degree": [4, 5, 6]}
best_fit_shear = RandomizedSearchCV(SVR_shear, param, random_state=0)
search = best_fit_shear.fit(df_shear_train_X,Y_shear_train)
search.best_params_

#training model
SVR_shear = SVR(kernel="poly", degree=4)
SVR_shear = BaggingRegressor(estimator=SVR_shear, n_estimators=10, random_state=0)
SVR_shear.fit(df_shear_train_X, Y_shear_train)

#CV
SVR_shear_c = SVR(kernel="poly", degree=4)
SVR_shear_c = BaggingRegressor(estimator=SVR_shear_c, n_estimators=10, random_state=0)
R_shear_SVR = cross_val_score(SVR_shear_c, df_shear_data, Y_shear_data, cv=5)
NLin_score_shear = R_shear_SVR.mean()

#model for bulk

#Grid search for bulk
SVR_shear = SVR(kernel="poly")
param = {"degree": [4, 5, 6]}
best_fit_shear = RandomizedSearchCV(SVR_shear, param, random_state=0)
search = best_fit_shear.fit(df_shear_train_X,Y_shear_train)
search.best_params_

#training model
SVR_bulk = SVR(kernel="poly", degree=4)
SVR_bulk = BaggingRegressor(estimator=SVR_bulk, n_estimators=10, random_state=0)
SVR_bulk.fit(df_shear_train_X, Y_bulk_train)

#CV
SVR_bulk_c = SVR(kernel="poly", degree=4)
SVR_bulk_c = BaggingRegressor(estimator=SVR_bulk_c, n_estimators=10, random_state=0)
R_bulk_SVR = cross_val_score(SVR_bulk_c, df_shear_data, Y_bulk_data, cv=5)
NLin_score_bulk = R_bulk_SVR.mean()

##################
#LASSO Regression#
##################

#model for shear

#Grid search for shear
Lasso_shear = linear_model.Lasso()
param = {"alpha": [0.001 ,0.01, 0.1, 0.5, 1]}
best_fit_shear = RandomizedSearchCV(Lasso_shear, param, random_state=0)
search = best_fit_shear.fit(df_shear_train_X,Y_shear_train)
search.best_params_

#training model
Lasso_shear = linear_model.Lasso(alpha=0.001)
Lasso_shear = BaggingRegressor(estimator=Lasso_shear, n_estimators=10, random_state=0)
Lasso_shear.fit(df_shear_train_X, Y_shear_train)

#CV
Lasso_shear_c = linear_model.Lasso(alpha=0.001)
Lasso_shear_c = BaggingRegressor(estimator=Lasso_shear_c, n_estimators=10, random_state=0)
R_shear_Lasso = cross_val_score(Lasso_shear_c, df_shear_data, Y_shear_data, cv=5)
Lasso_score_shear = R_shear_Lasso.mean()

#model for bulk

#Grid search for bulk
Lasso_bulk = linear_model.Lasso()
param = {"alpha": [0.001 ,0.01, 0.1, 0.5, 1]}
best_fit_shear = RandomizedSearchCV(Lasso_bulk, param, random_state=0)
search = best_fit_shear.fit(df_shear_train_X,Y_bulk_train)
search.best_params_

#Lasso Regressor for bulk
Lasso_bulk = linear_model.Lasso(alpha=0.01)
Lasso_bulk = BaggingRegressor(estimator=Lasso_bulk, n_estimators=10, random_state=0)
Lasso_bulk.fit(df_shear_train_X, Y_bulk_train)

#CV
Lasso_bulk_c = linear_model.Lasso(alpha=0.01)
Lasso_bulk_c = BaggingRegressor(estimator=Lasso_bulk_c, n_estimators=10, random_state=0)
R_bulk_Lasso = cross_val_score(Lasso_bulk_c, df_shear_data, Y_bulk_data, cv=5)
Lasso_score_bulk = R_bulk_Lasso.mean()

###############
#Authors model#
###############

#model for bulk

#Searching for best parameters
param = { "eta" : [0.05,0.1,0.2,0.5,1], "gamma" : [1,2,5,10,100] , "max_depth" : [5,8,11,15], "n_estimators": [10, 100, 500, 1000], "max_depth": [1,3,5,7,10]}
reg_xgb_orig = XGBRegressor()
best_fit_shear = RandomizedSearchCV(reg_xgb_orig, param, random_state=0)
search = best_fit_shear.fit(df_shear_train_X,Y_shear_train)
search.best_params_

#training model
reg_xgb = XGBRegressor(n_estimators= 500, max_depth= 5, gamma= 100, eta= 0.05)
ada_reg = AdaBoostRegressor(estimator=reg_xgb, random_state=0)
model_shear = ada_reg.fit(df_shear_train_X,Y_shear_train)

#CV
score = cross_val_score(ada_reg, df_shear_data, Y_shear_data, cv=5)
Ada_score_shear = score.mean()

#model for bulk

#Searching for best parameters
param = { "eta" : [0.05,0.1,0.2,0.5,1], "gamma" : [1,2,5,10,100] , "max_depth" : [5,8,11,15], "n_estimators": [10, 100, 500, 1000], "max_depth": [1,3,5,7,10]}
reg_xgb_orig = XGBRegressor()
best_fit_shear = RandomizedSearchCV(reg_xgb_orig, param, random_state=0)
search = best_fit_shear.fit(df_shear_train_X,Y_bulk_train)
search.best_params_

#training model
reg_xgb = XGBRegressor(n_estimators= 500, max_depth= 5, gamma= 100, eta= 0.05)
ada_reg = AdaBoostRegressor(estimator=reg_xgb, random_state=0)
model_shear = ada_reg.fit(df_shear_train_X,Y_bulk_train)

#CV
score = cross_val_score(ada_reg, df_shear_data, Y_bulk_data, cv=5)
Ada_score_bulk = score.mean()