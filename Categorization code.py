import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

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

#constructing the label set
Y_train = []
Y_test = []

for n in range(len(Y_bulk_train)):
    if Y_shear_train[n]/Y_bulk_train[n] > 0.57:
        Y_train.append(1)  #1 if material is brittle
    else:
        Y_train.append(0)  #0 if material is ductile

for n in range(len(Y_bulk_test)):
    if Y_shear_test[n]/Y_bulk_test[n] > 0.57:
        Y_test.append(1)  #1 if material is brittle
    else:
        Y_test.append(0)  #0 if material is ductile

#CV data
df_shear_data = pd.concat([df_shear_test_X, df_shear_train_X])
Y_data = pd.concat([pd.DataFrame(Y_test), pd.DataFrame(Y_train)])

#######################################################################################################################################################################################################
#df_shear_train_X and Y_shear/bulk_train used for training a model. This was used for predicting values based on the test set df _shear_test_X. Predicted Values were plotted against the real values.#
#For Validation the whole data set was used to get the best possible result                                                                                                                           #
#######################################################################################################################################################################################################

###############
#     KNN     #
###############

#gridsearch
Neigh = KNeighborsClassifier()
param = {"n_neighbors": [4, 5, 6, 7, 8, 9, 10]}
best_fit_shear = RandomizedSearchCV(Neigh, param, random_state=0)
search = best_fit_shear.fit(df_shear_train_X, Y_train)
search.best_params_

#training model
Neigh = KNeighborsClassifier(n_neighbors=3)
Neigh = BaggingClassifier(estimator=Neigh, n_estimators=10, random_state=0)
Neigh.fit(df_shear_train_X, Y_train)

#CV
Neigh_c = KNeighborsClassifier(n_neighbors=3)
Neigh_c = BaggingClassifier(estimator=Neigh_c, n_estimators=10, random_state=0)
R_Neigh = cross_val_score(Neigh_c, df_shear_data, Y_data, cv=5)
Neigh_score = R_Neigh.mean()

###################
#       SVM       #
###################

#training model
SVC_mod = LinearSVC(random_state=0, C=1.7)
SVC_mod = BaggingClassifier(estimator=SVC_mod, n_estimators=10, random_state=0)
SVC_mod.fit(df_shear_train_X, Y_train)

#CV
SVC_mod_c = LinearSVC(random_state=0, C=1.7)
SVC_mod_c = BaggingClassifier(estimator=SVC_mod_c, n_estimators=10, random_state=0)
R_SVC = cross_val_score(SVC_mod_c, df_shear_data, Y_data, cv=5)
SVC_score = R_SVC.mean()

###################
#  Random Forrest #
###################

#grid search
RndF = RandomForestClassifier()
param = {"max_depth": [5,6,7,8,9,10]}
best_fit_shear = RandomizedSearchCV(RndF, param, random_state=0)
search = best_fit_shear.fit(df_shear_train_X, Y_train)
search.best_params_

#training model
RndF = RandomForestClassifier(max_depth=10, random_state=0)
RndF = BaggingClassifier(estimator= RndF, n_estimators=10, random_state=0)
RndF.fit(df_shear_train_X, Y_train)

#CV
RndF_c = RandomForestClassifier(max_depth=10, random_state=0)
RndF_c = BaggingClassifier(estimator=RndF_c, n_estimators=10, random_state=0)
R_SVC = cross_val_score(RndF_c, df_shear_data, Y_data, cv=5)
RndF_score = R_SVC.mean()