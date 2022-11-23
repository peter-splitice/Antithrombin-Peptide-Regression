"""
This script is in charge of handling the data analysis of the results after we handle bucket classification and regression.
"""

# Import the dependencies.
import numpy as np
import pandas as pd
import os
import csv

# Argument Parser
import argparse

# Importing plotter
import matplotlib.pyplot as plt

# Loggin
import logging
import sys

# Joblib
from joblib import dump, load

# Models
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import Lasso

def load_saved_clf():
    """
    This section runs the finalized classification portion of the data across the various model types to bucketize our data
        for inference.
        
        - SVC w/RBF Kernel w/SFS and PCA @80% variance. {'C': 61, 'break_ties': True, 'class_weight': None, 'gamma': 0.001},
            Test MCC = 0.529094, Train MCC = 0.713933, Threshold @ 10.  Large Bucket Size 20, Small Bucket Size 44, Extra Bucket Size
            9.

        - XGBoost Classifier w/SFS. {'alpha': 0.0, 'gamma': 2, 'lambda': 1, 'max_depth': 2, 'n_estimators': 11, 'subsample': 0.5},
            Test MCC = 0.661811, Train MCC = 0.709423, Threshold @ 0.01.  Large Bucket Size 46, Small Bucket Size 18, Extra Bucket Size
            9.

        - Random Forest Classifier w/SFS and PCA @85% variance.  {'ccp_alpha': 0.1, 'criterion': 'gini', 'max_depth': 9, 
            'max_features': 1.0, 'n_estimators': 7}, Test MCC = 0.614015, Train MCC = 0.729953, Threshold @ 10.  Large Bucket Size 20,
            Small Bucket Size 44, Extra Bucket Size 9.

        - KNN Classifier w/SFS and PCA @100% variance. {'leaf_size': 5, 'n_neighbors': 7, 'p': 2, 'weights': 'uniform'}, 
            Test MCC = 0.61151, Train MCC = 0.564734, Threshold @10.  Large Bucket Size 20, Small Bucket Size 44, Extra Bucket Size 9.

    Returns
    -------
    saved_clf: saved dict containing the models with tuned hyperparameters, thresholds, PCA variances, and names of our classification section.

    """
    # Create the models with the relevant hyperparameters.
    rbf = SVC(kernel='rbf', C=61, break_ties=True, class_weight=None, gamma=0.001)
    xgb = XGBClassifier(alpha=0.0, gamma=2, reg_lambda=1, max_depth=2, n_estimators=11, subsample=0.5)
    rfc = RandomForestClassifier(ccp_alpha=0.1, criterion='gini', max_depth=9, max_features=1.0, n_estimators=7)
    knn = KNeighborsClassifier(leaf_size=5, n_neighbors=7, p=2, weights='uniform')
    
    clfs = [rbf, xgb, rfc, knn]
    thresholds = [10, 0.01, 10, 10]
    variances = [80, False, 85, 100]
    names = ['SVC with RBF Kernel', 'XGBoost Classifier', 'Random Forest Classifier', 'KNN Classifier']

    saved_clf = list(zip(thresholds, variances, names, clfs))

    return saved_clf

# Loading regression models.
def load_regression_models():
    """
    This function will create two lists that have their own sets of models, params, and names.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    small: list for our small threshold models.  Contains lists including strings representing the name, the models, and a dict of hyperparameters.

    medium: list for our medium threshold models.  Contains lists including strings representing the name, the models, and a dict of hyperparameters.
    """
    rbf_params = {'gamma': ['scale', 'auto'], 'C': np.arange(1,101,5), 'epsilon': np.arange(0.1, 1, 0.1)}
    lin_params = {'gamma': ['scale', 'auto'], 'C': np.arange(1,101,5), 'epsilon': np.arange(0.1, 1, 0.1)}
    las_params = {'alpha': [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4], 'selection': ['cyclic', 'random']}
    
    # names and hyperparameters to sort through are shared.
    names = ['SVR with RBF Kernel', 'SVR with Linear Kernel', 'Lasso Regression']
    params = [rbf_params, lin_params, las_params]

    # I need to instantiate new models for both the small and medium buckets.
    sml_models = [SVR(kernel='rbf'), SVR(kernel='linear'), Lasso()]
    med_models = [SVR(kernel='rbf'), SVR(kernel='linear'), Lasso()]

    # Create the list.
    reg_models = list(zip(names, params, sml_models, med_models))

    return reg_models

# Path
def results_import():

    path = os.getcwd()
    ## Note that the features selected here were done based on model.
    saved_clf = load_saved_clf()
    reg_names = ['SVR with RBF Kernel', 'SVR with Linear Kernel', 'Lasso Regression']

    for threshold, var_clf, name_clf, clf in saved_clf:
        features_df = pd.read_csv(path + '/SFS Extracted Features/Saved Features for Threshold %2.2f.csv' %(threshold))
        extracted_features = features_df['%s' %(name_clf)]
        sfs = load(path + '/%s/sfs/%s %2.2f fs.joblib' %(name_clf, name_clf, threshold))
        extracted_features = sfs.get_feature_names_out()

        # Information for the validation sets:
        for reg in reg_names:

            # Load Training Sets
            train_fold1 = pd.read_csv(path + '/%s/%s/Training Predictions Fold 1' %(name_clf, reg))
            train_fold2 = pd.read_csv(path + '/%s/%s/Training Predictions Fold 2' %(name_clf, reg))
            train_fold3 = pd.read_csv(path + '/%s/%s/Training Predictions Fold 3' %(name_clf, reg))
            train_fold4 = pd.read_csv(path + '/%s/%s/Training Predictions Fold 4' %(name_clf, reg))
            train_fold5 = pd.read_csv(path + '/%s/%s/Training Predictions Fold 5' %(name_clf, reg))
            
            

            # Validation Sets
            valid_fold1 = pd.read_csv(path + '/%s/%s/Validation Predictions Fold 1' %(name_clf, reg))
            valid_fold2 = pd.read_csv(path + '/%s/%s/Validation Predictions Fold 2' %(name_clf, reg))
            valid_fold3 = pd.read_csv(path + '/%s/%s/Validation Predictions Fold 3' %(name_clf, reg))
            valid_fold4 = pd.read_csv(path + '/%s/%s/Validation Predictions Fold 4' %(name_clf, reg))
            valid_fold5 = pd.read_csv(path + '/%s/%s/Validation Predictions Fold 5' %(name_clf, reg))



    return 0


results_import()