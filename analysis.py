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

# Models
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

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
    
    models = [rbf, xgb, rfc, knn]
    thresholds = [10, 0.01, 10, 10]
    variances = [80, False, 85, 100]
    names = ['SVC with RBF Kernel', 'XGBoost Classifier', 'Random Forest Classifier', 'KNN Classifier']

    saved_clf = list(zip(thresholds, variances, names, models))

    return saved_clf

# Path
def results_import():
    ## Note that the features selected here were done based on model.
    return 0
