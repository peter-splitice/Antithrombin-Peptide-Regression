"""
This portion of the pipeline is responsible for regression.  The pipeline for this will appear as follows:
    -> Split data into buckets
    -> Forward Selection
    -> PCA
    -> Models


"""

# Importing Dependencies
import pandas as pd
import numpy as np
import os
import csv

# Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler

# Models
from sklearn.linear_model import Lasso
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

# Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
 
# Metrics
from sklearn.metrics import matthews_corrcoef, mean_squared_error, accuracy_score, make_scorer

# Model Persistence
from joblib import dump, load

# Plotter
import matplotlib.pyplot as plt

# Argument Parser
import argparse

# Write to a log file
import logging
import sys

# Create a logger for various outputs.
def log_files(logname):
    """
    Create the meachanism for which we log results to a .log file.

    Parameters
    ----------
    logname:

    Returns
    -------
    logger:  The logger object we create to call on in other functions. 
    """

    # Instantiate the logger and set the formatting and minimum level to DEBUG.
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    # Display the logs in the output
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    # Write the logs to a file
    file_handler = logging.FileHandler(logname)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Adding the file and output handlers to the logger.
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    return logger

# Importing the datasets
def import_data(threshold):
    """
    Import the full dataset from the current path.  Also apply some of the necessary preprocessing.

    Parameters
    ----------
    threshold: Int value that determines where we split the data between small and medium buckets.

    Returns
    -------
    df:  Dataframe of the full KI training dataset, with any values above 50,000 removed.

    base_range:  Contains the range of values within the dataframe for rescaling purposes.

    """

    # Importing the full KI set into a dataframe.
    path = os.getcwd()
    df = pd.read_csv(path + '/PositivePeptide_Ki.csv')

    # Rescaling the dataframe in the log10 (-5,5) range.
    df['KI (nM) rescaled'], base_range  = rescale(df['KI (nM)'], destination_interval=(-5,5))

    # Create a colunmn in our dataframe to define the buckets.  I will be creating a series of models that classifies the buckets.
    df['Bucket'] = pd.cut(x=df['KI (nM)'], bins=(0, threshold, 4000, float('inf')), labels=(0,1,2))

    return df, base_range

# Rescaling the data into log scale.
def rescale(array=np.array(0), destination_interval=(-5,5)):
    """
    Rescale the KI values from nM to a log scale within the range of
        a given destination interval.

    Parameters
    ----------
    array:  A numpy array of KI values, in nM.

    destination_interval: the interval that we set the range of the log scale to

    Returns
    -------
    array:  Transformed array into the log scale.
    
    saved_range:  The (min, max) range of the original given array.  Used if we need
        to rescale back into "KI (nM)" form.
    
    """

    # Rescaling the values and saving the initial range.
    array = np.log(array)
    saved_range = (array.min(), array.max())
    array = np.interp(array, saved_range, destination_interval)

    return array, saved_range

# Scaling the data back into KI (nM) scale
def unscale(array, destination_interval, source_interval=(-5,5)):
    """
    Rescales an array of log-transformed values back into "KI (nM)" form.

    Parameters
    ----------
    array:  A numpy array of KI values in log-transformed form.

    destination_interval:  The original range of KI values.

    source_interval: The current range of KI log transformed values.  It'll default to -5,5

    Returns
    -------
    array:  A numpy array of the KI values back in the original format.

    """

    # Undoing the previous rescaling.
    array = np.interp(array, source_interval, destination_interval)
    array = np.exp(array)

    return array

# Optimization of hyperparameters for regression models using GridSearchCV
def hyperparameter_optimizer(x, y, params, model=SVR()):
    """
    This function is responsible for running GridSearchCV and opatimizing our hyperparameters.  I might need to fine-tune this.

    Parameters
    ----------
    x: Input values to perform GridSearchCV with.

    y: Output values to create GridSearchCV with.

    params: Dictionary of parameters to run GridSearchCV on.

    model: The model that we are using for GridSearchCV

    Returns
    -------
    bestvals: Optimzied hyperparameters for the model that we are running the search on.

    df: Pandas Dataframe that has the results of our hyperparameter tuning, sorted
        for results with the smallest standard deviation in the test scores.

    scores: Pandas DataFrame of the Training + Test Scores    
    """

    logger.debug('GridSearchCV Starting:')
    logger.debug('-----------------------\n')

    reg = GridSearchCV(model, param_grid=params, scoring='neg_root_mean_squared_error', cv=5, return_train_score=True,
                       n_jobs=-1)
    reg.fit(x,y)

    # Showing the best parameters found on the development set.
    logger.info('Best parameter set: %s' %(reg.best_params_))
    logger.info('-------------------------\n')

    # Save the best parameters.
    bestparams = reg.best_params_

    model.set_params(**bestparams)

    return model

# Loading saved classification models w/parameters.
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

# Inference for the classification + regression portion of the pipeline
def inference(x, y, buckets, ki_range, clf=SVC(), sml_reg=SVR(), med_reg=SVR()):

    # Training set
    # Bucketize
    buckets_actual = buckets[y.index]
    buckets_pred = clf.predict(x)

    # Make predictions for all of the buckets.  The large bucket we'll just predict as 0 for now.  Only make predictions if the arrays 
    #   aren't empty.
    
    if x[buckets_pred==0].size != 0:
        sml_pred = sml_reg.predict(x[buckets_pred==0])
    if x[buckets_pred==1].size !=0:
        med_pred = med_reg.predict(x[buckets_pred==1])
    lrg_pred = np.zeros(np.count_nonzero(x[buckets_pred==2]))

    # Put back the predictions in the original order.
    y_pred = np.array([])
    for i in buckets_pred:
        if i == 0:
            y_pred = np.append(y_pred, sml_pred[0])
            sml_pred = np.delete(sml_pred, 0)
        elif i == 1:
            y_pred = np.append(y_pred, med_pred[0])
            med_pred = np.delete(med_pred, 0)
        elif i == 2:
            y_pred = np.append(y_pred, lrg_pred[0])
            lrg_pred = np.delete(lrg_pred, 0)

    # Convert results back to KI (nM) scale from log scale.
    y_pred_unscaled = unscale(y_pred, ki_range)
    y_unscaled = unscale(y, ki_range)

    # Calculate RMSE metrics.
    train_rmse = mean_squared_error(y_unscaled, y_pred_unscaled)**0.5
    log_train_rmse = mean_squared_error(y, y_pred)**0.5

    # Save the results in a dataframe.
    cols = ['Log Y Actual', 'Log Y Predicted', 'Y Actual', 'Y Predicted', 'Actual Bucket', 'Predicted Bucket']
    df_data = zip(y, y_pred, y_unscaled, y_pred_unscaled, buckets_actual, buckets_pred)
    df = pd.DataFrame(data=df_data, columns=cols)

    return train_rmse, log_train_rmse, df

# Entire data pipeline for classification + regression dual model.
def clf_reg_pipeline(threshold, var, name_clf, clf=SVC()):
    """
    Code containing the pipeline for classification and regression.  All of the transformations are applied here as well.

    Parameters
    ----------
    threshold: threshold that we did the classification with.
    
    var: variance that we use for PCA, where applicable

    name: name of the classification model we are working with at the moment

    model: classification model

    """

    # Logging the model we are classifying with first.
    logger.info('Classification + Regression Pipeline using %s.\n' %(name_clf))

    # Data and path import.
    df, ki_range = import_data(threshold)
    path = os.getcwd()

    # Extract the x, y, and bucket information.
    x = df[df.columns[1:573]]
    y = df['KI (nM) rescaled']
    buckets = df['Bucket']

    # Apply MinMaxScaler to the initial x values.
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = pd.DataFrame(scaler.transform(x), columns=df.columns[1:573])

    # Sequential Feature Selection with the saved model.  Make sure to extract the features here too.
    sfs = load(path + '/%s/sfs/%s %2.2f fs.joblib' %(name_clf, name_clf, threshold))
    x = sfs.transform(x)
    fs_features = sfs.get_feature_names_out()

    # Where applicable, apply PCA tuning as well.
    if var != False:
        pca = load(path + '/%s/sfs-pca/%s %2.2f pca.joblib' %(name_clf, name_clf, threshold))
        x = pca.transform(x)

        # Dimensonality Reduction based on accepted variance.
        ratios = np.array(pca.explained_variance_ratio_)
        ratios = ratios[ratios.cumsum() <= (var/100)]
        
        # Readjust the dimensions of x based on the variance we want.
        length = len(ratios)
        x = x[:,0:length]

    # Load up the regression models here:
    reg_models = load_regression_models()

    ## Seed values and k-folding required variables.
    seeds = [33, 42, 55, 68, 74]
    folds = len(seeds)

    for name_reg, params, sml_reg, med_reg in reg_models:

        # Logger formatting
        logger.info('Model Results for %s:' %(name_reg))
        logger.info('---------------------\n')

        # Run the hyperparameter optimizer function to set the hyperparameters of the regression.
        logger.info('Hyperparameters for the small bucket regression model:')
        sml_reg = hyperparameter_optimizer(x, y, params, sml_reg)

        logger.info('Hyperparameters for the medium bucket regression model:')
        med_reg = hyperparameter_optimizer(x, y, params, med_reg)

        # Initialize metrics we are using.
        train_rmse_sum = 0
        train_rmse_log_sum = 0
        valid_rmse_sum = 0
        valid_rmse_log_sum = 0
        fold = 0

        # Create the necessary paths for result storage.
        if os.path.exists(path + '/%s/%s' %(name_clf, name_reg)) == False:
            os.mkdir('%s/%s' %(name_clf, name_reg))

        for seed in seeds:
            fold += 1

            # Create the training and validation sets.
            x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=(1/folds), random_state=seed, stratify=buckets)
            buckets_train = buckets[y_train.index]
            
            # Fitting the classification and regression models.
            clf.fit(x_train, buckets_train)
            sml_reg.fit(x_train[buckets_train==0], y_train[buckets_train==0])
            med_reg.fit(x_train[buckets_train==1], y_train[buckets_train==1])

            # Inference on the training and test sets.
            train_rmse, train_rmse_log, train_df = inference(x_train, y_train, buckets, ki_range, clf, sml_reg, med_reg)
            train_df.to_csv(path + '/%s/%s/Training Predictions Fold %i.csv' %(name_clf, name_reg, fold))

            valid_rmse, valid_rmse_log, valid_df = inference(x_valid, y_valid, buckets, ki_range, clf, sml_reg, med_reg)
            valid_df.to_csv(path + '/%s/%s/Validation Predictions Fold %i.csv' %(name_clf, name_reg, fold))

            # Log the individual folds
            logger.info('Training RMSE: %3.3f, Training RMSE (log): %3.3f, Validation RMSE: %3.3f, Validation RMSE (log): %3.3f,'
                        ' Fold: %i'  %(train_rmse, train_rmse_log, valid_rmse, valid_rmse_log, fold))

            train_rmse_sum += train_rmse
            train_rmse_log_sum += train_rmse_log
            valid_rmse_sum += valid_rmse
            valid_rmse_log_sum += valid_rmse_log
        
        train_rmse_avg = train_rmse_sum/folds
        train_rmse_log_avg = train_rmse_log_sum/folds
        valid_rmse_avg = valid_rmse_sum/folds
        valid_rmse_log_avg = valid_rmse_log_sum/folds

        logger.info('---------------------------------------------------------------------------------------------\n')
        logger.info('AVG Training RMSE: %3.3f, AVG training RMSE (log): %3.3f, AVG Validation RMSE: %3.3f, AVG Validation RMSE '
                    '(log): %3.3f\n' %(train_rmse_avg, train_rmse_log_avg, valid_rmse_avg, valid_rmse_log_avg))

# Main function that handles regression.
def regression():
    """
    This function is responsible for the initial phase of hyperparameter tuning for the regression section of the pipeline.
        The models used will be:
        -> SVR with RBF Kernel
        -> SVR with Linear Kernel
        -> Lasso Regression
        This is the first stage of hyperparameter tuning to be done before Forward Selection and PCA.

    """

    # Run the regression model without using the classification models?
    

    # Create the models with the relevant hyperparameters.
    saved_clf = load_saved_clf()

    # I'm going to run fitting and inference 4 different times, one for each cassifier.
    for threshold, var, name_clf, clf in saved_clf:
        clf_reg_pipeline(threshold, var, name_clf, clf)

# Argument parser section.
parser = argparse.ArgumentParser()
parser.add_argument('-reg', '--regressor', help='regressor = initial stage of hyperparmeter tuning for '
                    'various hyperparameters.', action='store_true')
args = parser.parse_args()

regressor = args.regressor

# Certain sections of the code will run depending on what we specify with the run script.
if regressor == True:
    logger = log_files('Regression_Log.log')
    regression()