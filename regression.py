"""
This portion of the pipeline is responsible for regression.  The pipeline for this will appear as follows:
    -> Split data into buckets
    -> Forward Selection
    -> PCA
    -> Models


"""

## Importing Dependencies

# Standard libraries
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

## Create the logger
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

## Import the complete dataset.
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

    ## Logarithmically scalling the values.
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

## Inverse of the rescale function to rescale the outputs.
def unscale(array, destination_interval, source_interval=(-5,5)):
    """
    Rescales an array of log-transformed values back into "KI (nM)" form.

    Parameters
    ----------
    array:  A numpy array of KI values in log-transformed form.

    destination_interval:  The original range of KI values.

    source_interval: The current range of KI log transformed values.

    Returns
    -------
    array:  A numpy array of the KI values back in the original format.

    """

    # Undoing the previous rescaling.
    array = np.interp(array, source_interval, destination_interval)
    array = np.exp(array)

    return array

def hyperparameter_optimizer(x, y, params, model=SVR()):
    """
    This function is responsible for running GridSearchCV and opatimizing our hyperparameters.  I might need to fine-tune this.

    Parameters
    ----------
    x: Input values to perform GridSearchCV with.

    y: Output values to create GridSearchCV with.
    
    buckets: Might need this.  This contains the bucket info of the data.

    params: Dictionary of parameters to run GridSearchCV on.

    model: The model that we are using for GridSearchCV

    Returns
    -------
    bestvals: Optimzied hyperparameters for the model that we are running the search on.

    df: Pandas Dataframe that has the results of our hyperparameter tuning, sorted
        for results with the smallest standard deviation in the test scores.

    scores: Pandas DataFrame of the Training + Test Scores    
    """

    logger.info('GridSearchCV Starting:\n')
    reg = GridSearchCV(model, param_grid=params, scoring='neg_root_mean_squared_error', cv=5, return_train_score=True,
                       n_jobs=-1)
    reg.fit(x,y)

    # Showing the best parameters found on the development set.
    logger.info('Best parameter set: %s' %(reg.best_params_))
    logger.info('-------------------------\n')

    # Testing on the development set.  Save the results to a pandas dataframe and then sort it by
    # standard deviation of the test set.
    df = pd.DataFrame(reg.cv_results_)
    index = reg.best_index_
    scores = [df['mean_train_score'][index], df['std_train_score'][index], reg.best_score_, df['std_test_score'][index], reg.best_params_]
    logger.info('Train RMSE Score: %3.3f.  StDev for Train RMSE: %3.3f.  Test RMSE Score: %3.3f.  StDev for Test RMSE: %3.3f.\n' 
                %(scores[0], scores[1], scores[2], scores[3]))

    df = df.sort_values(by=['std_test_score'])

    # Clean up the output for the hyperparameters.  Eliminate any values that have too low of a test ranking
    #   as well as eliminate anything with too high of a training score.
    max_test_rank = df['rank_test_score'].max()
    col_start = 'split0_train_score'
    index_start = df.columns.get_loc(col_start)
    df = df[~(df.iloc[:,index_start:]>0.98).any(1)]
    df = df[df['mean_train_score'] > 0.65]
    df = df[df['rank_test_score'] < (0.20*max_test_rank)]
    df = df[df['mean_test_score'] > 0.25]

    # Save the best parameters.
    bestparams = reg.best_params_

    return bestparams, df, scores

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


    """
    # Create the models with the relevant hyperparameters.
    rbf = SVC(kernel='rbf', C=61, break_ties=True, class_weight=None, gamma=0.001)
    rbf_threshold = 10
    rbf_var = 80

    xgb = XGBClassifier(alpha=0.0, gamma=2, reg_lambda=1, max_depth=2, n_estimators=11, subsample=0.5)
    xgb_threshold = 0.01
    xgb_var = False         # Variance of 'False' indicates that we will not be 

    rfc = RandomForestClassifier(ccp_alpha=0.1, criterion='gini', max_depth=9, max_features=1.0, n_estimators=7)
    rfc_threshold = 10
    rfc_var = 85

    knn = KNeighborsClassifier(leaf_size=5, n_neighbors=7, p=2, weights='uniform')
    knn_threshold = 10
    knn_var = 100

    # Put the model information in 4 different list.  Models, Thresholds, Variances, and Names
    models = [rbf, xgb, rfc, knn]
    thresholds = [rbf_threshold, xgb_threshold, rfc_threshold, knn_threshold]
    vars = [rbf_var, xgb_var, rfc_var, knn_var]
    names = ['SVC with RBF Kernel', 'XGBoost Classifier', 'Random Forest Classifier', 'KNN Classifier']

    saved_clf = zip(models,thresholds,vars, names)
    return saved_clf

def fit_and_inference(x, y, buckets, params, reg_threshold, bucket_name, model=SVR()):
    """
    Perform fitting on the optimized models and then make predictions.  Output values are in the log file.

    """

    ## Model Building
    # Initialized with seed and counter variable
    seeds = [33, 42, 55, 68, 74]
    i = 0
    folds = len(seeds)
    path = os.getcwd()

    # If Bucket Name is 'small' or 'medium', we will do the fit/inference accordingly using a bucketflag.
    if bucket_name == 'Small':
        bucketflag = 0
    elif bucket_name == 'Medium':
        bucketflag = 1

    # Initialize metrics are using.
    train_accuracy_sum = 0
    train_rmse_sum = 0
    valid_accuracy_sum = 0
    valid_rmse_sum = 0

    # Split into small and large X values.
    x_split = x[buckets==bucketflag]
    y_split = y[buckets==bucketflag]

    # Run Hyperparameter optimization on either the small or medium split.
    model_params, reg_tuning_results, reg_scores = hyperparameter_optimizer(x_split, y_split, params, model)
    model.set_params(**model_params)

    # First of all, training needs to be done on the proper 'bucketized' data.  But when we are running testing, we want to
    #   take pre-bucketed data, bucket them, and then predict.  This provides us with a different kind of challenge in our
    #   overall design.

    # Manual implementation of Stratified K-Fold.  First I'll gather up all the saved classification models.
    saved_clf = load_saved_clf()

    # Doing this the less-efficient way just to get it up and going.
    for clf, threshold, var, name in saved_clf:

        # If the thresholds line up.
        if reg_threshold==threshold:

            logger.info('Regression results with %s classification.' %(name))
            logger.info('-----------------------------------------\n')

            for seed in seeds:
                i += 1
                logger.debug('Training:')
                # Stratify!
                x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=(1/folds), random_state=seed, stratify=buckets)

                # Create a small and large training set
                buckets_train = buckets[y_train.index]
                #buckets_valid = buckets[y_valid.index]

                x_train_reg = x_train[buckets_train==bucketflag]
                buckets_train_reg = buckets_train[buckets_train==bucketflag]
                y_train_reg = y_train[buckets_train==bucketflag]

                model.fit(x_train_reg, y_train_reg)

                logger.debug('Training Finished.')

                # Now we need to train and apply our classifier on the original dataset.  I think we should apply whichever transformations
                #   already existing on that part of the pipeline.  We should already have the saved .joblib files so this should be easier.

                # Apply Sequential Feature Selection to the x_train values.
                sfs = load(path + '/%s/sfs/%s %2.2f fs.joblib' %(name, name, threshold))
                x_train_clf = sfs.transform(x_train)
                x_valid_clf = sfs.transform(x_valid)

                # Apply PCA if applicable with the necessary var.
                if var != False:
                    pca = load(path + '/%s/sfs-pca/%s %2.2f pca.joblib' %(name, name, threshold))
                    x_train_clf = pca.transform(x_train_clf)
                    x_valid_clf = pca.transform(x_valid_clf)
            
                    # Dimensonality Reduction based on accepted variance.
                    ratios = np.array(pca.explained_variance_ratio_)
                    ratios = ratios[ratios.cumsum() <= (var/100)]
                    
                    # Readjust the dimensions of x based on the variance we want.
                    length = len(ratios)
                    x_train_clf = x_train_clf[:,0:length]
                    x_valid_clf = x_valid_clf[:,0:length]

                clf.fit(x_train_clf, buckets_train)

                # Apply the transformations to the Validation set:
                bucket_valid = clf.predict(x_valid_clf)
                x_valid_reg = x_valid[bucket_valid==bucketflag]

                # Test the model on the training set.
                y_train_pred = model.predict(x_train_reg)
                train_accuracy = accuracy_score(y_train, y_train_pred)
                train_rmse = mean_squared_error(y_train, y_train_pred)

                # Test the model on the validation set.
                y_valid_pred = model.predict(x_valid_reg)
                valid_accuracy = accuracy_score(y_valid, y_valid_pred)
                valid_rmse = mean_squared_error(y_valid, y_valid_pred)

                # Log the individual folds
                logger.info('Training Accuracy: %3.3f, Training RMSE: %3.3f, Validation Accuracy: %3.3f, '
                            'Validation RMSE: %3.3f, Fold: %i'
                            %(train_accuracy, train_rmse, valid_accuracy, valid_rmse, i))


                # Add to the sums
                train_accuracy_sum += train_accuracy
                train_rmse_sum += train_rmse
                valid_accuracy_sum += valid_accuracy
                valid_rmse_sum += valid_rmse
            
            # Calculate the averages
            train_accuracy_avg = train_accuracy_sum/folds
            train_rmse_avg = train_rmse_sum/folds
            valid_accuracy_avg = valid_accuracy_sum/folds
            valid_rmse_avg = valid_rmse_sum/folds

            # Log the average scores for all the folds
            logger.info('AVG Training Accuracy: %3.3f, AVG Training RMSE: %3.3f, AVG Validation Accuracy: %3.3f, '
                        'AVG Validation RMSE: %3.3f\n' %(train_accuracy_avg, train_rmse_avg, valid_accuracy_avg, valid_rmse_avg))

    return reg_tuning_results, model, reg_scores

def hyperparameter_tuning(reg_threshold):
    """
    This function is responsible for the initial phase of hyperparameter tuning for the regression section of the pipeline.
        The models used will be:
        -> SVR with RBF Kernel
        -> SVR with Linear Kernel
        -> Lasso Regression
        This is the first stage of hyperparameter tuning to be done before Forward Selection and PCA.

    Parameters
    ----------
    threshold: Int value that is the KI threshold that we are doing hyperparameter tuning at.
    """

    # Import the necessary data.
    df, ki_range = import_data(reg_threshold)
    path = os.getcwd()

    # Extract the x, y, and bucket information.
    x = df[df.columns[1:573]]
    y = df['KI (nM) rescaled']
    buckets = df['Buckets']

    # Apply MinMaxScaler to the initial x values.
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = pd.DataFrame(scaler.transform(x), columns=df.columns[1:573])
    
    # Create the feature set for the 3 regressors.
    rbf_params = {}
    lin_params = {}
    las_params = {}
    all_params = [rbf_params, lin_params, las_params]

    # Models and names
    models = [SVR(kernel='rbf'), SVR(kernel='linear'), Lasso()]
    names = ['SVR with RBF Kernel', 'SVR with Linear Kernel', 'Lasso Regression']
    logger.info('Hyperparameter Tuning based on buckets created by setting a threshold of %2.2f' %(reg_threshold))

    # We will need to do tuning for all the models within the small and medium sized buckets.

    for name, model, params in zip(names, models, all_params):        
        # Make sure all of the necessary folders exist before we create and save the models.
        if os.path.exists(path + '/Threshold %2.2f/' %(reg_threshold)) == False:
            os.mkdir('Threshold %2.2f' %(reg_threshold))
        if os.path.exists(path + 'Threshold %2.2f/%s/' %(reg_threshold, name)) == False:
            os.mkdir('Threshold %2.2f/%s' %(reg_threshold, name))
        
        # Run the regression training here.
        results_sml, model_sml, scores_hp_sml = fit_and_inference(x, y, buckets, params, reg_threshold, bucket_name='Small',
                                                                  model=model)
        results_sml.to_csv(path + '%2.2f/%s/Initial Hyperparameter Tuning.csv' %(reg_threshold, name))

        results_med, model_med, scores_hp_med = fit_and_inference(x, y, buckets, params, reg_threshold, bucket_name='Medium',
                                                                  model=model)
        results_med.to_csv(path + '%2.2f/%s/Initial Hyperparameter Tuning.csv' %(reg_threshold, name))

parser = argparse.ArgumentParser()
parser.add_argument('-ht', '--hyperparameter_test', help='hyperparameter_test = initial stage of hyperparmeter tuning for '
                    'various hyperparameters.', type=float)
args = parser.parse_args()

hyperparameter_test = args.hyperparameter_test

# Certain sections of the code will run depending on what we specify with the run script.
if hyperparameter_test != None:
    logger = log_files('Regression_HT.log')
    hyperparameter_tuning(hyperparameter_test)