## Importing Dependencies

# Standard libraries
import pandas as pd
import numpy as np
import os
import csv

# Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Models
from sklearn.linear_model import Lasso
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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
def log_files():
    """
    Create the meachanism for which we log results to a .log file.

    Parameters
    ----------
    None

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
    file_handler = logging.FileHandler('threshold.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Adding the file and output handlers to the logger.
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    return logger


## Import the complete dataset.
def import_data():
    """
    Import the full dataset from the current path.  Also apply some of the necessary preprocessing.

    Parameters
    ----------
    None

    Returns
    -------
    df:  Dataframe of the full KI training dataset, with any values above 50,000 removed.

    base_range:  Contains the range of values within the dataframe for rescaling purposes.

    """

    # Importing the full KI set into a dataframe.
    path = os.getcwd()
    df = pd.read_csv(path + '/PositivePeptide_Ki.csv')
    logger.debug('The full dataset has %i examples.' %(len(df)))

    # Rescaling the dataframe in the log10 (-5,5) range.
    df['KI (nM) rescaled'], base_range  = rescale(df['KI (nM)'], destination_interval=(-5,5))

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


def test_model(name, x_train, y_train, x_valid, y_valid, model=SVR()):

    """
    Fit, predict, and determine + plot the rmse of a given model.'

    Displays a plot of the predicted vs actual validation results.

    Parameters
    ----------
    name: String-like name for the model you are creating.  Example: 'Baseline Lasso Regression.'

    model: Model from the sklearn library that you are trying to fit/predict/plot.

    x_train: Inputs from the training set.

    y_train: Outputs of the training set.

    x_valid: Inputs from the validation set.

    y_valid: Outputs of the validation set.


    Returns
    -------
    model: Fitted model given in the input.

    y_pred_log:  Log scale of the predicted validation set.

    rmse: Root Mean-Squared-Error of the predicted vs actual validation set.

    """

    # Fitting and testing the model.
    model.fit(x_train, y_train)

    # With the validation set
    y_pred_log = model.predict(x_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred_log))

    # Creating the plots for the results
    plt.figure(figsize=(10,6))
    plt.plot(y_valid.values, label='Actual', marker='o', linestyle='dashed')
    #plt.plot(y_train_log.values, label='Actual', marker='o', linestyle='dashed')
    plt.plot(y_pred_log, label='Predicted', marker='x', linestyle='dashed')
    plt.legend()
    plt.title('Actual vs predicted results for %s: Log scale, range 5.' %(name))
    plt.xlabel('Sample Number')
    plt.ylabel('KI (nM): Log scale, range 5')

    # Display the rmse that we need
    print('The RMSE of the log of the values is: %f' %(rmse))

    return model, y_pred_log, rmse


## Perform forward selection using svr w/RBF kernel
def fs_regressor(x, y):
    """
    Perform Sequentual Forward Selection on a given dataset.  This will
        return half of the features of the initial dataset.  The model used
        is SVR w/RBF kernel.

    Parameters
    ----------
    x: Input values of the dataset.

    y: Output values of the dataset.

    Returns
    -------
    x: Input values of the dataset with half the features selected.

    """
    # Fit a feature selector to SVR w/RBF kernel regressor and use the MSE score.
    reg = SVR(kernel='rbf')
    sfs = SequentialFeatureSelector(reg, n_jobs=-1, scoring='neg_mean_squared_error')
    sfs.fit(x, y)
    x = sfs.transform(x)

    return x, sfs

## Forward selection for our classifier.
def fs_classifier(x, y, model):
    """
    Perform Sequential Forward Selection on the given dataset, but for 
        the classifer portion of the model.  MCC is the scorer used.

    Parameters
    ----------
    x: Input values of the dataset.

    y: Output values for the different classes of the dataset.

    model: Model function used for Sequential Feature Selection.

    Returns
    -------
    x: Input values of the dataset with half of the features selected.

    sfs: The SequentialFeatureSelector model

    """

    # Fit a feature selector to SVM w/RBF kernel classifier and use the 'accuracy' score.
    logger.debug('Forward Selection Starting')
    sfs = SequentialFeatureSelector(model, n_jobs=-1, scoring=make_scorer(matthews_corrcoef))
    sfs.fit(x, y)
    x = sfs.transform(x)
    logger.debug('Forward Selection Finished')

    return x, sfs


## Code for Principal Component Analysis
def principal_component_analysis(x):
    """
    Perform PCA and return the transformed inputs with the principal components.

    Parameters
    ----------
    x: Input values to perform PCA on.

    Returns
    -------
    x: x input transformed with PCA.
    
    """

    # Run PCA on the given inputs.
    logger.debug('PCA Starting')
    pca = PCA()
    pca.fit(x)
    x = pca.transform(x)
    logger.debug('PCA Finished')

    return x, pca

def hyperparameter_optimizer(x, y, params, model=SVC()):
    """
    Perform GridSearchCV to find and return the best hyperparmeters.  I'll use MCC score here.
    
    Parameters
    ----------
    x: Input values to perform GridSearchCV with.

    y: Output values to create GridSearchCV with.

    params: Dictionary of parameters to run GridSearchCV on.

    model: The model that we are using for GridSearchCV

    Returns
    -------
    bestvals: Optimzied hyperparameters for the model that we are running the search on.
    
    """

    # Use GridsearchCV to get the optimized parameters.
    logger.debug('GridSearchCV Starting')
    clf = GridSearchCV(model, params, scoring=make_scorer(matthews_corrcoef), cv=5, n_jobs=-1)
    clf.fit(x,y)

    # Showing the best paramets found on the development set.
    logger.debug('Best parameters set found on development set:')
    logger.debug('')
    logger.debug(clf.best_params_)
    logger.debug('')

    # Testing on the development set.
    logger.debug('Grid scores on development set:')
    logger.debug('')
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        logger.debug('%0.3f (+/-%0.03f) for %r' % (mean, std*2, params))
    logger.debug('GridSearchCV Finished')
    logger.debug('')

    # Save the best parameters.
    bestvals = clf.best_params_

    return bestvals


def classifer_trainer(x, y, params, model=SVC()):
    """
    Perform fitting on the reduced datasets and then make predictions.  The output values are in the log file.

    Parameters
    ----------
    x: Reduced set of input values.

    y: Output KI values that we are using for the training and validation sets.

    Returns
    -------
    None

    """
    # Train our model
    seeds = [33, 42, 55, 68, 74]
    i = 0

    # Initialize the sums of the acc/mcc's.
    train_accuracy_sum = 0
    train_mcc_sum = 0
    valid_accuracy_sum = 0
    valid_mcc_sum = 0

    optimized_features = hyperparameter_optimizer(x, y, params, model)

    model.set_params(**optimized_features)

    for seed in seeds:
        i += 1
        logger.debug('Training:')
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=seed)
        model.fit(x_train, y_train)

        logger.debug('Training Finished.')

        # Test the model on the training set.
        y_train_pred = model.predict(x_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_mcc = matthews_corrcoef(y_train, y_train_pred)

        # Test the model on the validation set.
        y_valid_pred = model.predict(x_valid)
        valid_accuracy = accuracy_score(y_valid, y_valid_pred)
        valid_mcc = matthews_corrcoef(y_valid, y_valid_pred)

        # Log the individual folds
        logger.info('Training Accuracy: %3.3f, Training MCC: %3.3f, Validation Accuracy: %3.3f, '
                    'Validation MCC: %3.3f, Fold: %i'
                    %(train_accuracy, train_mcc, valid_accuracy, valid_mcc, i))

        # Add to the sums
        train_accuracy_sum += train_accuracy
        train_mcc_sum += train_mcc
        valid_accuracy_sum += valid_accuracy
        valid_mcc_sum += valid_mcc
    
    # Calculate the averages
    train_accuracy_avg = train_accuracy_sum/5
    train_mcc_avg = train_mcc_sum/5
    valid_accuracy_avg = valid_accuracy_sum/5
    valid_mcc_avg = valid_mcc_sum/5

    # Log the average scores for all the folds
    logger.info('AVG Training Accuracy: %3.3f, AVG Training MCC: %3.3f, AVG Validation Accuracy: %3.3f, '
                'AVG Validation MCC: %3.3f' %(train_accuracy_avg, train_mcc_avg, valid_accuracy_avg, valid_mcc_avg))

def classifier_pipeline(x, y, model, params):
    """
    This function is our pipeline for the bucket classifier.  The outputs are recorded into a log file.
    
    Parameters
    ----------
    x: Input variables
    
    y: Output classes
    
    model: Model that we are using
    
    """
    x, _ = fs_classifier(x, y, model)
    x, _ = principal_component_analysis(x)
    classifer_trainer(x, y, params, model)


# Function to separate items into buckets.
def bucket_seperator(threshold):
    """
    This function uses a classification threshold to split the data into large and small buckets.

    Parameters
    ----------
    threshold: Threshold value to set the KI classification to.

    df: Pandas DataFrame that is being "classified"

    Returns
    -------
    bucket: Dummy variable for now, returns a dummy variable
    """
    
    # Import the data.
    df, _ = import_data()

    # Creates a column in our dataframe to classify into 3 separate buckets.  A 'small' and 'large' bucket
    # based on the threshold, and a 'do not measure bucket' for anything with a KI value of > 4000
    df['Bucket'] = pd.cut(x=df['KI (nM)'], bins=(0, threshold, 4000, float('inf')), labels=(0,1,2))

    # Try basing the threshold off of log transform?

    large_bucket_count = df[df['Bucket'] == 1]['Name'].count()
    small_bucket_count = df[df['Bucket'] == 0]['Name'].count()
    extra_bucket_count = df[df['Bucket'] == 2]['Name'].count()

    # If either bucket is less than a third of the total nubmer of samples, I need to throw an exception.
    cutoff_length = int(len(df['Bucket'])/4)

    # The threshold was too large.
    if large_bucket_count < cutoff_length or small_bucket_count < cutoff_length:
        logger.error('Threshold of %3.3f was too large. Large Bucket Size: %i, Small Bucket Size: %i, Extra Bucket size: %i' 
                     %(threshold, large_bucket_count, small_bucket_count, extra_bucket_count))

    # The threshold isn't too large.
    else:
        logger.info('Threshold of %3.3f provides Large bucket size: %i, Small Bucket size: %i, Extra Bucket size: %i'
                    %(threshold, large_bucket_count, small_bucket_count, extra_bucket_count))
        x = df[df.columns[1:573]]
        y = df[df.columns[575]]

        # Create the feature set for the 3 classifiers.
        rbf_params = {'gamma': [1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]}
        xgb_params = {'n_estimators': [50, 100, 150], 'max_depth': [1, 2, 3, 4]}
        rf_params = {'class_weight': ['balanced'], 'n_estimators': [50, 100, 150], 'max_depth': [1, 2, 3, 4], 
                       'min_samples_leaf': [1, 2, 3], 'min_samples_split': [1, 2, 3, 4]}

        # Classifier pipeline for all 3 classifiers.
        logger.info('\nSVC w/RBF Kernel Results:')
        classifier_pipeline(x, y, SVC(kernel='rbf'), rbf_params)
        logger.info('\nXGBoost Classifier Results:')
        classifier_pipeline(x, y, XGBClassifier(), xgb_params)
        logger.info('\nRandom Forest Classifier Results:')
        classifier_pipeline(x, y, RandomForestClassifier(), rf_params)

    # Formatting for the logger.
    logger.info('-----------------------------------------------------')
    logger.info('')
    return df

def classifier():
    """
    This function creates our finalized classifier and saves it into a .joblib file for later calling.
    """

    threshold = 18
    df, _ = import_data()

    df['Bucket'] = df['KI (nM)'] > threshold
    x = df[df.columns[1:573]]
    y = df[df.columns[575]]

    x, sfs = fs_classifier(x, y, model=SVC(kernel='rbf'))
    x, pca = principal_component_analysis(x)

    # Fit the new x and y to a SVC w/rbf kernel.
    clf = SVC(kernel='rbf')
    clf.fit(x, y)

    # Save the models to external joblib files.
    dump(sfs, 'bucket_sfs.joblib')
    dump(pca, 'bucket_pca.joblib')
    dump(clf, 'bucket_clf.joblib')


def ki_pipeline(df=pd.DataFrame(), base_range=(-10,10)):
    """
    This function takes either one of the small or large bucketed dataframes and contains the KI regression pipeline.

    Parameters
    ----------
    df: Pandas DataFrame for either the small or large bucketed dataset.
    """

    # Initialize the x and y values.
    x = df[df.columns[1:573]]
    y_log = df[df.columns[574]]
    y = df[df.columns[573]]

    # Apply Forward Selection and PCA.  Keep the SFS and PCA pipeline components for use on the test set.
    x, sfs = fs_regressor(x,y_log)
    x, pca = principal_component_analysis(x)

    # Train/test splits
    x_train, x_valid, y_log_train, y_log_valid = train_test_split(x, y_log, test_size=0.2, random_state=42)
    _, _, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train and fit the model
    reg = SVR(kernel='rbf')
    reg.fit(x_train, y_log_train)
    y_log_pred = reg.predict(x_valid)

    # Reconvert back to base
    y_pred = unscale(y_log_pred, base_range)

    # Calculate the rmse for both log and base.
    rmse_log = np.sqrt(mean_squared_error(y_log_valid, y_log_pred))
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))


# Calculating the MCC scores of the classifiers.
def regressor():
    """
    This function loads the classification models, calculates the MCC scores, stores them in dataframes,
        and then visualizes them in a chart.  This will export my results into an external file.


    """

    bucket_clf = load('bucket_clf.joblib')
    bucket_sfs = load('bucket_sfs.joblib')
    bucket_pca = load('bucket_pca.joblib')

    df, base_range = import_data()

    # Transform the input data and then use the created classifier to split the values into buckets.
    x = df[df.columns[1:573]]
    x = bucket_sfs.transform(x)
    x = bucket_pca.transform(x)
    df['Bucket'] = bucket_clf.predict(x)

    # Create new Dataframes with the Large and Small buckets
    df_large = df[df['Bucket'] == True]
    df_small = df[df['Bucket'] == False]

    ki_pipeline(df_large, base_range)
    ki_pipeline(df_small, base_range)



## Use argparse to pass various thresholds.
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', help='threshold = set the threshold to split the dataset into'
                    ' large and small buckets', type=float)
parser.add_argument('-b', '--bucket', help='bucket = generate the final model that splits the dataset'
                    ' into large and small buckets', action='store_true')
parser.add_argument('-r', '--regression', help='regression = use the pre-generated models to apply'
                    ' regression onto the whole dataset.', action='store_true')
           
args = parser.parse_args()

threshold = args.threshold
bucket = args.bucket
regression = args.regression

## Initialize the logger here after I get the threshold value.  Then run the classifier

logger = log_files()

if threshold != None:
    bucket_seperator(threshold)
elif bucket == True:
    classifier()
elif regression == True:
    regressor()

## Add email to the slurm address to get notifications.

## 

## Requirements.txt 
## python -m pip freeze
## pipe it into a txt file
