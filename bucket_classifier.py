## Importing Dependencies

# Standard libraries
import pandas as pd
import numpy as np
import os
import csv

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Models
from sklearn.linear_model import Lasso
from sklearn.svm import SVR, SVC

# Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector

# Metrics
from sklearn.metrics import matthews_corrcoef, mean_squared_error, accuracy_score

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
def log_files(threshold):
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

    # Print the logs
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    # Write the logs to a file
    file_handler = logging.FileHandler('Threshold_%i.log' %(threshold))
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

    # Data where KI > 50uM (50,000nM) is an outlier.  Total of 69 different values.
    df = df[df['KI (nM)']<75000]
    logger.debug('Without outliers, the dataset has %i examples.' %(len(df)))

    # Rescaling the dataframe in the log10 (-5,5) range.
    df['KI (nM) rescaled'], base_range  = rescale(df['KI (nM)'], destination_interval=(-5,5))

    return df, base_range


## Logarithmically scalling the values.
def rescale(array=np.array(0), destination_interval=any):
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
    sfs.get_support()
    x = sfs.transform(x)

    return x

## Forward selection for our classifier.
def fs_classifier(x, y):
    """
    Perform Sequential Forward Selection on the given dataset, but for 
        the classifer portion of the model.  The model used is SVM w/RBF Kernel.

    Parameters
    ----------
    x: Input values of the dataset.

    bucket: True = large bucket, False = small bucket.

    Returns
    -------
    x: Input values of the dataset with half of the features selected.

    """

    # Fit a feature selector to SVM w/RBF kernel classifier and use the 'accuracy' score.
    logger.debug('Forward Selection Starting')
    clf = SVC(kernel='rbf')
    sfs = SequentialFeatureSelector(clf, n_jobs=-1, scoring='accuracy')
    sfs.fit(x, y)
    x = sfs.transform(x)
    logger.debug('Forward Selection Finished')
    
    return x


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

    return x

def classifer_trainer(x, y):
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

    for seed in seeds:
        i += 1
        logger.debug('Training:')
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=seed)
        clf = SVC(kernel='rbf')
        clf.fit(x_train, y_train)

        logger.debug('Training Finished.')

        # Test the model on the training set.
        y_train_pred = clf.predict(x_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_mcc = matthews_corrcoef(y_train, y_train_pred)

        # Test the model on the validation set.
        y_valid_pred = clf.predict(x_valid)
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



# Function to separate items into buckets.
def bucket_seperator(threshold, df=pd.DataFrame()):
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
    
    # Creates a column in our dataframe to classify into a "small" and "large" bucket based on a threshold.
    df['Bucket'] = df['KI (nM)'] > threshold

    # Try basing the threshold off of log transform?

    large_bucket_count = df[df['Bucket'] == True]['Name'].count()
    small_bucket_count = df[df['Bucket'] == False]['Name'].count()

    # If either bucket is less than a third of the total nubmer of samples, I need to throw an exception.
    cutoff_length = int(len(df['Bucket'])/3)

    # The threshold was too large.
    if large_bucket_count < cutoff_length or small_bucket_count < cutoff_length:
        logger.error('Threshold of %i was too large. Large Bucket Size: %i, Small Bucket Size: %i' 
                     %(threshold, large_bucket_count, small_bucket_count))

    # The threshold isn't too large.
    else:
        logger.info('Threshold of %i provides Large bucket size: %i, Small Bucket size: %i'
                    %(threshold, large_bucket_count, small_bucket_count))
        x = df[df.columns[1:572]]
        y = df[df.columns[575]]

        # SVM w/RBF kernel is our model.  We need to do this in conjuinction with Forward Selection and PCA
        x = fs_classifier(x, y)
        x = principal_component_analysis(x)
        classifer_trainer(x, y)
  
    return df

# Generate our classifier to classify new test data into buckets since we don't know their KI
def classifier(threshold):
    """
    This function serves as our classifier to categorize the data into tiers.  This classifier will be done with
        a SVM w/RBF Kernel in conjunction with Forward Selection and PCA.  Note that this function will output
        the models as dynamically named .joblib files.
    
    Parameters
    ----------
    threshold: The threshold values to set for spliting the dataset into larger/smaller buckets.

    Returns
    -------
    n/a

    """
    # Import the data and seperate it into buckets.
    df, _ = import_data()
    df = bucket_seperator(threshold, df)

# Calculating the MCC scores of the classifiers.
def verify_clf():
    """
    This function loads the classification models, calculates the MCC scores, stores them in dataframes,
        and then visualizes them in a chart.  This will export my results into an external file.

    
    """

    clf = load()

    accuracy = accuracy_score()
    mcc = matthews_corrcoef()


## Use argparse to pass various thresholds.
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', help='threshold = set the threshold to split the dataset into'
                    ' large and small buckets', type=int)
           
args = parser.parse_args()

threshold = args.threshold

## Initialize the logger here after I get the threshold value.  Then run the classifier
logger = log_files(threshold)
classifier(threshold)

## Add email to the slurm address to get notifications.


## I only need one node, use multiple threads.
## Check for multi-threading.
## Use HTOP to check for multi-threading.  I might need to install it.

## 

## Requirements.txt 
## python -m pip freeze
## pipe it into a txt file