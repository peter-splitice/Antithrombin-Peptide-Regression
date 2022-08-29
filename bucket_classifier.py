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
    
    # Remove all but the 73 features that were relavant in Nivedha's Classification pipeline
    labels = pd.read_json('features.json')
    df = df[labels]

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

## Forward selection for our classifier.
def forward_selection(x, y, model):
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
    sfs = SequentialFeatureSelector(model, n_jobs=-1, scoring=make_scorer(matthews_corrcoef), tol=None,
                                    n_features_to_select='auto')
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

    pca: The PrincipalComponentAnalysis model.
    
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
    Perform GridSearchCV to find and return the best hyperparmeters.  I'll use MCC score here.  I'll also display the training
        and validation scores as they come up too.
    
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
    
    """

    # Use GridsearchCV to get the optimized parameters.
    logger.debug('GridSearchCV Starting')
    clf = GridSearchCV(model,param_grid=params,scoring=make_scorer(matthews_corrcoef),cv=5,
                       return_train_score=True,n_jobs=-1)
    clf.fit(x,y)

    # Showing the best paramets found on the development set.
    logger.info('Best parameter set: %s\n' %(clf.best_params_))
    logger.info('Best MCC score: %3.3f\n' %(clf.best_score_))

    # Testing on the development set.  Save the results to a pandas dataframe and then sort it by
    # standard deviation of the test set.
    df = pd.DataFrame(clf.cv_results_)
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
    bestvals = clf.best_params_

    return bestvals, df


def classifier_trainer(df, x, y, params, model=SVC()):
    """
    Perform fitting on the reduced datasets and then make predictions.  The output values are in the log file.

    Parameters
    ----------
    x: Reduced set of input values.

    y: Output KI values that we are using for the training and validation sets.

    Returns
    -------
    optimizer_results: Pandas Dataframe that has the results of our hyperparameter tuning, sorted
        for results with the smallest standard deviation in the test scores.

    """
    # Train our model
    seeds = [33, 42, 55, 68, 74]
    i = 0

    # Initialize the sums of the acc/mcc's.
    train_accuracy_sum = 0
    train_mcc_sum = 0
    valid_accuracy_sum = 0
    valid_mcc_sum = 0

    optimized_features, optimizer_results = hyperparameter_optimizer(x, y, params, model)

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

        # Save the results into a dataframe and display them in the logger file.
        trial = pd.DataFrame()
        trial['y_valid'] = y_valid
        trial['y_valid_pred'] = y_valid_pred
        trial['KI (nM)'] = df['KI (nM)'][trial.index]
        #logger.info('Actual: | Predicted: | KI (nM)')
        #for valid, pred, ki in zip(trial['y_valid'], trial['y_valid_pred'], trial['KI (nM)']):
        #    logger.info(' %i      |  %i         | %f' %(valid, pred, ki))
        #logger.info('Fold %i finished:\n' %(i))

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
                'AVG Validation MCC: %3.3f\n' %(train_accuracy_avg, train_mcc_avg, valid_accuracy_avg, valid_mcc_avg))
    
    return optimizer_results

# Function to separate items into buckets.
def threshold_finder(threshold):
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
    path = os.getcwd()

    # Creates a column in our dataframe to classify into 3 separate buckets.  A 'small' and 'large' bucket
    # based on the threshold, and a 'do not measure bucket' for anything with a KI value of > 4000   
    df['Bucket'] = pd.cut(x=df['KI (nM)'], bins=(0, threshold, 4000, float('inf')), labels=(0,1,2))

    large_bucket_count = df[df['Bucket'] == 1]['Name'].count()
    small_bucket_count = df[df['Bucket'] == 0]['Name'].count()
    extra_bucket_count = df[df['Bucket'] == 2]['Name'].count()

    # If either bucket is less than a third of the total nubmer of samples, I need to throw an exception.
    cutoff_length = int(len(df['Bucket'])/4)

    # The threshold was too large.
    if large_bucket_count < cutoff_length or small_bucket_count < cutoff_length:
        logger.error('Threshold of %2.2f was does not work. Large Bucket Size: %i, Small Bucket Size: %i, Extra Bucket size: %i' 
                     %(threshold, large_bucket_count, small_bucket_count, extra_bucket_count))

    # The threshold isn't too large.
    else:
        logger.info('Threshold of %2.2f provides Large bucket size: %i, Small Bucket size: %i, Extra Bucket size: %i\n'
                    %(threshold, large_bucket_count, small_bucket_count, extra_bucket_count))
        x = df[df.columns[1:573]]
        y = df['Bucket']

        # Add MinMaxScaler here.  Data seems to be overfitting.
        scaler = MinMaxScaler()
        scaler.fit(x)
        x = pd.DataFrame(scaler.transform(x), columns=df.columns[1:573])

        # In a for loop, create a directory for the 3 models and then deposit the hyperparameter tuning results as well
        #   as the SFS and PCA models/
        attributes = param_name_model_zipper()
        extracted_features = pd.DataFrame()

        for params, name, model in attributes:
            # Every time I iterate through this loop, I need to recreate x.
            x = df[df.columns[1:573]]
            x = pd.DataFrame(scaler.transform(x), columns=df.columns[1:573])

            logger.info('%s Results:\n' %(name))

            # Create a the directories for the models if they doesn't exist.
            if os.path.exists(path + '/%s' %(name)) == False:
                os.mkdir('%s' %(name))
            if os.path.exists(path + '/%s/sfs-pca' %(name)) == False:
                os.mkdir('%s/sfs-pca' %(name))
            if os.path.exists(path + '/%s/sfs' %(name)) == False:
                os.mkdir('%s/sfs' %(name))
            if os.path.exists(path + '/SFS Extracted Features') == False:
                os.mkdir('SFS Extracted Features')

            # Our main pipeline is Forward Selection -> Principal Component Analysis -> Hyperparameter Tuning
            x, sfs = forward_selection(x, y, model)

            # SFS stage
            logger.info('SFS only results:\n')
            results_sfsonly = classifier_trainer(df, x, y, params, model)
            results_sfsonly.to_csv(path + '/%s/sfs/%s SFS only results with threshold %2.2f.csv' %(name, name, threshold))
            extracted_features[name] = sfs.get_feature_names_out()
            dump(sfs, path + '/%s/sfs/%s %2.2f fs.joblib' %(name, name, threshold))

            # Now do PCA.
            logger.info('Results after PCA:\n')
            x, pca = principal_component_analysis(x)
            results = classifier_trainer(df, x, y, params, model)

            # Exporting the hyperparameter optimizations, sfs, and pca models.
            results.to_csv(path + '/%s/sfs-pca/%s results with threshold %2.2f.csv' %(name, name, threshold))
            dump(sfs, path + '/%s/sfs-pca/%s %2.2f fs.joblib' %(name, name, threshold))
            dump(pca, path + '/%s/sfs-pca/%s %2.2f pca.joblib' %(name, name, threshold))
        
        # Exporting the extracted features.
        extracted_features.to_csv(path + '/SFS Extracted Features/Saved Features for Threshold %2.2f.csv' %(threshold))

    # Formatting for the logger.
    logger.info('-----------------------------------------------------\n')

def forward_selection_only(threshold):
    """
    Use this function when you want to do forward selection and have already saved the models.  Note:  Don't do this
        until you have done the threshold finder, or it will fail.  This should be called upon with a bash script, called 
        in a for loop

    Parameters
    ----------
    Threshold: The thresholds with which we are going to be running our models through.    
    """

    # Import the data.
    df, _ = import_data()
    path = os.getcwd()

    # Creates a column in our dataframe to classify into 3 separate buckets.  A 'small' and 'large' bucket
    # based on the threshold, and a 'do not measure bucket' for anything with a KI value of > 4000
    df['Bucket'] = pd.cut(x=df['KI (nM)'], bins=(0, threshold, 4000, float('inf')), labels=(0,1,2))

    large_bucket_count = df[df['Bucket'] == 1]['Name'].count()
    small_bucket_count = df[df['Bucket'] == 0]['Name'].count()
    extra_bucket_count = df[df['Bucket'] == 2]['Name'].count()

    # If either bucket is less than a third of the total number of samples, I need to throw an exception.
    cutoff_length = int(len(df['Bucket'])/4)

    # The threshold was too large.
    if large_bucket_count < cutoff_length or small_bucket_count < cutoff_length:
        logger.error('Threshold of %2.2f was does not work. Large Bucket Size: %i, Small Bucket Size: %i, Extra Bucket size: %i' 
                     %(threshold, large_bucket_count, small_bucket_count, extra_bucket_count))

    # The threshold isn't too large.
    else:
        logger.info('Threshold of %2.2f provides Large bucket size: %i, Small Bucket size: %i, Extra Bucket size: %i\n'
                    %(threshold, large_bucket_count, small_bucket_count, extra_bucket_count))
        x = df[df.columns[1:573]]
        y = df['Bucket']

        # Add MinMaxScaler here.  Data seems to be overfitting.
        scaler = MinMaxScaler()
        scaler.fit(x)
        x = pd.DataFrame(scaler.transform(x), columns=df.columns[1:573])

        # In a for loop, create a directory for the 3 models and then deposit the hyperparameter tuning results as well
        #   as the SFS and PCA models/
        attributes = param_name_model_zipper()
        extracted_features = pd.DataFrame()
        for params, name, model in attributes:
            # Reevaluate X every time you iterate the loop.
            x = df[df.columns[1:573]]
            x = pd.DataFrame(scaler.transform(x), columns=df.columns[1:573])

            logger.info('%s Results:\n' %(name))

            # Create the needed directories if they don't exist.
            if os.path.exists(path + '/%s' %(name)) == False:
                os.mkdir('%s' %(name))
            if os.path.exists(path + '/%s/sfs' %(name)) == False:
                os.mkdir('%s/sfs' %(name))
            if os.path.exists(path + '/SFS Extracted Features') == False:
                os.mkdir('SFS Extracted Features')

            # If the .joblib files have been saved already from previous runs, use them.  If not, create new ones and save
            #   them in the SFS only folder.
            if os.path.exists(path + '/%s/sfs-pca/%s %2.2f fs.joblib' %(name, name, threshold)) == True:
                sfs = load(path + '/%s/sfs-pca/%s %2.2f fs.joblib' %(name, name, threshold))
                x = sfs.transform(x)
            else:
                x, sfs = forward_selection(x, y, model)
                dump(sfs, path + '/%s/sfs/%s %2.2f fs' %(name, name, threshold))

            # We will want to show our extracted features from sfs
            extracted_features[name] = sfs.get_feature_names_out()

            # Next we run the trainer to optimize.
            results = classifier_trainer(df, x, y, params, model)

            # Exporting the gridsearch results.
            results.to_csv(path + '/%s/sfs/%s SFS only results with threshold %2.2f.csv' %(name, name, threshold))
        
        # Exporting the extracted features.
        extracted_features.to_csv(path + '/SFS Extracted Features/Saved Features for Threshold %2.2f.csv' %(threshold))

def pca_tuning(threshold):

    path = os.getcwd()

    # DataFrame importing and adding 'Bucket' column
    df, _ = import_data()
    df['Bucket'] = pd.cut(x=df['KI (nM)'], bins=(0, threshold, 4000, float('inf')), labels=(0,1,2))

    # Get x and y values.
    x = df[df.columns[1:573]]
    y = df['Bucket']

    # Add minMaxScaler here to reduce overfitting.
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = pd.DataFrame(scaler.transform(x), columns=df.columns[1:573])

    attributes = param_name_model_zipper()

    for params, name, model in attributes:
        x = df[df.columns[1:573]]
        sfs = load(path + '/%s/sfs/%s %2.2f fs.joblib' %(name, name, threshold))
        x = sfs.transform(x)

        if os.path.exists(path + '/%s/PCA Tuning' %(name)) == False:
            os.mkdir('%s/PCA Tuning' %(name))

        logger.info('Beginning PCA only section')
        pca = PCA()
        pca.fit(x)


def param_name_model_zipper():
    """
    Zips up and creates 
    """
    
    # Create the feature set for the 3 classifiers.  Put them all into an array.
    rbf_params = {'gamma': [1e-1,1e-2,1e-3,1e-4,'scale','auto'], 'C': [5,10,50,100,250,500,1000],
                  'class_weight': [None,'balanced'], 'break_ties': [False,True]}
    xgb_params = {'max_depth': np.arange(2,11,1), 'n_estimators': np.arange(1,25,1), 'gamma': np.arange(0,4,1),
                  'subsample': [0.5,1], 'lambda': [1,5,9], 'alpha': np.arange(0,1.1,0.2)}
    rfc_params = {'criterion': ['gini','entropy'], 'max_features': ['sqrt','log2',1.0,0.3], 'ccp_alpha': np.arange(0,0.3,0.1),
                  'n_estimators': np.arange(1,25,1), 'max_depth': np.arange(2,11,1)}
    params_list = [rbf_params, xgb_params, rfc_params]

    # Create the string titles for the various models.
    rbf_name = 'SVC with RBF Kernel'
    xgb_name = 'XGBoost Classifier'
    rfc_name = 'Random Forest Classifier'
    names = [rbf_name, xgb_name, rfc_name]

    # Create the models.  We've selected our 'base' hyperparameters from earlier.

    # Mean_Test_Score = 0.394548, Std_Test_score = 0.133936, mean_train_score = 0.94489, std_train_score = 0.022332.  It looks like
    # when gamma = 0.01, C = 1 and when gamma = 0.001, c = 100.  break_ties can be true or false.  no effect.  class weight always 'None'
    rbf = SVC(C=10,gamma=0.01,break_ties=True,class_weight=None)

    # mean_test_score = 0.43961, std_test_score = 0.0879, mean_train_score = 0.854369, std_train_score = 0.32907
    # Max depth doesn't seem to matter too much past 3.
    # Another parameter set can be {'alpha': 0.4, 'gamma': 0, 'lambda': 5, 'max_depth': 9, 'n_estimators': 9, 'subsample': 0.5}
    # mean_test_score = 0.466982, std_test_score = 0.070494, mean_train_score = 0.815888, std_train_score = 0.049071
    xgb = XGBClassifier(alpha=1.0,gamma=1,reg_lambda=1,max_depth=4,n_estimators=22,subsample=0.5)

    # mean_test_score = 0.465951, std_test_score = 0.056089, mean_test_score = 0.894403, std_train_score = 0.01336
    rfc = RandomForestClassifier(ccp_alpha=0.0,criterion='gini',max_depth=3,max_features='sqrt',n_estimators=23)
    models = [rbf, xgb, rfc]

    # In a for loop, create a directory for the 3 models and then deposit the hyperparameter tuning results as well
    #   as the SFS and PCA models/
    attributes = zip(params_list, names, models)

    return attributes


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
    x, sfs = forward_selection(x,y_log)
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
    # Import the bucket classifier models we created from before.
    bucket_clf = load('bucket_clf.joblib')
    bucket_sfs = load('bucket_sfs.joblib')
    bucket_pca = load('bucket_pca.joblib')

    # This time, I actually want to use base_range
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

def hyperparameter_pipeline(threshold):
    """
    This function is responsible for testing and optimizing for our hyperparameters.  The models used will be:
        SVC w/RBF kernel, Random Forest Classifier, and XGBoost Classifier.  This is the first stage of hyperparameter
        tuning to be done before Forward Selection and Principal Component Analysis.

    Parameters
    ----------
    threshold: int value that we are setting to be our threshold between the small and large buckets.
    """

    # Import the data.
    df, _ = import_data()
    path = os.getcwd()

    # Creates a column in our dataframe to classify into 3 separate buckets.  A 'small' and 'large' bucket
    # based on the threshold, and a 'do not measure bucket' for anything with a KI value of > 4000   
    df['Bucket'] = pd.cut(x=df['KI (nM)'], bins=(0, threshold, 4000, float('inf')), labels=(0,1,2))

    # Create the x and y values.  X = all the features.  y = the columns of buckets
    x = df[df.columns[1:573]]
    y = df[df.columns[575]]

    # Add MinMaxScaler here.  Data seems to be overfitting.
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = pd.DataFrame(scaler.transform(x), columns=df.columns[1:573])


    # Create the feature set for the 3 classifiers.
    rbf_params = {'gamma': [1e-1,1e-2,1e-3,1e-4,'scale','auto'], 'C': [5,10,50,100,250,500,1000],
                  'class_weight': [None,'balanced'], 'break_ties': [False,True]}
    xgb_params = {'max_depth': np.arange(2,11,1), 'n_estimators': np.arange(1,25,1), 'gamma': np.arange(0,4,1),
                  'subsample': [0.5,1], 'lambda': [1,5,9], 'alpha': np.arange(0,1.1,0.2)}
    rfc_params = {'criterion': ['gini','entropy'], 'max_features': ['sqrt','log2',1.0,0.3], 'ccp_alpha': np.arange(0,0.3,0.1),
                  'n_estimators': np.arange(1,25,1), 'max_depth': np.arange(2,11,1)}
    all_params = [rbf_params, xgb_params, rfc_params]

    # Models and names
    models = [SVC(), XGBClassifier(), RandomForestClassifier()]
    names = ['SVC with RBF Kernel', 'XGBoost Classifier', 'Random Forest Classifier']
    
    logger.info('Hyperparameter Tuning for Threshold %2.2f:\n' %(threshold))
    # Classifier Training for all 3 classifiers.
    for name, model, params in zip(names, models, all_params):
        logger.info('GridSearchCV on %s:\n' %(name))
        if os.path.exists(path + '/%s/' %(name)) == False:
            os.mkdir('%s' %(name))
        if os.path.exists(path + '/%s/Initial Hyperparameter Tuning' %(name)) == False:
            os.mkdir('%s/Initial Hyperparameter Tuning' %(name))
        results = classifier_trainer(df, x, y, params, model=model)
        results.to_csv(path + '/%s/Initial Hyperparameter Tuning/%s Initial Hyperparameter Tuning at Threshold %2.2f.csv'
                       %(name, name, threshold))


## Use argparse to pass various thresholds.
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', help='threshold = set the threshold to split the dataset into'
                    ' large and small buckets', type=float)
parser.add_argument('-r', '--regression', help='regression = use the pre-generated models to apply'
                    ' regression onto the whole dataset.', action='store_true')
parser.add_argument('-ht', '--hyperparameter_test', help='hyperparameter_test = test for various hyperparameters',
                    type=float)
parser.add_argument('-sfs', '--sfs_only', help='sfs_only = only do Sequential Forward Selection.  Note: do not do this'
                    ' until after you already passed the --threshold argument at least once.', type=float)
           
args = parser.parse_args()

threshold = args.threshold
regression = args.regression
hyperparameter_test = args.hyperparameter_test
sfsonly = args.sfs_only

## Initialize the logger here after I get the threshold value.  Then run the classifier
if threshold != None:
    logger = log_files('threshold.log')
    threshold_finder(threshold)
elif hyperparameter_test != None:
    logger = log_files('hyperparameter test.log')
    hyperparameter_pipeline(hyperparameter_test)
elif regression == True:
    logger = log_files('regressor.log')
    regressor()
elif sfsonly != None:
    logger = log_files('forward selection only.log')
    forward_selection_only(sfsonly)

## Add email to the slurm address to get notifications.

## 

## Requirements.txt 
## python -m pip freeze
## pipe it into a txt file
