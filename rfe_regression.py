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

from operator import add

# Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
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
from pickle import dump, load

# Plotter
import matplotlib.pyplot as plt

# Argument Parser
import argparse

# Write to a log file
import logging
import sys

## Packages to use the .fasta file.
# Compute protein descriptors
from propy import PyPro
from propy import AAComposition
from propy import CTD

# Build Sequence Object
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Read Fasta File
from pyfaidx import Fasta

# Grouping iterable
from itertools import chain

# Return file path
import glob

# Global Variables
PATH = os.getcwd()
FOLDS = 5
RAND = 42

def inferenceSingleSeqence(seq):
    
    """ The inference function gets the protein sequence, trained model, preprocessing function and selected
    features as input. 
    
    The function read the sequence as string and extract the peptide features using appropriate packages into 
    the dataframe.
    
    The necessary features are selected from the extracted features which then undergoes preprocessing function, the
    target value is predicted using trained function and give out the results. """
    
    # empty list to save the features
    listing = []
    
    # Make sure the sequence is a string
    s = str(seq)
    
    # replace the unappropriate peptide sequence to A
    s = s.replace('X','A')
    s = s.replace('x','A')
    s = s.replace('U','A')
    s = s.replace('Z','A')
    s = s.replace('B','A')
    
    # Calculating primary features
    analysed_seq = ProteinAnalysis(s)
    wt = analysed_seq.molecular_weight()
    arm = analysed_seq.aromaticity()
    instab = analysed_seq.instability_index()
    flex = analysed_seq.flexibility()
    pI = analysed_seq.isoelectric_point()
    
    # create a list for the primary features
    pFeatures = [seq, s, len(seq), wt, arm, instab, pI]
    
    # Get secondary structure in a list
    sectruc = analysed_seq.secondary_structure_fraction()
    sFeatures = list(sectruc)
    
    # Get Amino Acid Composition (AAC), Composition Transition Distribution (CTD) and Dipeptide Composition (DPC)
    resultAAC = AAComposition.CalculateAAComposition(s)
    resultCTD = CTD.CalculateCTD(s)
    resultDPC = AAComposition.CalculateDipeptideComposition(s)
    
    # Collect all the features into lists
    aacFeatures = [j for i,j in resultAAC.items()]
    ctdFeatures = [l for k,l in resultCTD.items()]
    dpcFeatures = [n for m,n in resultDPC.items()]
    listing.append(pFeatures + sFeatures + aacFeatures + ctdFeatures + dpcFeatures)
    
    # Collect feature names
    name1 = ['Name','Seq' ,'SeqLength','Weight','Aromaticity','Instability','IsoelectricPoint','Helix','Turn','Sheet']
    name2 = [i for i,j in resultAAC.items()]
    name3 = [k for k,l in resultCTD.items()]
    name4 = [m for m,n in resultDPC.items()]
    name  = []
    name.append(name1+name2+name3+name4)
    flatten_list = list(chain.from_iterable(name))
    
    # create dataframe using all extracted features and the names
    allFeatures = pd.DataFrame(listing, columns = flatten_list)

    return allFeatures

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
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    # Display the logs in the output
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
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

    # Extracting peptide sequence + formatting
    peptide_sequences = pd.read_excel(PATH + '/Positive_KI.xlsx')
    peptide_sequences = peptide_sequences.replace(r"^ +| +$", r"", regex=True) # r before a string represents raw string
    peptide_sequences.rename(columns={'Sequence':'Name'}, inplace=True)

    # Feature Extraction
    df = pd.DataFrame()
    for i in range(len(peptide_sequences)):
        df = pd.concat([df, inferenceSingleSeqence(peptide_sequences.iloc[i][0])])

    # Merging into a single dataframe. Removing extra seq column and others.
    df = pd.merge(df,peptide_sequences)
    df = df.drop(columns=['Seq','Helix','Turn','Sheet'])

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
        
        - SVC w/Linear Kernel w/rfe. {'C': 21, 'break_ties': False, 'class_weight': None, 'gamma': 'auto'},
            Test MCC = 0.714823, Train MCC = 0.897055, Threshold @ 0.01.  Large Bucket Size 46, Small Bucket Size 18, Extra Bucket Size
            9. (old)

        - XGBoost Classifier w/rfe. {'alpha': 0.0, 'gamma': 0, 'lambda': 1, 'max_depth': 5, 'n_estimators': 20, 'subsample': 1},
            Test MCC = 0.672625, Train MCC = 0.955382, Threshold @ 0.01.  Large Bucket Size 46, Small Bucket Size 18, Extra Bucket Size
            9.

        - Random Forest Classifier w/rfe and PCA @80% variance.  {'ccp_alpha': 0.0, 'criterion': 'gini', 'max_depth': 4, 
            'max_features': log2, 'n_estimators': 6}, Test MCC = 0.604743, Train MCC = 0.86625, Threshold @ 10.  Large Bucket Size 20,
            Small Bucket Size 44, Extra Bucket Size 9.

    Returns
    -------
    saved_clf: saved dict containing the models with tuned hyperparameters, thresholds, PCA variances, and names of our classification section.

    """
    # Create the models with the relevant hyperparameters.
    lin = SVC(kernel='linear', C=1, break_ties=True, class_weight=None, gamma=0.1)
    xgb = XGBClassifier(alpha=0.0, gamma=0, reg_lambda=1, max_depth=3, n_estimators=3, subsample=0.5)
    rfc = RandomForestClassifier(ccp_alpha=0.2, criterion='entropy', max_depth=6, max_features=1.0, n_estimators=6)
    
    models = [lin, xgb, rfc]
    thresholds = [10, 18, 15]   # Changed Random Forest and SVC Linear
    variances = [100, 75, 100]
    names = ['SVC with Linear Kernel', 'XGBoost Classifier', 'Random Forest Classifier']

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
    if x[buckets_pred==1].size != 0:
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

    # Save the results in a dataframe.
    cols = ['Log Y Actual', 'Log Y Predicted', 'Y Actual', 'Y Predicted', 'Actual Bucket', 'Predicted Bucket']
    df_data = zip(y, y_pred, y_unscaled, y_pred_unscaled, buckets_actual, buckets_pred)
    df = pd.DataFrame(data=df_data, columns=cols)

    # Calculate RMSE metrics.
    rmse = mean_squared_error(df['Y Actual'], df['Y Predicted'])**0.5
    log_rmse = mean_squared_error(df['Log Y Actual'], df['Log Y Predicted'])**0.5

    # We also want to calculate the classification errors.  In this case we care about MCC score.
    mcc = matthews_corrcoef(df['Actual Bucket'], df['Predicted Bucket'])

    # We want to find the RMSE's where we remove all of the values that we predicted to be in bucket 2 (KI > 4000 nM)
    y_trimmed = df['Y Actual'][df['Predicted Bucket'] != 2]
    y_pred_trimmed = df['Y Predicted'][df['Predicted Bucket'] != 2]
    log_y_trimmed = df['Log Y Actual'][df['Predicted Bucket'] != 2]
    log_y_pred_trimmed = df['Log Y Predicted'][df['Predicted Bucket'] != 2]

    # Associated RMSE scores.
    rmse_trimmed = mean_squared_error(y_trimmed, y_pred_trimmed)**0.5
    log_rmse_trimmed = mean_squared_error(log_y_trimmed, log_y_pred_trimmed)**0.5

    # List of the various RMSE scores.
    scores_list = [rmse, log_rmse, rmse_trimmed, log_rmse_trimmed, mcc]

    return scores_list, df

# Entire data pipeline for classification + regression dual model.
def clf_reg_pipeline(threshold, var, name_clf, results_df = pd.DataFrame(), clf=SVC()):
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

    # Extract the x, y, and bucket information.
    x = df[df.columns[1:573]]
    y = df['KI (nM) rescaled']
    buckets = df['Bucket']

    # Apply MinMaxScaler to the initial x values.
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = pd.DataFrame(scaler.transform(x), columns=df.columns[1:573])

    # Sequential Feature Selection with the saved model.  Make sure to extract the features here too.
    rfe = load(open(PATH + '/%s/rfe/%s %2.2f rfe.pkl' %(name_clf, name_clf, threshold), 'rb'))
    x = pd.DataFrame(rfe.transform(x), columns=rfe.get_feature_names_out())

    # Where applicable, apply PCA tuning as well.
    if var != False:
        pca = load(open(PATH + '/%s/rfe-pca/%s %2.2f pca.pkl' %(name_clf, name_clf, threshold), 'rb'))
        x = pd.DataFrame(pca.transform(x))

        # Dimensonality Reduction based on accepted variance.
        ratios = np.array(pca.explained_variance_ratio_)
        ratios = ratios[ratios.cumsum() <= (var/100)]
        
        # Readjust the dimensions of x based on the variance we want.
        length = len(ratios)
        if length > 0:
            x = x[x.columns[0:length]]

    # Load up the regression models here:
    reg_models = load_regression_models()

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
        train_scores_list_sum = [0, 0, 0, 0, 0]
        valid_scores_list_sum = [0, 0, 0, 0, 0]

        fold = 0

        # Create the necessary paths for result storage.
        if os.path.exists(PATH + '/%s/%s' %(name_clf, name_reg)) == False:
            os.mkdir('%s/%s' %(name_clf, name_reg))

        skf = StratifiedKFold(n_splits=FOLDS, random_state=RAND, shuffle=True)
        for train_index, test_index in skf.split(x,buckets):
            fold+=1

            # Create the training and validation sets.
            x_train, x_valid = x.loc[train_index], x.loc[test_index]
            y_train, y_valid = y[train_index], y[test_index]
            buckets_train = buckets[train_index]

            # Fitting the classification and regression models.
            clf.fit(x_train, buckets_train)
            sml_reg.fit(x_train[buckets_train==0], y_train[buckets_train==0])
            med_reg.fit(x_train[buckets_train==1], y_train[buckets_train==1])

            # Inference on the training and test sets.
            train_scores_list, train_df = inference(x_train, y_train, buckets, ki_range, clf, sml_reg, med_reg)
            train_df.to_csv(PATH + '/%s/%s/Training Predictions Fold %i.csv' %(name_clf, name_reg, fold))

            valid_scores_list, valid_df = inference(x_valid, y_valid, buckets, ki_range, clf, sml_reg, med_reg)
            valid_df.to_csv(PATH + '/%s/%s/Validation Predictions Fold %i.csv' %(name_clf, name_reg, fold))

            # Log the individual folds
            logger.info('Fold %i' %(fold))
            logger.info('Classifier Training MCC: %3.3f, Classifier Validation MCC: %3.3f' 
                        %(train_scores_list[4], valid_scores_list[4]))
            logger.info('Without removing any of the > 4000 values.')
            logger.info('Training RMSE: %3.3f, Training RMSE (log): %3.3f, Validation RMSE: %3.3f, Validation RMSE (log): %3.3f'
                        %(train_scores_list[0], train_scores_list[1], valid_scores_list[0], valid_scores_list[1]))
            logger.info('After removing predicted > 4000 values.')
            logger.info('Training RMSE: %3.3f, Training RMSE (log): %3.3f, Validation RMSE: %3.3f, Validation RMSE (log): %3.3f'
                        %(train_scores_list[2], train_scores_list[3], valid_scores_list[2], valid_scores_list[3]))

            # We will take the RMSE values from each individual fold and then add them to the sum total of the RMSE values.
            train_scores_list_sum = list(map(add, train_scores_list_sum, train_scores_list))
            valid_scores_list_sum = list(map(add, valid_scores_list_sum, valid_scores_list))
        
        # Get the average values of the RMSE by dividing by the total number of folds.
        train_scores_list_avg = [i/FOLDS for i in train_scores_list_sum]
        valid_scores_list_avg = [i/FOLDS for i in valid_scores_list_sum]

        # Display the overall results within the logger file.
        logger.info('Averaged Results:')
        logger.info('---------------------------------------------------------------------------------------------\n')
        logger.info('AVG Classifier Training MCC: %3.3f, AVG Classifier Validation MCC: %3.3f'
                    %(train_scores_list_avg[4], valid_scores_list_avg[4]))
        logger.info('Without removing any of the >4000 values.')
        logger.info('AVG Training RMSE: %3.3f, AVG training RMSE (log): %3.3f, AVG Validation RMSE: %3.3f, AVG Validation RMSE '
                    '(log): %3.3f\n' %(train_scores_list_avg[0], train_scores_list_avg[1], valid_scores_list_avg[0], valid_scores_list_avg[1]))
        logger.info('After removing predicted >4000 values.')
        logger.info('AVG Training RMSE: %3.3f, AVG Training RMSE (log): %3.3f, AVG Validation RMSE: %3.3f, AVG Validation RMSE (log): %3.3f\n'
                    %(train_scores_list_avg[2], train_scores_list_avg[3], valid_scores_list_avg[2], valid_scores_list_avg[3]))

        results_df.loc[len(results_df)] = [name_clf, name_reg, train_scores_list_avg[0], valid_scores_list_avg[0], train_scores_list_avg[1],
                                           valid_scores_list_avg[1], train_scores_list_avg[2], valid_scores_list_avg[2],
                                           train_scores_list_avg[3], valid_scores_list_avg[3], train_scores_list_avg[4],
                                           valid_scores_list_avg[4]]

    # Return the end result and reuse in later callouts with different classification models.
    return results_df

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

    # Create the models with the relevant hyperparameters.
    saved_clf = load_saved_clf()

    # Create an empty pandas dataframe dataframe with the results.
    results_cols = ['Classification Model', 'Regression Model', 'Training RMSE', 'Validation RMSE', 'Log Training RMSE',
                    'Log Validation RMSE', 'Trimmed Training RMSE', 'Trimmed Validation RMSE', 'Trimmed Log Training RMSE',
                    'Trimmed Log Validation RMSE', 'Training MCC', 'Validation MCC']
    results_df = pd.DataFrame(columns=results_cols)

    # I'm going to run fitting and inference 4 different times, one for each cassifier.
    for threshold, var, name_clf, clf in saved_clf:
        results_df = clf_reg_pipeline(threshold, var, name_clf, results_df, clf)

    results_df.to_csv(PATH + '/Results/rfe_results.csv')
    graph_results()

# Graphing out our results.
def graph_results():
    """ This function handles all of the graphing of our results."""
    
    # Import our results.
    results_df = pd.read_csv(PATH + '/Results/rfe_results.csv')
    results_df = results_df.iloc[:,1:]

    # Running the results:
    clf_names = results_df['Classification Model'].unique()

    # Empty array for the training/test mcc's
    train_mccs = np.ndarray(0)
    valid_mccs = np.ndarray(0)

    for name in clf_names:
        # Get a smaller subset of the results based on the classification model.
        clf_results = results_df[results_df['Classification Model'] == name]
        clf_results = clf_results.iloc[:,1:]

        # Data for graphs
        labels = list(clf_results['Regression Model'])
        train_rmse = clf_results['Trimmed Log Training RMSE']
        valid_rmse = clf_results['Trimmed Log Validation RMSE']

        # Positioning of Bars
        x = np.arange(len(labels))
        width = 0.35

        # Baseline
        fig, ax = plt.subplots(figsize=(7.5,5))
        rects1 = ax.bar(x - width/2, train_rmse, width, label='train')
        rects2 = ax.bar(x + width/2, valid_rmse, width, label='valid')

        # Text Formatting
        ax.set_ylabel('Log RMSE Scores')
        ax.set_title(name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        def autolabel(rects):
            """ Attach a text label above each bar in rects, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{0:2.2f}'.format(height),
                            xy=(rect.get_x() + rect.get_width() /2, height),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()
        fig.savefig(PATH + '/%s/%s Reduced Log RMSE Scores (rfe).png' %(name, name))

        # Extract the mcc's.
        train_mcc_clf = clf_results['Training MCC'].mean()
        valid_mcc_clf = clf_results['Validation MCC'].mean()

        train_mccs = np.append(train_mccs, train_mcc_clf)
        valid_mccs = np.append(valid_mccs, valid_mcc_clf)

    x = np.arange(len(clf_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5,5))
    rects1 = ax.bar(x - width/2, train_mccs, width, label='train')
    rects2 = ax.bar(x + width/2, valid_mccs, width, label='valid')

    ax.set_ylabel('MCC Scores')
    ax.set_title('RFECV Classifier Comparisons')
    ax.set_xticks(x)
    ax.set_xticklabels(clf_names)
    ax.legend()

    def autolabel(rects):
        """ Attach a text label above each bar in rects, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{0:2.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() /2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    fig.savefig(PATH + '/Figures/classifier_results_rfe.png')


# Argument parser section.
parser = argparse.ArgumentParser()
parser.add_argument('-reg', '--regressor', help='regressor = initial stage of hyperparmeter tuning for '
                    'various hyperparameters.', action='store_true')
parser.add_argument('-gr', '--grapher', help='grapher = create plots for all of our results and save them.',
                    action='store_true')

args = parser.parse_args()

regressor = args.regressor
grapher = args.grapher

# Certain sections of the code will run depending on what we specify with the run script.
if regressor == True:
    logger = log_files(PATH + '/Log Files/rfe_regression_log.log')
    regression()
elif grapher == True:
    graph_results()
