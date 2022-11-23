"""
Same as the bucket classifier but with RFE instead.
"""

# Importing Dependencies
import argparse
import csv

# Write to a log file
import logging
import os
import sys

# Plotter
import matplotlib.pyplot as plt
import numpy as np

# Standard libraries
import pandas as pd

# Model Persistence
from pickle import dump, load

# Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

# Models
from sklearn.linear_model import Lasso

# Metrics
from sklearn.metrics import (accuracy_score, make_scorer, matthews_corrcoef,
                             mean_squared_error)

# Preprocessing
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier

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

## Global Variables
PATH = os.getcwd()

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

# Creating a logger to record and save information.
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

# Import the complete dataset.
def import_data(threshold):
    """
    Import the full dataset from the current path.  Also apply some of the necessary preprocessing.

    Parameters
    ----------
    None

    Returns
    -------
    df:  Dataframe of the full KI training dataset, with any values above 50,000 removed.

    """

    # Extracting peptide sequence + formatting
    peptide_sequences = pd.read_excel(PATH + '/Positive_KI.xlsx')
    peptide_sequences = peptide_sequences.replace(r"^ +| +$", r"", regex=True)
    peptide_sequences.rename(columns={'Sequence':'Name'}, inplace=True)

    # Feature Extraction
    df = pd.DataFrame()
    for i in range(len(peptide_sequences)):
        df = pd.concat([df, inferenceSingleSeqence(peptide_sequences.iloc[i][0])])

    # Merging into a single dataframe. Removing extra seq column and others.
    df = pd.merge(df,peptide_sequences)
    df = df.drop(columns=['Seq','Helix','Turn','Sheet'])

    # Creates a column in our dataframe to classify into 3 separate buckets.  A 'small' and 'large' bucket
    # based on the threshold, and a 'do not measure bucket' for anything with a KI value of > 4000   
    df['Bucket'] = pd.cut(x=df['KI (nM)'], bins=(0, threshold, 4000, float('inf')), labels=(0,1,2))

    return df

# Sequential RFE for feature reduction in our data.
def recursive_feature_elimination(x, y, model=SVC()):
    """
    Perform RFECV on the given dataset, but for 
        the classifer portion of the model.  MCC is the scorer used.

    Parameters
    ----------
    x: Input values of the dataset.

    y: Output values for the different classes of the dataset.

    model: Model function used for Sequential Feature Selection.

    Returns
    -------
    x: Input values of the dataset with half of the features selected.

    rfe: The SequentialFeatureSelector model

    """

    # Fit a feature selector to SVM w/Linear kernel classifier and use the 'accuracy' score.
    logger.debug('RFE Starting')
    rfe = RFECV(model, n_jobs=-1, scoring=make_scorer(matthews_corrcoef))
    rfe = rfe.fit(x, y)
    x = pd.DataFrame(rfe.transform(x), columns=rfe.get_feature_names_out())
    logger.debug('RFE Finished')

    return x, rfe

# Code for Principal Component Analysis
def principal_component_analysis(x, var):
    """
    Perform PCA and return the transformed inputs with the principal components.

    Parameters
    ----------
    x: Input values to perform PCA on.

    var: Parameter that reduces dimensionality of the PCA.  Enter as an int from 0-100.

    Returns
    -------
    x: x input transformed with PCA.

    pca: The PrincipalComponentAnalysis model.
    
    """

    # Run PCA on the given inputs.
    logger.debug('PCA Starting')
    pca = PCA()
    pca.fit(x)
    x_pca = pd.DataFrame(pca.transform(x))
    
    # Dimensonality Reduction based on accepted variance.
    ratios = np.array(pca.explained_variance_ratio_)
    ratios = ratios[ratios.cumsum() <= (var/100)]
    
    # Readjust the dimensions of x based on the variance we want.
    length = len(ratios)
    if length > 0:
        logger.info('Selecting %i principal components making up %i%% of the variance.\n' %(length,var))
        x_pca = x_pca[x_pca.columns[0:length]]
    else:
        logger.info('Kept all principal components for %i%% of the variance.\n' %(var))
    logger.debug('PCA Finished')

    return x_pca, pca

# Optimize the hyperparameters of the classifiers using GridSearchCV
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

    scores: Pandas DataFrame of the Training + Test Scores
    """

    # Use GridsearchCV to get the optimized parameters.
    logger.debug('GridSearchCV Starting')
    clf = GridSearchCV(model,param_grid=params,scoring=make_scorer(matthews_corrcoef),cv=5,
                       return_train_score=True,n_jobs=-1)
    clf.fit(x,y)

    # Showing the best parameters found on the development set.
    logger.info('Best parameter set: %s' %(clf.best_params_))
    logger.info('-------------------------\n')

    # Testing on the development set.  Save the results to a pandas dataframe and then sort it by
    # standard deviation of the test set.
    df = pd.DataFrame(clf.cv_results_)
    index = clf.best_index_
    scores = [df['mean_train_score'][index], df['std_train_score'][index], clf.best_score_, df['std_test_score'][index], clf.best_params_]
    logger.info('Train MCC Score: %3.3f.  StDev for Train MCC: %3.3f.  Test MCC Score: %3.3f.  StDev for Test MCC: %3.3f.\n' 
                %(scores[0], scores[1], scores[2], scores[3]))

    # Save the best parameters.
    bestparams = clf.best_params_

    return bestparams, scores

# Perform optimization on the classifier as well as a k-fold cross validation.
def classifier_trainer(x, y, params, model=SVC()):
    """
    Perform fitting on the reduced datasets and then make predictions.  The output values are in the log file.

    Parameters
    ----------
    x: Reduced set of input values.

    y: Output KI values that we are using for the training and validation sets.

    params: List of hyperparameters we will be doing hyperparameter tuning with.

    model: Our model that we are optimizing hyperparameters for.

    Returns
    -------
    optimizer_results: Pandas DataFrame that has the results of our hyperparameter tuning, sorted
        for results with the smallest standard deviation in the test scores.

    model: Modfied model that has the optimized hyperparameters.

    scores: Pandas DataFrame of the training and test scores.
    """
    # Train our model
    seeds = [33, 42, 55, 68, 74]
    i = 0

    folds = len(seeds)

    # Initialize the sums of the acc/mcc's.
    train_accuracy_sum = 0
    train_mcc_sum = 0
    valid_accuracy_sum = 0
    valid_mcc_sum = 0

    optimized_features, scores = hyperparameter_optimizer(x, y, params, model)

    model.set_params(**optimized_features)

    # Manual implementation of Stratified K-Fold.
    for seed in seeds:
        i += 1
        logger.debug('Training:')
        # Stratify!
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=(1/folds), random_state=seed, stratify=y)
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
    train_accuracy_avg = train_accuracy_sum/folds
    train_mcc_avg = train_mcc_sum/folds
    valid_accuracy_avg = valid_accuracy_sum/folds
    valid_mcc_avg = valid_mcc_sum/folds

    # Log the average scores for all the folds
    logger.info('AVG Training Accuracy: %3.3f, AVG Training MCC: %3.3f, AVG Validation Accuracy: %3.3f, '
                'AVG Validation MCC: %3.3f\n' %(train_accuracy_avg, train_mcc_avg, valid_accuracy_avg, valid_mcc_avg))
    
    return model, scores

# Function to separate items into buckets.
def threshold_finder(threshold):
    """
    This function uses a classification threshold to split the data into large and small buckets.

    Parameters
    ----------
    threshold: Threshold value to set the KI classification to.

    Returns
    -------
    bucket: Dummy variable for now, returns a dummy variable
    """
    
    # Import the data.
    df = import_data(threshold)

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
        #   as the RFE and PCA models/
        attributes = param_name_model_zipper()
        vars = [75, 80, 85, 90, 95, 100]
        cols = ['Name', 'Stage', 'Features Selected', 'Train MCC', 'Train Stdev', 'Test MCC', 'Test Stdev', 'Params']
        
        for params, name, model in attributes:
            # Every time I iterate through this loop, I need to recreate x.
            x = df[df.columns[1:573]]
            x = pd.DataFrame(scaler.transform(x), columns=df.columns[1:573])
            logger.info('%s Results:\n' %(name))

            # Create a the directories for the models if they doesn't exist.
            if os.path.exists(PATH + '/%s' %(name)) == False:
                os.mkdir('%s' %(name))
            if os.path.exists(PATH + '/%s/rfe-pca' %(name)) == False:
                os.mkdir('%s/rfe-pca' %(name))
            if os.path.exists(PATH + '/%s/rfe' %(name)) == False:
                os.mkdir('%s/rfe' %(name))
            if os.path.exists(PATH + '/%s/baseline' %(name)) == False:
                os.mkdir('%s/baseline' %(name))
            if os.path.exists(PATH + '/%s/results' %(name)) == False:
                os.mkdir('%s/results' %(name))

            model_scores = pd.DataFrame(columns=cols)

            # Our main pipeline is Initial Hyperparameter Tuning -> RFE -> Principal Component Analysis -> Hyperparameter Tuning

            # Baseline
            model, scores_baseline = classifier_trainer(x, y, params, model)
            model_scores.loc[len(model_scores)] = [name, 'Baseline', len(x.columns[:]), scores_baseline[0], scores_baseline[1], scores_baseline[2], 
                                                   scores_baseline[3], scores_baseline[4]]

            # Recursive Feature Elimination
            x_rfe, rfe = recursive_feature_elimination(x, y, model)
            logger.info('RFE only results:\n')

            # RFE Block
            model_rfe, scores_rfe = classifier_trainer(x_rfe, y, params, model)
            model_scores.loc[len(model_scores)] = [name, 'rfe', len(x_rfe.columns[:]), scores_rfe[0], scores_rfe[1], scores_rfe[2],
                                                   scores_rfe[3], scores_rfe[4]]

            dump(rfe, open(PATH + '/%s/rfe/%s %2.2f rfe.pkl' %(name, name, threshold), 'wb'))

            # Now do PCA.
            logger.info('Results after PCA:')
            logger.info('------------------\n')

            # Do it for different 75-100 variances.
            for var in vars:
                # Run PCA.
                x_rfe_pca, pca = principal_component_analysis(x_rfe, var)
                _, scores_pca = classifier_trainer(x_rfe_pca, y, params, model_rfe)
                model_scores.loc[len(model_scores)] = [name, 'PCA %i%% variance' %(var), len(x_rfe.columns[:]), scores_pca[0], scores_pca[1], scores_pca[2],
                                                         scores_pca[3], scores_pca[4]]

            model_scores.to_csv(PATH + '/%s/results/%s scores with threshold %2.2f (rfe).csv' %(name, name, threshold))
            dump(pca, open(PATH + '/%s/rfe-pca/%s %2.2f pca.pkl' %(name, name, threshold), 'wb'))

    # Formatting for the logger.
    logger.info('-----------------------------------------------------\n')

# Put together the variosu classification models.
def param_name_model_zipper():
    """
    This function initializes the models, parameters, and names and zips them up.  This is done before a lot of 
        the for loops between models.

    Returns
    -------
    attributes: zipped up parameters, names, and models.
    """
    
    # Create the feature set for the 3 classifiers.  adjustments to C here too
    lin_params = {'gamma': [1e-1,1e-2,1e-3,1e-4,'scale','auto'], 'C': np.arange(1,101,5),
                  'class_weight': [None,'balanced'], 'break_ties': [False,True]}
    xgb_params = {'max_depth': np.arange(2,11,1), 'n_estimators': np.arange(1,25,1), 'gamma': np.arange(0,4,1),
                  'subsample': [0.5,1], 'lambda': [1,5,9], 'alpha': np.arange(0,1.1,0.2)}
    rfc_params = {'criterion': ['gini','entropy'], 'max_features': ['sqrt','log2',1.0,0.3], 'ccp_alpha': np.arange(0,0.3,0.1),
                  'n_estimators': np.arange(1,25,1), 'max_depth': np.arange(2,11,1)}
    all_params = [lin_params, xgb_params, rfc_params]

    # Create the string titles for the various models.
    lin_name = 'SVC with Linear Kernel'
    xgb_name = 'XGBoost Classifier'
    rfc_name = 'Random Forest Classifier'
    names = [lin_name, xgb_name, rfc_name]
    names = ['SVC with Linear Kernel', 'XGBoost Classifier', 'Random Forest Classifier']
    
    # Initialize models with the necessary hyperparameters.
    lin = SVC(kernel='linear',C=10,gamma=0.01,break_ties=True,class_weight=None)
    xgb = XGBClassifier(alpha=1.0,gamma=1,reg_lambda=1,max_depth=4,n_estimators=22,subsample=0.5)
    rfc = RandomForestClassifier(ccp_alpha=0.0,criterion='gini',max_depth=3,max_features='sqrt',n_estimators=23)

    models = [lin, xgb, rfc]

    # In a for loop, create a directory for the 3 models and then deposit the hyperparameter tuning results as well
    #   as the rfe and PCA models/
    attributes = zip(all_params, names, models)

    return attributes

# Function that finds the initial set of hyperparameters for the classifiers, pre-data transformation.
def hyperparameter_pipeline(threshold):
    """
    This function is responsible for testing and optimizing for our hyperparameters.  The models used will be:
        SVC w/Linear kernel, Random Forest Classifier, and XGBoost Classifier.  This is the first stage of hyperparameter
        tuning to be done before RFE and Principal Component Analysis.

    Parameters
    ----------
    threshold: int value that we are setting to be our threshold between the small and large buckets.
    """

    # Import the data.
    df = import_data(threshold)

    # Create the x and y values.  X = all the features.  y = the columns of buckets
    x = df[df.columns[1:573]]
    y = df['Bucket']

    # Add MinMaxScaler here.  Data seems to be overfitting.
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = pd.DataFrame(scaler.transform(x), columns=df.columns[1:573])

    # Create the feature set for the 3 classifiers.  adjustments to C here too
    lin_params = {'gamma': [1e-1,1e-2,1e-3,1e-4,'scale','auto'], 'C': np.arange(1,101,5),
                  'class_weight': [None,'balanced'], 'break_ties': [False,True]}
    xgb_params = {'max_depth': np.arange(2,11,1), 'n_estimators': np.arange(1,25,1), 'gamma': np.arange(0,4,1),
                  'subsample': [0.5,1], 'lambda': [1,5,9], 'alpha': np.arange(0,1.1,0.2)}
    rfc_params = {'criterion': ['gini','entropy'], 'max_features': ['sqrt','log2',1.0,0.3], 'ccp_alpha': np.arange(0,0.3,0.1),
                  'n_estimators': np.arange(1,25,1), 'max_depth': np.arange(2,11,1)}
    all_params = [lin_params, xgb_params, rfc_params]

    # Models and names
    models = [SVC(), XGBClassifier(), RandomForestClassifier()]
    names = ['SVC with Linear Kernel', 'XGBoost Classifier', 'Random Forest Classifier']
    
    logger.info('Hyperparameter Tuning for Threshold %2.2f:\n' %(threshold))
    # Classifier Training for all 3 classifiers.
    for name, model, params in zip(names, models, all_params):
        logger.info('GridSearchCV on %s:\n' %(name))
        if os.path.exists(PATH + '/%s/' %(name)) == False:
            os.mkdir('%s' %(name))
        if os.path.exists(PATH + '/%s/Initial Hyperparameter Tuning' %(name)) == False:
            os.mkdir('%s/Initial Hyperparameter Tuning' %(name))
        results, model, scores_hp = classifier_trainer(x, y, params, model=model)
        results.to_csv(PATH + '/%s/Initial Hyperparameter Tuning/%s Initial Hyperparameter Tuning at Threshold %2.2f.csv'
                       %(name, name, threshold))

def rfe_grapher():
    thresholds = [0.01, 0.1, 0.5, 5, 10, 15, 18]
    plt.close('all')
    # Do this for every threshold.
    for threshold in thresholds:
        # Perform the plotting for each one here.
        attributes = param_name_model_zipper()
        for params, name, model in attributes:
            print(name, threshold)
            rfe = load(open(PATH + '/%s/rfe/%s %2.2f rfe.pkl' %(name, name, threshold), 'rb'))
            rfe_results = rfe.cv_results_
            rfe_mean_test_mcc = list(rfe_results.values())[0]
            
            # Plot here
            plt.figure()
            plt.plot(rfe_mean_test_mcc)
            plt.xlabel('Number of Features Selected')
            plt.ylabel('Test MCC')
            plt.title('RFECV for %s at threshold %2.2f' %(name, threshold))
            plt.savefig(PATH + '/Figures/%s/RFECV for %s at threshold %2.2f.png' %(name, name, threshold))
            plt.close()
        print(threshold)


# Use argparse to pass various thresholds.
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', help='threshold = set the threshold to split the dataset into'
                    ' large and small buckets', type=float)
parser.add_argument('-ht', '--hyperparameter_test', help='hyperparameter_test = test for various hyperparameters',
                    type=float)
parser.add_argument('-reg', '--regressor', help='regressor = perform the regression section of the code once we have finished with '
                    'the classification section of the pipeline', action='store_true')
parser.add_argument('-gr', '--grapher', help='grapher = graph out the results of RFECV of the various functions and then save them',
                    action='store_true')           
args = parser.parse_args()

threshold = args.threshold
hyperparameter_test = args.hyperparameter_test
grapher = args.grapher

## Initialize the logger here after I get the threshold value.  Then run the classifier
if threshold != None:
    if os.path.exists(PATH + '/RFE Bucket Classifier Logs') == False:
        os.mkdir('RFE Bucket Classifier Logs')
    logger = log_files(PATH + '/RFE Bucket Classifier Logs/RFE_Threshold %2.2f.log' %(threshold))
    threshold_finder(threshold)
elif hyperparameter_test != None:
    logger = log_files('HP_Test.log')
    hyperparameter_pipeline(hyperparameter_test)
elif grapher == True:
    rfe_grapher()