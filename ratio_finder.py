"""
This section of code covers the classifier which takes takes the input data and splits it into three separate buckets based on 
    KI (nM) values.
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
from pickle import dump

# Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector

# Metrics
from sklearn.metrics import accuracy_score, make_scorer, matthews_corrcoef

# Preprocessing
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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

    # Save it to the directory.
    df.to_csv(PATH + '/Results/Current_Positive_Set.csv')

    return df

# Sequential Forward Selection for feature reduction in our data.
def sequential_selection(x, y, name, threshold, ratios, model=SVC()):
    """
    Perform Sequential Selection on the given dataset, but for the classifer portion of the 
        model.  MCC is the scorer used.  We perform both forward and backward selection.

    Parameters
    ----------
    x: Input values of the dataset.

    y: Output values for the different classes of the dataset.

    model: Model function used for Sequential Feature Selection.

    Returns
    -------
    final_x_sfs: Input values of the dataset with the proper number of features selected.

    final_sfs: The SequentialFeatureSelector model selected
    """

    # Fit a feature selector to SVM w/RBF kernel classifier and use the 'accuracy' score.
    # Forward Selection Loop
    logger.info('Forward Selection Starting')
    
    cols = ['Features Selected', 'Train Accuracy Score', 'Train MCC Score', 'Test Accuracy Score', 'Test MCC Score']
    scores_df = pd.DataFrame(columns=cols)

    # Iterate through selecting from 10%-90% of the features in increments of 10.
    high_test_mcc = 0

    for ratio in ratios:
        sfs = SequentialFeatureSelector(model, n_jobs=-1, scoring=make_scorer(matthews_corrcoef),
                                        n_features_to_select=ratio, direction='forward')
        sfs.fit(x, y)
        x_sfs = pd.DataFrame(sfs.transform(x), columns=sfs.get_feature_names_out())
        
        # Initialize measurements.
        train_accuracy_sum = 0
        train_mcc_sum = 0
        test_accuracy_sum = 0
        test_mcc_sum = 0
        
        # Stratified Kfold to test the results of sequental feature selection.
        skf = StratifiedKFold(n_splits=FOLDS, random_state=RAND, shuffle=True)
        for train_index, test_index in skf.split(x_sfs,y):
            x_train, x_test = x_sfs.loc[train_index], x_sfs.loc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(x_train, y_train)

            # Predicting on the test set.
            y_test_pred = model.predict(x_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_mcc = matthews_corrcoef(y_test, y_test_pred)

            # Predicting on the Trainign set.
            y_train_pred = model.predict(x_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_mcc = matthews_corrcoef(y_train, y_train_pred)

            # Add to the sums
            train_accuracy_sum += train_accuracy
            train_mcc_sum += train_mcc
            test_accuracy_sum += test_accuracy
            test_mcc_sum += test_mcc
        avg_test_mcc = test_mcc_sum/FOLDS
        if (avg_test_mcc > high_test_mcc) or (high_test_mcc == 0):
            high_test_mcc = avg_test_mcc
            final_x_sfs = x_sfs
            final_sfs = sfs

        # Calculate the averages
        scores_df.loc[len(scores_df)] = [x_sfs.shape[1],train_accuracy_sum/FOLDS, train_mcc_sum/FOLDS, test_accuracy_sum/FOLDS, 
                                         test_mcc_sum/FOLDS]
    
    # Display the results:
    plt.figure()
    plt.plot(scores_df['Features Selected'], scores_df['Train MCC Score'])
    plt.plot(scores_df['Features Selected'], scores_df['Test MCC Score'], '-.')
    plt.xlabel('Number of features selected')
    plt.ylabel('Cross Validation Score (MCC)')
    plt.title('Forward Selection for %s at Threshold %2.2f' %(name, threshold))
    plt.legend(['Train MCC', 'Test MCC'])
    plt.savefig(PATH + '/%s/narrowed/Forward Selection for %s at Threshold %2.2f.png' %(name, name, threshold))

    logger.info('Forward Selection Finished')

    scores_df.to_csv(PATH + '/%s/narrowed/%s features selected with threshold %2.2f.csv' %(name, name, threshold))

    return final_x_sfs, final_sfs

# Code for Principal Component Analysis
def principal_component_analysis(x, var):
    """
    Perform PCA and return the transformed inputs with the principal components.

    Parameters
    ----------
    x: Input values to perform PCA on.

    var: Parameter that reduces dimensionality of the PCA.  Enter as an int from 0-100.  100 keeps full dimensionality

    Returns
    -------
    x: x input transformed with PCA.  The number of principal components returned is based on the "var" selected.

    pca: The PrincipalComponentAnalysis model.
    
    """

    # Run PCA on the given inputs.
    logger.info('PCA Starting')
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
    logger.info('PCA Finished')

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
    logger.info('GridSearchCV Starting')
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
    model: Modfied model that has the optimized hyperparameters.

    scores: Pandas DataFrame of the training and test scores.
    """
    # Train our model
    i = 0

    # Initialize the sums of the acc/mcc's.
    train_accuracy_sum = 0
    train_mcc_sum = 0
    valid_accuracy_sum = 0
    valid_mcc_sum = 0

    optimized_features, scores = hyperparameter_optimizer(x, y, params, model)

    model.set_params(**optimized_features)

    # We use Stratified K-Fold for cross-validation of our examples.
    skf = StratifiedKFold(n_splits=FOLDS, random_state=RAND, shuffle=True)
    for train_index, test_index in skf.split(x,y):
        i+=1
        x_train, x_test = x.loc[train_index], x.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train, y_train)

        # Predicting on the test set.
        y_test_pred = model.predict(x_test)
        valid_accuracy = accuracy_score(y_test, y_test_pred)
        valid_mcc = matthews_corrcoef(y_test, y_test_pred)

        # Predicting on the Trainign set.
        y_train_pred = model.predict(x_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_mcc = matthews_corrcoef(y_train, y_train_pred)

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
    train_accuracy_avg = train_accuracy_sum/FOLDS
    train_mcc_avg = train_mcc_sum/FOLDS
    valid_accuracy_avg = valid_accuracy_sum/FOLDS
    valid_mcc_avg = valid_mcc_sum/FOLDS

    # Log the average scores for all the folds
    logger.info('AVG Training Accuracy: %3.3f, AVG Training MCC: %3.3f, AVG Validation Accuracy: %3.3f, '
                'AVG Validation MCC: %3.3f\n' %(train_accuracy_avg, train_mcc_avg, valid_accuracy_avg, valid_mcc_avg))
    
    return model, scores

# Function to separate items into buckets.
def ratio_finder():
    """
    This function finds the best ratio for all 4 of the models.  By this point, the best thresholds have already been found.

    Returns
    -------
    bucket: Dummy variable for now, returns a dummy variable
    """

    # In a for loop, create a directory for the 3 models and then deposit the hyperparameter tuning results as well
    #   as the SFS and PCA models/
    attributes = param_name_model_zipper()
    vars = [75, 80, 85, 90, 95, 100]
    cols = ['Name', 'Stage', 'Features', 'Train MCC', 'Train Stdev', 'Test MCC', 'Test Stdev', 'Params']
    
    for params, name, model, threshold, ratios in attributes:
        # Craete a path for each narrowed model.
        if os.path.exists(PATH + '/%s/narrowed' %(name)) == False:
            os.mkdir('%s/narrowed' %(name))

        # Every time I iterate through this loop, I need to recreate x.
        df = import_data(threshold)
        x = df[df.columns[1:573]]
        y = df['Bucket']

        # MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(x)
        x = pd.DataFrame(scaler.transform(x), columns=df.columns[1:573])
        logger.info('%s Results:\n' %(name))
        logger.info('-------------------\n')

        model_scores = pd.DataFrame(columns=cols)

        # Our main pipeline is Initial Hyperparameter Tuning -> Sequential Selection -> Principal Component Analysis -> Hyperparameter Tuning
        # Baseline
        logger.info('Baseline Results\n')
        logger.info('-------------------\n')
        model, scores_baseline = classifier_trainer(x, y, params, model)
        model_scores.loc[len(model_scores)] = [name, 'Baseline', len(x.columns[:]), scores_baseline[0], scores_baseline[1], scores_baseline[2], 
                                                scores_baseline[3], scores_baseline[4]]

        # Sequential Feature Selection
        x_sfs, sfs = sequential_selection(x, y, name, threshold, ratios, model)
        logger.info('SFS only results:\n')
        logger.info('-------------------\n')

        # SFS Block
        model_sfs, scores_sfs = classifier_trainer(x_sfs, y, params, model)
        model_scores.loc[len(model_scores)] = [name, 'SFS', len(x_sfs.columns[:]), scores_sfs[0], scores_sfs[1], scores_sfs[2], 
                                                scores_sfs[3], scores_sfs[4]]
        dump(sfs, open(PATH + '/%s/narrowed/%s Threshold %2.2f SFS.pkl' %(name, name, threshold), 'wb'))

        # Now do PCA.
        logger.info('Results after PCA:')
        logger.info('------------------\n')

        # Run PCA for different 75-100 variances.
        for var in vars:
            # SFS block
            x_sfs_pca, pca = principal_component_analysis(x_sfs, var)
            _, scores_sfs_pca = classifier_trainer(x_sfs_pca, y, params, model_sfs)
            model_scores.loc[len(model_scores)] = [name, 'SFS + PCA w/%i%% variance' %(var), len(x_sfs_pca.columns[:]), scores_sfs_pca[0],
                                                    scores_sfs_pca[1], scores_sfs_pca[2], scores_sfs_pca[3], scores_sfs_pca[4]]

        model_scores.to_csv(PATH + '/%s/narrowed/%s Scores with Threshold %2.2f.csv' %(name, name, threshold))
        dump(pca, open(PATH + '/%s/narrowed/%s Threshold %2.2f PCA.pkl' %(name, name, threshold), 'wb'))
    
    # Formatting for the logger.
    logger.info('--------------------------------------------------------------------------------\n')

# Put together the various classification models.  Here we will want to give each one their unique threshold as well as features selected.
def param_name_model_zipper():
    """
    This function initializes the models, parameters, and names and zips them up.  This is done before a lot of 
        the for loops between models.

    Returns
    -------
    attributes: zipped up parameters, names, and models.
    """
    
    # Create the feature set for the 3 classifiers.  adjustments to C here too
    rbf_params = {'gamma': [1e-1,1e-2,1e-3,1e-4,'scale','auto'], 'C': np.arange(1,101,5),
                  'class_weight': [None,'balanced'], 'break_ties': [False,True]}
    xgb_params = {'max_depth': np.arange(2,11,1), 'n_estimators': np.arange(1,25,1), 'gamma': np.arange(0,4,1),
                  'subsample': [0.5,1], 'lambda': [1,5,9], 'alpha': np.arange(0,1.1,0.2)}
    rfc_params = {'criterion': ['gini','entropy'], 'max_features': ['sqrt','log2',1.0,0.3], 'ccp_alpha': np.arange(0,0.3,0.1),
                  'n_estimators': np.arange(1,25,1), 'max_depth': np.arange(2,11,1)}
    knn_params = {'n_neighbors': np.arange(1,55,2), 'weights': ['uniform', 'distance'], 'leaf_size': np.arange(5,41,2),
                  'p': [1, 2]}
    all_params = [rbf_params, xgb_params, rfc_params, knn_params]

    # Create the string titles for the various models.
    rbf_name = 'SVC with RBF Kernel'
    xgb_name = 'XGBoost Classifier'
    rfc_name = 'Random Forest Classifier'
    knn_name = 'KNN Classifier'
    names = [rbf_name, xgb_name, rfc_name, knn_name]
    names = ['SVC with RBF Kernel', 'XGBoost Classifier', 'Random Forest Classifier', 'KNN Classifier']
    
    # Initialize models with the necessary hyperparameters.
    rbf = SVC(C=10,gamma=0.01,break_ties=True,class_weight=None)
    xgb = XGBClassifier(alpha=1.0,gamma=1,reg_lambda=1,max_depth=4,n_estimators=22,subsample=0.5)
    rfc = RandomForestClassifier(ccp_alpha=0.0,criterion='gini',max_depth=3,max_features='sqrt',n_estimators=23)
    knn = KNeighborsClassifier()
    models = [rbf, xgb, rfc, knn]

    # Thresholds
    thresholds = [0.01, 0.01, 15, 0.1]

    # Ratio range to select from.
    rbf_ratios = np.arange(0.05, 0.16, 0.01)
    xgb_ratios = np.arange(0.3, 0.36, 0.01)
    rfc_ratios = np.arange(0.20, 0.31, 0.01)
    knn_ratios = np.arange(0.05, 0.11, 0.01)
    all_ratios = [rbf_ratios, xgb_ratios, rfc_ratios, knn_ratios]

    # In a for loop, create a directory for the 3 models and then deposit the hyperparameter tuning results as well
    #   as the SFS and PCA models/
    attributes = zip(all_params, names, models, thresholds, all_ratios)

    return attributes

# Use argparse to pass various thresholds.
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--narrow', help='narrow = sNarrow down the range of potential ratios for feature selection', action='store_true')
parser.add_argument('-st', '--store_file', help='store_file = stores the extracted files into a csv', action='store_true')
           
args = parser.parse_args()

narrow = args.narrow
extract_file = args.store_file

## Initialize the logger here after I get the threshold value.  Then run the classifier
if narrow == True:
    logger = log_files(PATH + '/Log Files/ratio_finder.log')
    ratio_finder()
elif extract_file == True:
    df = import_data(threshold=10)