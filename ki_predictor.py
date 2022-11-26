## Importing Dependencies
# Standard
import pandas as pd
import numpy as np
import os
import csv
import json

from operator import add

# Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

# Models
from sklearn.linear_model import Lasso
from sklearn.svm import SVR, SVC

# Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
 
# Metrics
from sklearn.metrics import matthews_corrcoef, mean_squared_error, accuracy_score, make_scorer

# Model Persistence
import pickle

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

## Global Variables
#
PATH = os.getcwd()
FOLDS = 5
RAND = 42
THRESHOLD = 0.01
CLF_NAME = 'SVC with RBF Kernel'
REG_NAME = 'SVR with RBF Kernel'

## Getting the Protein Sequences.
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

# Importing the model reference dataset:
def training_data():
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

    # Rescaling the dataframe in the log10 (-5,5) range.
    df['KI (nM) rescaled'], base_range  = rescale(df['KI (nM)'], destination_interval=(-5,5))

    # Create a colunmn in our dataframe to define the buckets.  I will be creating a series of models that classifies the buckets.
    df['Bucket'] = pd.cut(x=df['KI (nM)'], bins=(0, THRESHOLD, 4000, float('inf')), labels=(0,1,2))

    return df, base_range

# Importing the test data
def test_data():
    """
    Import the full test dataset from the current path.

    Parameters
    ----------
    None

    Returns
    -------
    x_test: DataFrame containing the test dataset.
    """
    # Import, format, and drop duplicates.
    peptide_sequences = pd.read_csv('combined_hits.csv')
    peptide_sequences = peptide_sequences.replace(r"^ +| +$", r"", regex=True)
    name_index = peptide_sequences.columns.get_loc('Seq')
    peptide_sequences.rename(columns={'Seq':'Name'}, inplace=True)
    peptide_sequences = peptide_sequences.drop_duplicates(subset=['Name'])

    # Create a dataframe for the extracted features for the peptide sequences.
    df = pd.DataFrame()
    for i in range(len(peptide_sequences)):
        df = pd.concat([df, inferenceSingleSeqence(peptide_sequences.iloc[i][name_index])])
    df = df.drop(columns=['Seq','Helix','Turn','Sheet'])

    return df

# Main function
def main():
    """
    Perform the entirety of the model training and analysis on the best model pipeline we have alongside the best hyperparameters.
    """
    ## Model Training and saving
    # Import the training data.
    df, ki_range = training_data()

    # MinMaxScaler on the data
    # Extract the x, y, and bucket information.
    x = df[df.columns[1:573]]
    y = df['KI (nM) rescaled']
    buckets = df['Bucket']

    # Apply MinMaxScaler to the initial x values.
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = pd.DataFrame(scaler.transform(x), columns=df.columns[1:573])

    # Dump built model
    pickle.dump(scaler, open(PATH + '/Inference Models/MinMaxScaler transformation.pkl', 'wb'))

    # Load the saved Sequential Forward Selection Model and apply the transform.
    sfs = pickle.load(open(PATH + '/%s/narrowed/%s Threshold %2.2f SFS.pkl' %(CLF_NAME, CLF_NAME, THRESHOLD), 'rb'))
    x = pd.DataFrame(sfs.transform(x), columns=sfs.get_feature_names_out())
    features = sfs.get_feature_names_out()

    # Extract the features to a .json
    features = sfs.get_feature_names_out()
    features = features.tolist()
    with open('selected_features.json', 'w') as outfile:
        json.dump(features, outfile)

    # Classification model.  Train this on the entire dataset.
    clf = SVC(kernel='rbf', C=21, break_ties=False, class_weight=None, gamma='auto')
    clf.fit(x, buckets)

    # Dump built model
    pickle.dump(clf, open(PATH + '/Inference Models/%s trained model.pkl' %(CLF_NAME), 'wb'))

    # Regression model.  Train 2 models, one on the smaller bucket and one on the medium bucket.
    sml_reg = SVR(kernel='rbf', C=6, epsilon=0.2, gamma='scale')
    med_reg = SVR(kernel='rbf', C=6, epsilon=0.2, gamma='scale')
    sml_reg.fit(x[buckets==0], y[buckets==0])
    med_reg.fit(x[buckets==1], y[buckets==1])
    
    # Dump models as .pkl
    pickle.dump(sml_reg, open(PATH + '/Inference Models/%s trained model small bucket.pkl' %(REG_NAME), 'wb'))
    pickle.dump(med_reg, open(PATH + '/Inference Models/%s trained model medium bucket.pkl' %(REG_NAME), 'wb'))
    
    ## Inference
    # Get the test data and the napply the necessary transforms.
    test_set = test_data()
    x_test = test_set[test_set.columns[1:573]]
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
    x_test = pd.DataFrame(sfs.transform(x_test), columns=sfs.get_feature_names_out())

    # Predict the buckets.
    buckets_pred = clf.predict(x_test)
    test_set['Bucket'] = buckets_pred

    # Make predictions for all of the buckets.  The large bucket we'll just predict as 0 for now.  Only make predictions if the arrays 
    #   aren't empty.
    if x_test[buckets_pred==0].size != 0:
        sml_pred = sml_reg.predict(x_test[buckets_pred==0])
    if x_test[buckets_pred==1].size != 0:
        med_pred = med_reg.predict(x_test[buckets_pred==1])
    lrg_pred = np.zeros(np.count_nonzero(x_test[buckets_pred==2]))

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

    y_pred_unscaled = unscale(y_pred, ki_range)
    test_set['KI (nM) Predicted'] = y_pred_unscaled

    # Save the results in a new .csv
    results = test_set[['Name','Bucket','KI (nM) Predicted']]
    results.to_csv(PATH + 'blind_set_predictions.csv')

    # Logger file for the KI_range for unscaling
    logger = log_files(PATH + '/Log Files/KI_predictions.log')
    logger.info('Displaying KI Range below.')
    logger.info('--------------------------')
    logger.info(ki_range)

main()