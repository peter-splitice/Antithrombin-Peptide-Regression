## Importing Dependencies
from common_dependencies import *
import json

# Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

# Models
from sklearn.linear_model import Lasso
from sklearn.svm import SVR, SVC
 
# Model Persistence
import pickle

## Global Variables
# Note:  Set VARIANCE to 'False' if PCA is not being used.
THRESHOLD = 10
CLF_NAME = 'SVC with Linear Kernel'
REG_NAME = 'Lasso Regression'
VARIANCE = 90

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
    rfe = pickle.load(open(PATH + '/%s/rfe/%s Threshold %2.2f RFE.pkl' %(CLF_NAME, CLF_NAME, THRESHOLD), 'rb'))
    x = pd.DataFrame(rfe.transform(x), columns=rfe.get_feature_names_out())
    features = rfe.get_feature_names_out()

    # Extract the features to a .json
    features = rfe.get_feature_names_out()
    features = features.tolist()
    with open('rfe_selected_features.json', 'w') as outfile:
        json.dump(features, outfile)

    # Import Principal Component Analysis
    with open(PATH + '/%s/rfe-pca/%s %2.2f rfe-pca.pkl' %(CLF_NAME, CLF_NAME, THRESHOLD), 'rb') as fh:
        pca = pickle.load(fh)

    # Depending on the variance we select, apply PCA to the reduced feature set.
    if VARIANCE != False:
        x = pd.DataFrame(pca.transform(x))

        # Dimensonality Reduction based on accepted variance.
        ratios = np.array(pca.explained_variance_ratio_)
        ratios = ratios[ratios.cumsum() <= (VARIANCE/100)]
        
        # Readjust the dimensions of x based on the variance we want.
        length = len(ratios)
        if length > 0:
            x = x[x.columns[0:length]]

    # Classification model.  Train this on the entire dataset.
    clf = SVC(kernel='linear', C=6, break_ties=False, class_weight=None, gamma=0.1)
    clf.fit(x, buckets)

    # Dump built model
    pickle.dump(clf, open(PATH + '/Inference Models/%s trained model (rfe).pkl' %(CLF_NAME), 'wb'))

    # Regression model.  Train 2 models, one on the smaller bucket and one on the medium bucket.
    sml_reg = Lasso(alpha=0.1, selection='cyclic')
    med_reg = Lasso(alpha=0.1, selection='cyclic')
    sml_reg.fit(x[buckets==0], y[buckets==0])
    med_reg.fit(x[buckets==1], y[buckets==1])
    
    # Dump models as .pkl
    pickle.dump(sml_reg, open(PATH + '/Inference Models/%s trained model small bucket (rfe).pkl' %(REG_NAME), 'wb'))
    pickle.dump(med_reg, open(PATH + '/Inference Models/%s trained model medium bucket (rfe).pkl' %(REG_NAME), 'wb'))
    
    ## Inference
    # Get the test data and the napply the necessary transforms.
    test_set = test_data()
    x_test = test_set[test_set.columns[1:573]]
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
    x_test = pd.DataFrame(rfe.transform(x_test), columns=rfe.get_feature_names_out())

    ## If we utilize PCA:
    if VARIANCE != False:
        x_test = pd.DataFrame(pca.transform(x_test))

        # Dimensionality Reduction based on accepted variance.
        ratios = np.array(pca.explained_variance_ratio_)
        ratios = ratios[ratios.cumsum() <= (VARIANCE/100)]

        # Readjust the dimensions of x based on the variance we want.
        length = len(ratios)
        if length > 0:
            x_test = x_test[x_test.columns[0:length]]
            
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
    logger = log_files(PATH + '/Log Files/rfe_ki_predictions.log')
    logger.info('Displaying KI Range below.')
    logger.info('--------------------------')
    logger.info(ki_range)

main()