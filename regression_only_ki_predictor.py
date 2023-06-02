## Importing Dependencies
from common_dependencies import *
import json

# Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

# Models
from sklearn.svm import SVR

# Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector

# Model Persistence
import pickle

# Return file path
import glob

## Global Variables.  Modify as needed based on the model we're running inference with.
REG_NAME = 'SVR with RBF Kernel'
VARIANCE = 95       # Set this to 'False' if we don't use PCA.
MODEL_PARAMS = {'C': 36, 'epsilon': 0.1, 'gamma': 'scale'}

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
    peptide_sequences = pd.read_excel(PATH + '/Positive KI.xlsx')
    peptide_sequences = peptide_sequences.replace(r"^ +| +$", r"", regex=True)
    peptide_sequences = peptide_sequences[['Seq', 'KI (nM)']]
    peptide_sequences.rename(columns={'Seq':'Name'}, inplace=True)

    # Feature Extraction
    df = pd.DataFrame()
    for i in range(len(peptide_sequences)):
        df = pd.concat([df, inferenceSingleSeqence(peptide_sequences.iloc[i][0])])

    # Merging into a single dataframe. Removing extra seq column and others.
    df = pd.merge(df,peptide_sequences)
    df = df.drop(columns=['Seq','Helix','Turn','Sheet'])

    # Rescaling the dataframe in the log10 (-5,5) range.
    df['KI (nM) rescaled'], base_range  = rescale(df['KI (nM)'], destination_interval=(-5,5))

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

    # Apply MinMaxScaler to the initial x values.
    with open(PATH + '/Regression Only Results/regression only scaler.pkl', 'rb') as fh:
        scaler = pickle.load(fh)

    x = pd.DataFrame(scaler.transform(x), columns=df.columns[1:573])

    # Load the saved Sequential Forward Selection Model and apply the transform.
    with open(PATH + '/Regression Only Results/SFS for %s.pkl' %(REG_NAME), 'rb') as fh:
        sfs = pickle.load(fh)

    x = pd.DataFrame(sfs.transform(x), columns=sfs.get_feature_names_out())
    features = sfs.get_feature_names_out()

    # Extract the features to a .json
    features = features.tolist()
    with open('regression_only_selected_features.json', 'w') as outfile:
        json.dump(features, outfile)

    # Depending on the variance we select, apply PCA to the reduced feature set.
    if VARIANCE != False:
        with open(PATH + '/Regression Only Results/PCA for %s.pkl' %(REG_NAME), 'rb') as fh:
            pca=  pickle.load(fh)
        x = pd.DataFrame(pca.transform(x))

        # Dimensonality Reduction based on accepted variance.
        ratios = np.array(pca.explained_variance_ratio_)
        ratios = ratios[ratios.cumsum() <= (VARIANCE/100)]
        
        # Readjust the dimensions of x based on the variance we want.
        length = len(ratios)
        if length > 0:
            x = x[x.columns[0:length]]

    # Regression model
    model = SVR(kernel='rbf')
    model.set_params(**MODEL_PARAMS)
    model.fit(x,y)
    
    # Dump models as .pkl
    pickle.dump(model, open(PATH + '/Inference Models/regression only %s trained model.pkl' %(REG_NAME), 'wb'))
    
    ## Inference
    # Get the test data and the napply the necessary transforms.
    test_set = test_data()
    x_test = test_set[test_set.columns[1:573]]
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
    x_test = pd.DataFrame(sfs.transform(x_test), columns=sfs.get_feature_names_out())

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
            
    # Make the predictions
    y_pred = model.predict(x_test)

    y_pred_unscaled = unscale(y_pred, ki_range)
    test_set['KI (nM) Predicted'] = y_pred_unscaled

    # Save the results in a new .csv
    results = test_set[['Name','KI (nM) Predicted']]
    results.to_csv(PATH + '/Regression Only Results/reg only blind set predictions.csv')

    # Logger file for the KI_range for unscaling
    logger = log_files(PATH + '/Log Files/regression_only_ki_predictions.log')
    logger.info('Displaying KI Range below.')
    logger.info('--------------------------')
    logger.info(ki_range)

main()