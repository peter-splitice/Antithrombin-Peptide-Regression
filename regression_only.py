"""
This script handles regression if there is no bucketizing of the data.  Note that I'll have to do 
feature reduction and PCA in this mode."""

## Imports
from common_dependencies import *

# Preprocessing and feature reduction
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

# Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector, RFECV

# Models
from sklearn.linear_model import Lasso
from sklearn.svm import SVR

# Metrics
from sklearn.metrics import mean_squared_error

# Model Persistence
import pickle

def import_data():
    """
    Import the full dataset fro mthe current path.  Also apply some of the necessary preprocessing.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    df: Dataframe of the full KI training dataset.
    
    base_range: Caontains the range of values within the dataframe for rescaling purposes.
    """

    # Extracting peptide sequence + formatting
    peptide_sequences = pd.read_excel(PATH + '/Positive KI.xlsx')
    peptide_sequences = peptide_sequences.replace(r"^ +| +$", r"", regex=True)
    peptide_sequences = peptide_sequences[['Seq', 'KI (nM)']]
    peptide_sequences.rename(columns={'Seq':'Name'}, inplace=True)

    # Feature Extraction
    df = pd.DataFrame()
    for i in range(len(peptide_sequences)):
        print(peptide_sequences.iloc[i][0])
        df = pd.concat([df, inferenceSingleSeqence(peptide_sequences.iloc[1][0])])

    # Merging into a single dataframe. Removing extra seq column and others.
    df = pd.merge(df, peptide_sequences)
    df = df.drop(columns=['Seq','Helix','Turn','Sheet'])

    # Rescaling the dataframe in the log10 (-5,5) range.
    df['KI (nM) rescaled'], base_range  = rescale(df['KI (nM)'], destination_interval=(-5,5))

    return df, base_range

def load_regression_models():
    """
    This function will create a list with its set of models, params, and names
    
    Parameters
    ----------
    None
    
    Returns
    -------
    reg_models: list for our models, containing the name, models, and a dict of hyperparameters to
    tune
    """
    # Hyperparameters
    rbf_params = {'gamma': ['scale', 'auto'], 'C': np.arange(1,101,5), 'epsilon': np.arange(0.1, 1, 0.1)}
    lin_params = {'gamma': ['scale', 'auto'], 'C': np.arange(1,101,5), 'epsilon': np.arange(0.1, 1, 0.1)}
    las_params = {'alpha': [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4], 'selection': ['cyclic', 'random']}

    # names and hyperparameters to sort through are shared.
    names = ['SVR with RBF Kernel', 'SVR with Linear Kernel', 'Lasso Regression']
    params = [rbf_params, lin_params, las_params]

    # I need to instantiate new models for both the small and medium buckets.
    models = [SVR(kernel='rbf'), SVR(kernel='linear'), Lasso()]

    # Each model set has its own distinct range i want to run Sequential Forward Selection with.
    fs_ranges = [(0.05, 0.11), (0.30, 0.35), (0.05, 0.10)]

    reg_models = list(zip(names, params, models, fs_ranges))

    return reg_models    

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
    bestparams: Optimized hyperparameters for the model that we are running the search on.
    """

    logger.debug('GridSearchCV Starting:')
    logger.debug('-----------------------\n')

    reg = GridSearchCV(model, param_grid=params, scoring='neg_root_mean_squared_error', cv=5, 
                       return_train_score=True, n_jobs=-1)
    reg.fit(x,y)

    # Showing the best parameters found on the development set.
    logger.info('Best parameter set: %s' %(reg.best_params_))
    logger.info('-------------------------\n')

    # Save the best parameters.
    bestparams = reg.best_params_

    return bestparams

# Sequential Forward Selection for feature reduction in our data.
def sequential_selection(x, y, name, ki_range, fs_range, model=SVR()):
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

    final_sfs: The SequentialFeatureSelector model selected.  Automatically select the best one
    """

    # Fit a feature selector to SVM w/RBF kernel classifier and use the 'accuracy' score.
    # Forward Selection Loop
    logger.info('Forward Selection Starting')
    ratios = np.arange(fs_range[0], fs_range[1], 0.01)

    cols = ['Features Selected', 'Training RMSE Score', 'Training Log RMSE Score',
            'Validation RMSE Score', 'Validation RMSE Score']
    scores_df = pd.DataFrame(columns=cols)
    low_test_log_rmse=0

    # Iterate through selecting from 10%-90% of the features in increments of 10.
    for ratio in ratios:
        sfs = SequentialFeatureSelector(model, n_jobs=-1, scoring='neg_root_mean_squared_error',
                                        n_features_to_select=ratio, direction='forward')
        sfs.fit(x, y)
        x_sfs = pd.DataFrame(sfs.transform(x), columns=sfs.get_feature_names_out())
        
        # Initialize measurements.
        train_rmse_sum = 0
        train_log_rmse_sum = 0
        test_rmse_sum = 0
        test_log_rmse_sum = 0
        
        # Kfold to test the results of sequental feature selection.
        kf = KFold(n_splits=FOLDS, random_state=RAND, shuffle=True)
        fold = 0

        for train_index, test_index in kf.split(x_sfs,y):
            fold +=1
            
            x_train, x_test = x_sfs.loc[train_index], x_sfs.loc[test_index]
            y_train_log, y_test_log = y[train_index], y[test_index]

            model.fit(x_train, y_train_log)

            # Predicting on the test and training sets.  We get the log of the predictions here.
            y_test_log_pred = model.predict(x_test)
            y_train_log_pred = model.predict(x_train)

            # Unscaling the data:
            y_test_pred = unscale(y_test_log_pred, ki_range)
            y_test = unscale(y_test_log, ki_range)
            y_train_pred = unscale(y_train_log_pred, ki_range)
            y_train = unscale(y_train_log, ki_range)
            
            # Calculate rmse for the training set
            train_log_rmse = mean_squared_error(y_train_log, y_train_log_pred)**0.5
            train_rmse = mean_squared_error(y_train, y_train_pred)**0.5

            # Calculate rmse for the test set
            test_log_rmse = mean_squared_error(y_test_log, y_test_log_pred)**0.5
            test_rmse = mean_squared_error(y_test, y_test_pred)**0.5

            # Add to the sums
            train_rmse_sum += train_rmse
            train_log_rmse_sum += train_log_rmse
            test_rmse_sum += test_rmse
            test_log_rmse_sum += test_log_rmse

        avg_test_log_rmse = test_log_rmse_sum/FOLDS
        if (avg_test_log_rmse < low_test_log_rmse) or (ratio == fs_range[0]):
            low_test_log_rmse = avg_test_log_rmse
            final_x_sfs = x_sfs
            final_sfs = sfs

        # Calculate the averages
        scores_df.loc[len(scores_df)] = [x_sfs.shape[1], train_rmse_sum/FOLDS, train_log_rmse_sum/FOLDS, 
                                         test_rmse_sum/FOLDS, test_log_rmse_sum/FOLDS]

    logger.info('Forward Selection Finished')

    scores_df.to_csv(PATH + '/Regression Only Results/%s features selected.csv' %(name))

    return final_x_sfs, final_sfs

# Perform optimization on the classifier as well as a k-fold cross validation.
def regressor_trainer(x, y, ki_range, params, model=SVR()):
    """
    This function does the following:
    - Hyperparameter Tuning
    - Fitting (with kfold cross validation)
    - Predictions

    Parameters
    ----------
    x: Reduced set of input values.

    y: Output KI values that we are using for the training and validation sets.

    ki_range:  Initial range of KI values so that we can use for unscaling.

    params: List of hyperparameters we will be doing hyperparameter tuning with.

    model: Our model that we are optimizing hyperparameters for.

    Returns
    -------
    model: Modfied model that has the optimized hyperparameters.

    scores: Pandas DataFrame of the training and test scores.
    """
    # Train our model
    i = 0

    # Initialize measurements.
    train_rmse_sum = 0
    train_log_rmse_sum = 0
    test_rmse_sum = 0
    test_log_rmse_sum = 0

    optimized_features = hyperparameter_optimizer(x, y, params, model)

    model.set_params(**optimized_features)

    # Kfold Cross-Validation

    kf = KFold(n_splits=FOLDS, random_state=RAND, shuffle=True)

    for train_index, test_index in kf.split(x,y):
        i += 1
        logger.info('Training:')
        # Stratify!
        x_train, x_test = x.loc[train_index], x.loc[test_index]
        y_train_log, y_test_log = y[train_index], y[test_index]
        model.fit(x_train, y_train_log)

        logger.info('Training Finished.')

        # Predicting on the test and training sets.  We get the log of the predictions here.
        y_test_log_pred = model.predict(x_test)
        y_train_log_pred = model.predict(x_train)

        # Unscaling the data:
        y_test_pred = unscale(y_test_log_pred, ki_range)
        y_test = unscale(y_test_log, ki_range)
        y_train_pred = unscale(y_train_log_pred, ki_range)
        y_train = unscale(y_train_log, ki_range)

        # Calculate rmse for the training set
        train_log_rmse = mean_squared_error(y_train_log, y_train_log_pred)**0.5
        train_rmse = mean_squared_error(y_train, y_train_pred)**0.5

        # Calculate rmse for the test set
        test_log_rmse = mean_squared_error(y_test_log, y_test_log_pred)**0.5
        test_rmse = mean_squared_error(y_test, y_test_pred)**0.5

        # Log the individual folds
        logger.info('Log Training RMSE: %3.3f, Training RMSE: %3.3f, Log Test RMSE: %3.3f, '
                    'Test RMSE MCC: %3.3f, Fold: %i'
                    %(train_log_rmse, train_rmse, test_log_rmse, test_rmse, i))
        
        # Add to the sums
        train_rmse_sum += train_rmse
        train_log_rmse_sum += train_log_rmse
        test_rmse_sum += test_rmse
        test_log_rmse_sum += test_log_rmse        

    # Calculate the averages
    train_rmse_avg = train_rmse_sum/FOLDS
    train_log_rmse_avg = train_log_rmse_sum/FOLDS
    test_rmse_avg = test_rmse_sum/FOLDS
    test_log_rmse_avg = test_log_rmse_sum/FOLDS

    # Log the average scores for all the folds
    logger.info('AVG Log Training RMSE: %3.3f, AVG Training RMSE: %3.3f, AVG Log Test RMSE: %3.3f, '
                'AVG Test RMSE: %3.3f\n' %(train_log_rmse_avg, train_rmse_avg, test_log_rmse_avg, 
                                           test_rmse_avg))
    
    scores = [train_rmse_avg, test_rmse_avg, train_log_rmse_avg, test_log_rmse_avg, optimized_features]
    
    return model, scores

def variance_analyzer(x_pca, variance, pca):
    """
    Perform PCA and return the transformed inputs with the principal components.

    Parameters
    ----------
    x_pca: The input matrix after we applied PCA to reduce dimensionality.

    variance: Parameter that reduces dimensionality of the PCA.  Enter as an int from 0-100.  100 keeps full dimensionality

    Returns
    -------
    x_pca_reduced: The input matrix, further reduced based on how much variance we want accounted for in the principal
        components.

    pca: The PrincipalComponentAnalysis model.
    
    """
    
    # Dimensonality Reduction based on accepted variance.
    ratios = np.array(pca.explained_variance_ratio_)
    ratios = ratios[ratios.cumsum() <= (variance/100)]
    
    # Readjust the dimensions of x based on the variance we want.
    length = len(ratios)
    if length > 0:
        logger.info('Selecting %i principal components making up %i%% of the variance.\n' %(length,variance))
        x_pca = x_pca[x_pca.columns[0:length]]
    # A length of zero indicates that the one principal component accounts for *all* of the variance.
    else:
        logger.info('Kept all principal components for %i%% of the variance.\n' %(variance))
    logger.info('PCA Finished')

    return x_pca


def regression():
    """
    This function is responsible for the initial phase of hyperparameter tuning for the regression
        section of the pipeline.  The models used will be:
        -> SVR with RBF Kernel
        -> SVR with Linear Kernel
        -> Lasso Regression
    """
    # Create the directory for the regression results.
    if os.path.exists(PATH + '/Regression Only Results') == False:
        os.mkdir('Regression Only Results')

    # Load up the regression models here:
    reg_models = load_regression_models()
    variances = [75, 80, 85, 90, 95, 100]
    cols = ['Name', 'Stage', 'Features', 'Training RMSE', 'Validation RMSE', 'Log Training RMSE',
            'Log Validation RMSE', 'Parameters']
    
    results_df = pd.DataFrame(columns=cols)
 
    # Extract the X and Y information
    df, ki_range = import_data()
    x = df[df.columns[1:573]]
    y = df['KI (nM) rescaled']

    # Always do MinMaxScaler first
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = pd.DataFrame(scaler.transform(x), columns=df.columns[1:573])
    pickle.dump(scaler, open(PATH + '/Regression Only Results/regression only scaler.pkl', 'wb'))
            
    for name, params, model, fs_range in reg_models:
        # Data import whenever you switch models
        logger.info('Logging results for %s:' %(name))
        logger.info('-----------------------------------------------------------\n')
        x = df[df.columns[1:573]]
        x = pd.DataFrame(scaler.transform(x), columns=df.columns[1:573])

        # Optimizing hyperparameters for the baseline model:
        logger.info('Calculations for the baseline model:')
        logger.info('------------------------------------\n')

        model, scores_baseline = regressor_trainer(x, y, ki_range, params, model)

        results_df.loc[len(results_df)] = [name, 'Baseline', x.shape[1], scores_baseline[0],
                                           scores_baseline[1], scores_baseline[2], scores_baseline[3], scores_baseline[4]]

        # Apply Feature Selection.  Use SFS for Linear and Lasso
        x_sfs, sfs = sequential_selection(x, y, name, ki_range, fs_range, model)
        pickle.dump(sfs, open(PATH + '/Regression Only Results/SFS for %s.pkl' %(name), 'wb'))
        model_sfs, scores_sfs = regressor_trainer(x_sfs, y, ki_range, params, model)

        results_df.loc[len(results_df)] = [name, 'Feature Selection', x_sfs.shape[1], scores_sfs[0],
                                           scores_sfs[1], scores_sfs[2], scores_sfs[3], scores_sfs[4]]
        
        ## Then we run PCA.
        logger.info('PCA Starting')
        pca = PCA()
        pca.fit(x_sfs)
        x_sfs_pca = pd.DataFrame(pca.transform(x_sfs))
        pickle.dump(pca, open(PATH + '/Regression Only Results/PCA for %s.pkl' %(name), 'wb'))

        # Select principal components based on how much variance we want to account for, and then
        #   pick out the best performing variance percentage (up to 100)
        for variance in variances:
            x_sfs_pca_var = variance_analyzer(x_sfs_pca, variance, pca)
            model_sfs_pca, scores_sfs_pca = regressor_trainer(x_sfs_pca_var, y, ki_range, params, model)
            results_df.loc[len(results_df)] = [name, 'PCA w/%i%% variance' %(variance), x_sfs_pca_var.shape[1], scores_sfs_pca[0],
                                               scores_sfs_pca[1], scores_sfs_pca[2], scores_sfs_pca[3], scores_sfs_pca[4]]
        
    results_df.to_csv(PATH + '/Results/regression_only_results.csv')

def graph_results():
    """
    From here, we'll visualize hyperparameter tuning of our best selected model.  In this case, we will
        use SVR with RBF Kernel using Sequential Feature Selection and PCA @ 85% variance accounted for 
        in the number of principal components.
    """
    # Initializations.  Change this code whenever the best model changes
    model = SVR(kernel='rbf')
    params = {'gamma': ['scale', 'auto'], 'C': np.arange(1,101,5), 'epsilon': np.arange(0.01, 1.01, 0.05)}
    variance = 85
    name = 'SVR with RBF Kernel'

    # Extract the X and Y information
    df, ki_range = import_data()
    x = df[df.columns[1:573]]
    y = df['KI (nM) rescaled']

    ## Pipeline
    # Scaler transformation
    with open(PATH + '/Regression Only Results/regression only scaler.pkl', 'rb') as fh:
        scaler = pickle.load(fh)
    x = pd.DataFrame(scaler.transform(x), columns=df.columns[1:573])

    # Sequential Forward Selection
    with open (PATH + '/Regression Only Results/SFS for %s.pkl' %(name), 'rb') as fh:
        sfs = pickle.load(fh)

    x = pd.DataFrame(sfs.transform(x), columns=sfs.get_feature_names_out())
    
    # Principal Component Analysis
    with open(PATH + '/Regression Only Results/PCA for %s.pkl' %(name), 'rb') as fh:
        pca = pickle.load(fh)

    # Depending on the variance we select, paply PCA to the reduced feature set.
    if variance != False:
        x = pd.DataFrame(pca.transform(x))

        # Dimensonality Reduction based on accepted variance.
        ratios = np.array(pca.explained_variance_ratio_)
        ratios = ratios[ratios.cumsum() <= (variance/100)]
        
        # Readjust the dimensions of x based on the variance we want.
        length = len(ratios)
        if length > 0:
            x = x[x.columns[0:length]]

    # Apply gridsearchcv
    grid = GridSearchCV(model, param_grid=params, scoring='neg_root_mean_squared_error', cv=5, 
                       return_train_score=True, n_jobs=-1)
    grid.fit(x,y)

    # Save the acquired information into another dataframe and then output as file.
    df_param_combos = pd.DataFrame(grid.cv_results_)
    df_param_combos.to_csv(PATH + '/Regression Only Results/GridSearch Data.csv')

    gammas = ('auto', 'scale')

    # Creating plots for both of the 'auto' and 'scale' gammas
    for gamma in gammas:
        # x y and z
        x_grid = df_param_combos['param_C'][df_param_combos['param_gamma'] == gamma]
        y_grid = df_param_combos['param_epsilon'][df_param_combos['param_gamma'] == gamma]
        z_grid = df_param_combos['mean_test_score'][df_param_combos['param_gamma'] == gamma]

        # Create the figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projections='3d')
        ax.scatter(x_grid, y_grid, z_grid)
        ax.set_title('Dot plot of RMSE given C and Epsilon values for %s Gamma Parameter' %(gamma))
        ax.set_xlabel('C')
        ax.set_ylabel('Epsilon')
        ax.set_zlabel('Mean Validation RMSE')

        # Save and clear figure.
        fig.savefig(PATH + '/Figures/RMSE plot for gamma %s.fig' %(gamma))
        fig.clf()




# Argument parser section.
parser = argparse.ArgumentParser()
parser.add_argument('-reg', '--regressor', help='regressor = applies regression and finds the optimal '
                    'models with hyperparameters.', action='store_true')
parser.add_argument('-gr', '--grapher', help='grapher = create plots for all of our results and save '
                    'them.', action='store_true')

args = parser.parse_args()

regressor = args.regressor
grapher = args.grapher

if regressor == True:
    logger = log_files(PATH + '/Log Files/regression_only.log')
    regression()
if grapher == True:
    graph_results()