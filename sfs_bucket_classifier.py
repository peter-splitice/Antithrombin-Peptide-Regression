"""
This section of code covers the classifier which takes takes the input data and splits it into three separate buckets based on 
    KI (nM) values.
"""

# Importing Dependencies
from common_dependencies import *
import csv

# Model Persistence
from pickle import dump

# Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector

# Metrics
from sklearn.metrics import accuracy_score, make_scorer, matthews_corrcoef

# Preprocessing
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier

# Return file path
import glob

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

# Sequential Forward Selection for feature reduction in our data.
def sequential_selection(x, y, name, threshold, model=SVC()):
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
    ratios = np.arange(0.05, 0.55, 0.05)
    
    cols = ['Features Selected', 'Training Accuracy Score', 'Training MCC Score',
            'Validation Accuracy Score', 'Validation MCC Score']
    scores_df = pd.DataFrame(columns=cols)

    # Iterate through selecting from 10%-90% of the features in increments of 10.
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
        high_test_mcc = 0
        
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
        if (avg_test_mcc > high_test_mcc) or (ratio == 0.05):
            high_test_mcc = avg_test_mcc
            final_x_sfs = x_sfs
            final_sfs = sfs

        # Calculate the averages
        scores_df.loc[len(scores_df)] = [x_sfs.shape[1],train_accuracy_sum/FOLDS, train_mcc_sum/FOLDS, test_accuracy_sum/FOLDS, 
                                         test_mcc_sum/FOLDS]

    logger.info('Forward Selection Finished')

    scores_df.to_csv(PATH + '/%s/sfs/%s features selected with threshold %2.2f.csv' %(name, name, threshold))

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
    bestparams: Optimzied hyperparameters for the model that we are running the search on.

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
        logger.info('Training:')
        # Stratify!
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=(1/folds), random_state=seed, stratify=y)
        model.fit(x_train, y_train)

        logger.info('Training Finished.')

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
        #   as the SFS and PCA models/
        attributes = param_name_model_zipper()
        vars = [75, 80, 85, 90, 95, 100]
        cols = ['Name', 'Stage', 'Features', 'Train MCC', 'Train Stdev', 'Test MCC', 'Test Stdev', 'Params']
        
        for params, name, model in attributes:
            # Every time I iterate through this loop, I need to recreate x.
            x = df[df.columns[1:573]]
            x = pd.DataFrame(scaler.transform(x), columns=df.columns[1:573])
            logger.info('%s Results:\n' %(name))

            # Create a the directories for the models if they doesn't exist.
            if os.path.exists(PATH + '/%s' %(name)) == False:
                os.mkdir('%s' %(name))
            if os.path.exists(PATH + '/%s/sfs-pca' %(name)) == False:
                os.mkdir('%s/sfs-pca' %(name))
            if os.path.exists(PATH + '/%s/sfs' %(name)) == False:
                os.mkdir('%s/sfs' %(name))    
            if os.path.exists(PATH + '/%s/baseline' %(name)) == False:
                os.mkdir('%s/baseline' %(name))
            if os.path.exists(PATH + '/%s/results' %(name)) == False:
                os.mkdir('%s/results' %(name))

            model_scores = pd.DataFrame(columns=cols)

            # Our main pipeline is Initial Hyperparameter Tuning -> Sequential Selection -> Principal Component Analysis -> Hyperparameter Tuning

            # Baseline
            model, scores_baseline = classifier_trainer(x, y, params, model)
            model_scores.loc[len(model_scores)] = [name, 'Baseline', len(x.columns[:]), scores_baseline[0], scores_baseline[1], scores_baseline[2], 
                                                   scores_baseline[3], scores_baseline[4]]

            # Sequential Feature Selection
            x_sfs, sfs = sequential_selection(x, y, name, threshold, model)
            logger.info('SFS only results:\n')

            # SFS Block
            model_sfs, scores_sfs = classifier_trainer(x_sfs, y, params, model)
            model_scores.loc[len(model_scores)] = [name, 'SFS', len(x_sfs.columns[:]), scores_sfs[0], scores_sfs[1], scores_sfs[2], 
                                                   scores_sfs[3], scores_sfs[4]]
            dump(sfs, open(PATH + '/%s/sfs/%s %2.2f sfs.pkl' %(name, name, threshold), 'wb'))

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

            model_scores.to_csv(PATH + '/%s/results/%s scores with threshold %2.2f.csv' %(name, name, threshold))
            dump(pca, open(PATH + '/%s/sfs-pca/%s %2.2f sfs-pca.pkl' %(name, name, threshold), 'wb'))
    
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

    # In a for loop, create a directory for the 3 models and then deposit the hyperparameter tuning results as well
    #   as the SFS and PCA models/
    attributes = zip(all_params, names, models)

    return attributes

# Function that finds the initial set of hyperparameters for the classifiers, pre-data transformation.
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
    df = import_data(threshold)

    # Create the x and y values.  X = all the features.  y = the columns of buckets
    x = df[df.columns[1:573]]
    y = df['Bucket']

    # Add MinMaxScaler here.  Data seems to be overfitting.
    scaler = MinMaxScaler()
    scaler.fit(x)
    x = pd.DataFrame(scaler.transform(x), columns=df.columns[1:573])

    # Create the feature set for the 3 classifiers.  adjustments to C here too
    rbf_params = {'gamma': [1e-1,1e-2,1e-3,1e-4,'scale','auto'], 'C': np.arange(1,101,5),
                  'class_weight': [None,'balanced'], 'break_ties': [False,True]}
    xgb_params = {'max_depth': np.arange(2,11,1), 'n_estimators': np.arange(1,25,1), 'gamma': np.arange(0,4,1),
                  'subsample': [0.5,1], 'lambda': [1,5,9], 'alpha': np.arange(0,1.1,0.2)}
    rfc_params = {'criterion': ['gini','entropy'], 'max_features': ['sqrt','log2',1.0,0.3], 'ccp_alpha': np.arange(0,0.3,0.1),
                  'n_estimators': np.arange(1,25,1), 'max_depth': np.arange(2,11,1)}
    knn_params = {'n_neighbors': np.arange(1,55,2), 'weights': ['uniform', 'distance'], 'leaf_size': np.arange(5,41,2),
                  'p': [1, 2], 'keepdims': [False,True]}
    all_params = [rbf_params, xgb_params, rfc_params, knn_params]

    # Models and names
    models = [SVC(), XGBClassifier(), RandomForestClassifier(), KNeighborsClassifier()]
    names = ['SVC with RBF Kernel', 'XGBoost Classifier', 'Random Forest Classifier', 'KNN Classifier']
    
    logger.info('Hyperparameter Tuning for Threshold %2.2f:\n' %(threshold))
    # Classifier Training for all 3 classifiers.
    for name, model, params in zip(names, models, all_params):
        logger.info('GridSearchCV on %s:\n' %(name))
        if os.path.exists(PATH + '/%s/' %(name)) == False:
            os.mkdir('%s' %(name))
        if os.path.exists(PATH + '/%s/Initial Hyperparameter Tuning' %(name)) == False:
            os.mkdir('%s/Initial Hyperparameter Tuning' %(name))
        results, model = classifier_trainer(x, y, params, model=model)
        results.to_csv(PATH + '/%s/Initial Hyperparameter Tuning/%s Initial Hyperparameter Tuning at Threshold %2.2f.csv'
                       %(name, name, threshold))

def graph_results():
    """
    This function graphs the results we already have stored.  I created this to allow for graph formatting independent of
        the model analysis.  There are no arguments passed into this this function or taken from this function.
    """
    thresholds = [0.01, 0.1, 0.5, 5, 10, 15, 18]
    names = ['SVC with RBF Kernel', 'XGBoost Classifier', 'Random Forest Classifier', 'KNN Classifier']
    for threshold in thresholds: 
        for name in names:
            # Load the data.
            scores_df = pd.read_csv(PATH + '/%s/sfs/%s features selected with threshold %2.2f.csv' %(name, name, threshold))

            # Display the results:
            plt.figure()
            plt.plot(scores_df['Features Selected'], scores_df['Training MCC Score'])
            plt.plot(scores_df['Features Selected'], scores_df['Validation MCC Score'], '-.')
            plt.xlabel('Number of features selected')
            plt.ylabel('Cross Validation Score (MCC)')
            plt.title('Forward Selection for %s at Threshold %2.2f' %(name, threshold))
            plt.legend(['Training MCC', 'Validation MCC'])
            plt.savefig(PATH + '/Figures/%s/Forward Selection for %s at Threshold %2.2f.png' %(name, name, threshold))
            plt.close()

# Use argparse to pass various thresholds.
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--threshold', help='threshold = set the threshold to split the dataset into'
                    ' large and small buckets', type=float)
parser.add_argument('-ht', '--hyperparameter_test', help='hyperparameter_test = test for various hyperparameters',
                    type=float)
parser.add_argument('-reg', '--regressor', help='regressor = perform the regression section of the code once we have finished with '
                    'the classification section of the pipeline', action='store_true')
parser.add_argument('-st', '--store_file', help='store_file = stores the extracted files into a csv', action='store_true')
parser.add_argument('-gr', '--grapher', help='grapher = graphs the results generated from the classification training',
                    action='store_true')
           
args = parser.parse_args()

threshold = args.threshold
hyperparameter_test = args.hyperparameter_test
extract_file = args.store_file
grapher = args.grapher

## Initialize the logger here after I get the threshold value.  Then run the classifier
if threshold != None:
    if os.path.exists(PATH + '/SFS Bucket Classifier Logs') == False:
        os.mkdir('SFS Bucket Classifier Logs')
    logger = log_files(PATH + '/SFS Bucket Classifier Logs/Threshold %2.2f.log' %(threshold))
    threshold_finder(threshold)
elif hyperparameter_test != None:
    logger = log_files(PATH + '/Log Files/HP_Test.log')
    hyperparameter_pipeline(hyperparameter_test)
elif extract_file == True:
    df = import_data(threshold=10)
elif grapher == True:
    graph_results()
