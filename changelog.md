## 12/23/2022 Changes:
- Changed the code to have two separate branches of scripts.
    * Branch 1 uses Recursive Feature Elimination to apply feature reduction.
    * Branch 2 uses Sequential Forward Selection to apply feature reduction.
- Each branch has the following scripts:
    * Bucket classifier script
    * Regression script
    * KI predictor script
    * Branch activation bash script

## 12/24/2022 Changes:
- Added a dependencies script "common_dependencies.py" for both the SFS and RFE branches to clean up the code.
- Further cleaned up the code and removed anything unnecessary from each function.

## 12/29/2022 Changes:
- Created regression inference models for SVR with Linear Kernel and SVR with RBF Kernel.

## 1/2/2023 Changes
- Changed rfe_ki_predictor.py
    * Fixed bug where I created "Lasso" instead of "SVR with RBF Kernel" models in the trained inference files.

## 1/12/2023 Changes
- Changed sfs_regression.py
    * When running the 'regressor' argparse, I moved calling the graph_results() function to be after running the regression() function instead of running it last within the regression() function.
    * Modified formatting of the graphing within the graph_results() function.
- Changed sfs_activation.sh
    * Updated to incorporate the graphing function within the sfs_regression.py script.
- Changed rfe_bucket_classifier.py
    * Changed rfe_grapher() y label from "Test MCC" to "Validation MCC."
- Changed sfs_bucket_classifier.py
    * Added a function graph_results() with a corresponding argparse to separate graphing the data from gathering and saving the data.  This enables for easier modification of the graphs.
- Changed sfs_range_narrower.py
    * Added a function graph_results() with a corresponding argparse to separate graphing teh data from gathering and saving the data.  This enables for easier modification of the graphs.
- Changed rfe_regression.py
    * When running the 'regressor' argparse, I moved calling the graph_results() function to be after running the regression() function instead of running it last within the regression() function.
    * Modified formatting of the graphing within the graph_results() function.

## 3/2/2023 Changes
- Created regression_only.py
    * This script does regression only without a classification section that splits the data into buckets.
- Changed rfe_regression.py
    * Cleaned up some formatting
    
## 3/17/2023 Changes
- Finalized the ki predictor on the regression only pipeline.
- Saved the models created for inference.

## 3/30/2023 Changes
- Changed the variance on the regression only model to 85

## 4/3/2023 Changes
- Added graphing fucntion to regression_only.py that graphs the all of the resutls from hyper-
    parameter tuning on the best model that we chose.  No changes were made to the final selected
    model

## 4/20/2023 Changes
- Added two new files:  'Positive KI.xlsx' and 'Positive Peptides with ref.xlsx'.  This serves as
    the new data that we are building our models on.

## 5/20/2023 Changes
- Fixed a bug in the bucket classifier.