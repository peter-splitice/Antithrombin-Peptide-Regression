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
- Added a dependencies script to both the SFS and RFE branches to clean up the code.
- Further cleaned up the code and removed anything unnecessary from each function.

## 12/29/2022 Changes:
- Created regression inference models for SVR with Linear Kernel and SVR with RBF Kernel.

## 1/2/2022 Changes
- Changed rfe_ki_predictor.py
    * Fixed bug where I created "Lasso" instead of "SVR with RBF Kernel" models in the trained inference files.