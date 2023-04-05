## Update Script
git add "__pycache__"

# Git Control
git add ".gitignore"
git add ".vscode"

# Folders
git add "Figures"
git add "Inference Models"
git add "KNN Classifier"
git add "Log Files"
git add "Negative Peptides"
git add "Positive Peptides"
git add "Random Forest Classifier"
git add "Regression Only Results"
git add "Results"
git add "RFE Bucket Classifier Logs"
git add "SFS Bucket Classifier Logs"
git add "SVC with RBF Kernel"
git add "SVC with Linear Kernel"
git add "XGBoost Classifier"

# bash scripts
git add add_git.sh                  # Update script
git add rfe_activation.sh
git add sfs_activation.sh

## Python Scripts
git add analysis.py
git add common_dependencies.py

git add rfe_bucket_classifier.py
git add rfe_regression.py
git add rfe_ki_predictor.py

git add sfs_bucket_classifier.py
git add sfs_regression.py
git add sfs_ki_predictor.py
git add sfs_range_narrower.py

git add regression_only.py
git add regression_only_ki_predictor.py

# Text Files
git add changelog.md
git add README.md
git add requirements.txt
git add rfe_selected_features.json
git add sfs_selected_features.json
git add regression_only_feastures.json

# Imports
git add combined_hits.csv           # Input file for our test set.
git add Positive_Ki.xlsx            # Initial peptides for the regression training/validation sets.

# Testbench
git add testbench.ipynb

# push the update.
git commit -m '4/3/2023: Added graphing function to regression_only.py..'
# git push -u origin main