## Update Script

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
git add "Results"
git add "RFE Bucket Classifier Logs"
git add "RFE Extracted Features"
git add "SFS Bucket Classifier Logs"
git add "SFS Extracted Features"
git add "SVC with RBF Kernel"
git add "SVC with Linear Kernel"
git add "XGBoost Classifier"

# bash scripts
git add activate_hpc.sh             # Action script for HPC
git add activate_interactive.sh     # HPC interactive script
git add activate_pc.sh              # Activation script for PC
git add activate_mac.sh             # Activation script for Mac
git add add_git.sh                  # Update script

# Python Scripts
git add analysis.py
git add ki_predictor.py
git add ratio_finder.py
git add rfe_bucket_classifier.py
git add rfe_regression.py
git add sfs_bucket_classifier.py
git add sfs_regression.py

# Text Files
git add changelog.md
git add README.md
git add requirements.txt
git add selected_features.json

# Imports
git add combined_hits.csv           # Input file for our test set.
git add Positive_Ki.xlsx            # Initial peptides for the regression training/validation sets.

# Testbench
git add testbench.ipynb

# push the update.
git commit -m 'Update Version 12/3/2022'
# git push -u origin main