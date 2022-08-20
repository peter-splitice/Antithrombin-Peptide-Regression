## Version 1.1
- Changed threshold input from positional type to optional type.
- Added optional input argument "seed"
- Modified functions to have a dynamic seed value.
- Added multithreading functionality

## Version 1.2
- Removed optional input argument "seed."
- Added 5 different seed values to create 5-fold cross-validation equivalent.

## Version 1.3
- Removed the threshold of 75,000 nM
- Changed the bucket classifer to a 3 class classifier.  One threshold is dynamic to
    split between large and small buckets.  The other threshold is set to 4000 KI (nM)
    to throw out KI values that are too high to bother with regression.
- Added a GridSearchCV stage after Forward Selection and PCA to optimize hyperparameters.
- Pipeline is now FS -> PCA -> GS -> CV

## Version 1.4
- Added MinMaxScaler to beginning stage of pipeline.
