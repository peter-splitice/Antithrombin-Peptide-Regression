for n in 0.01 0.1 0.5 5 10 15 18;
do
    echo 'Threshold level' $n 'starting'
    python ./rfe_bucket_classifier.py -t $n
    echo 'Threshold level' $n 'complete'
    echo ''
done
python ./rfe_bucket_classifier.py -gr
python ./rfe_regression.py -reg
# other thresholds:  0.01 0.1 0.5 5 10 15 18