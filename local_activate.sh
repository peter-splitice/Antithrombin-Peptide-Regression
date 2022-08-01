for n in 0.01 0.1 0.5 2 3 5 7 10 15 17 18 20;
do 
    echo 'Threshold level' $n 'starting'
    python ./bucket_classifier.py -t $n
    echo 'Threshold level' $n 'complete'
    echo ''
done