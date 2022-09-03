for n in 0.01 0.1 0.5 5 10 15 18;
do 
    echo 'Threshold level' $n 'starting'
    #python ./bucket_classifier.py -t $n
    python ./bucket_classifier.py -pca $n
    echo 'Threshold level' $n 'complete'
    echo ''
done

# other thresholds:  0.01 0.1 0.5 5