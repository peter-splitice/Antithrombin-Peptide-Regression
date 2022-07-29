# Set the filename
for n in 3 5 7 10 12 15 18 20;
do 
    echo 'Threshold level' $n 'starting'
    for s in 33 42 55 68 74;
    do
        echo 'Seed value' $s 'starting'
        python bucket_classifier.py -t $n -s $s
        echo 'Seed value' $s 'complete'
    done
    echo 'Threshold level' $n 'complete'
    echo ''
done