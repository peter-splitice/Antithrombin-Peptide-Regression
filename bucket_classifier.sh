# Set the filename
for n in 3 5 7 10 12 15 18 20;
do 
    echo 'Threshold level' $n 'starting'
    python bucket_classifier.py $n
    echo 'Threshold level' $n 'complete'
    echo ''
done