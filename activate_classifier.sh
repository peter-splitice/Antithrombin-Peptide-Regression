#!/bin/bash
#
#SBATCH --job-name=positive_peptide_bucket_classifier
#SBATCH --output=bucket_classifier.log
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-user=peter.v.pham@sjsu.edu
#SBATCH --mail-type=END
export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

for n in 1e-2 1e-1 2 5 7 10 12 15 18 20;
do 
    echo 'Threshold level' $n 'starting'
    python bucket_classifier.py -t $n
    echo 'Threshold level' $n 'complete'
    echo ''
done