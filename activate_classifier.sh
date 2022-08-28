#!/bin/bash
#
#SBATCH --job-name=positive_peptide_bucket_classifier
#SBATCH --output=bucket_classifier.log
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-user=peter.v.pham@sjsu.edu
#SBATCH --mail-type=END
export OMP_NUM_THREADS=28
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

module load python3

source /home/007457801/antithrombin/env/bin/activate

for n in 0.01 0.1 0.5 5 10 15 18;
do 
    echo 'Threshold level' $n 'starting'
    python3 ./bucket_classifier.py -t $n
    echo 'Threshold level' $n 'complete'
    echo ''
done

deactivate