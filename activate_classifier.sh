#!/bin/bash
#
#SBATCH --job-name=positive_peptide_classifier
#SBATCH --output=positive_peptide_classifier-srun.log
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-user=007457801@sjsu.edu
#SBATCH --mail-type=END
export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

echo t