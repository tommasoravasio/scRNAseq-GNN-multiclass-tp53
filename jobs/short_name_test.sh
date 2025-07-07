#!/bin/bash
#SBATCH --job-name=short_name_test
#SBATCH --output=short_name_test_%j.out
#SBATCH --error=short_name_test_%j.err
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:05:00

source /software/miniconda3/etc/profile.d/conda.sh
conda activate tp53

python src/short_name_test.py 