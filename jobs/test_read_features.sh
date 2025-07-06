#!/bin/bash
#SBATCH --job-name=test_read_features
#SBATCH --output=test_read_features_%j.out
#SBATCH --error=test_read_features_%j.err
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

source /software/miniconda3/etc/profile.d/conda.sh
conda activate test_pandas

python src/test_read_features.py 