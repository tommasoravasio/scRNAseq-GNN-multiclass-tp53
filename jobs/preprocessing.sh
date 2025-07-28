#!/bin/bash
#SBATCH --job-name=preprocessing_tp53
#SBATCH --output=preprocessing_output_%j.out
#SBATCH --error=preprocessing_error_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=defq


module load miniconda3
eval "$(conda shell.bash hook)"
conda activate tp53


# python src/preprocessing.py --feature_selection target --batch_correction combat --local_testing 
python src/preprocessing.py --feature_selection target --batch_correction harmony 