#!/bin/bash

# =============================================================================
# MODEL COMPARISON JOB SCRIPT
# =============================================================================
# This script runs a model comparison using the specified configuration file.
# It is intended to be submitted to a SLURM cluster with GPU resources.
# The script loads the required Conda environment and executes the model_comparison.py script.
# =============================================================================

#SBATCH --job-name=model_comparison
#SBATCH --output=model_comparison_output_%j.out
#SBATCH --error=model_comparison_error_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate tp53

# Run the model comparison script with the desired configuration file

# python src/model_comparison.py --config configs/comparison_template.json

python src/model_comparison.py --config configs/comparison_target.json
