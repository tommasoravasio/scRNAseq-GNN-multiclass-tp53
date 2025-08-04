#!/bin/bash

# =============================================================================
# MODEL CONSTRUCTOR JOB SCRIPT
# =============================================================================
# This script is intended to be run on a SLURM cluster with GPU resources.
# It loads the required Conda environment and runs the model_constructor.py script.
# You can either train the model or run Optuna hyperparameter optimization.
# =============================================================================

#SBATCH --job-name=run_model
#SBATCH --output=model_output_%j.out
#SBATCH --error=model_error_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate tp53

# Train the model using the specified configuration
python src/model_constructor.py --mode train --config configs/simple_run_final_configuration.json

# Run Optuna hyperparameter optimization, uncomment the line below:
# python src/model_constructor.py --mode optuna --config configs/optuna_balanced_gat.json