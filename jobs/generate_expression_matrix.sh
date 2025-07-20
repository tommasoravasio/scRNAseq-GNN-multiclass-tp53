#!/bin/bash

############################################################
# generate_expression_matrix.sh
#
# Usage:
#   - Uncomment ONE of the dataset sections below (Kinker or Gambardella)
#   - Run this script to generate the expression matrix for the selected dataset
#   - Output and error logs will be saved with a unique job ID if not running under SLURM
#
# Sections:
#   1. Kinker dataset (blockwise, h5ad only)
#   2. Gambardella dataset (10x Genomics format)
#
# Only one section should be active at a time!
############################################################

# Redirect output and error to files with a pseudo-job ID if not running under SLURM
if [ -z "$SLURM_JOB_ID" ]; then
  JOB_ID=$(date +%s)
  exec > generate_expression_matrix_${JOB_ID}.out 2> generate_expression_matrix_${JOB_ID}.err
fi

# SLURM directives 
#SBATCH --job-name=generate_expression_matrix
#SBATCH --output=generate_expression_matrix_%j.out
#SBATCH --error=generate_expression_matrix_%j.err
#SBATCH --partition=defq
#SBATCH --partition=long-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=980G
#SBATCH --time=04:00:00

# Activate Conda environment
source /software/miniconda3/etc/profile.d/conda.sh
conda activate tp53
echo "Activated Conda environment: $CONDA_DEFAULT_ENV"

############################################################
# SECTION 1: Kinker Dataset (blockwise, h5ad only)
############################################################

# echo "Starting Python script: generate_expression_matrix.py (Kinker)"
# python src/generate_expression_matrix.py \
#     --data kinker \
#     --raw_file data/Kinker/UMIcount_data.txt \
#     --block_dir output/blocks \
#     --block_size 1000 \
#     --output_h5ad output/expression_matrix_kinker.h5ad

# if [ $? -eq 0 ]; then
#     echo "Python script completed successfully."
# else
#     echo "Python script failed. Check error logs: generate_expression_matrix_*.err and generate_expression_matrix_*.out"
#     exit 1
# fi

# echo "Job finished."

############################################################
# SECTION 2: Gambardella Dataset (10x Genomics format)
############################################################
# Uncomment this block to process the Gambardella dataset

BASE_DIR=$(pwd)
INPUT_DIRECTORY="${BASE_DIR}/data/Gambardella"
OUTPUT_DIR="${BASE_DIR}/output"
OUTPUT_H5AD="${OUTPUT_DIR}/expression_matrix_gambardella.h5ad"

mkdir -p "${OUTPUT_DIR}"

echo "Starting Python script: generate_expression_matrix.py (Gambardella)"
python src/generate_expression_matrix.py \
    --data gambardella \
    --input_dir "${INPUT_DIRECTORY}" \
    --output_h5ad "${OUTPUT_H5AD}"

if [ $? -eq 0 ]; then
    echo "Python script completed successfully."
else
    echo "Python script failed. Check error logs: generate_expression_matrix_*.err and generate_expression_matrix_*.out"
    exit 1
fi

echo "Job finished."
