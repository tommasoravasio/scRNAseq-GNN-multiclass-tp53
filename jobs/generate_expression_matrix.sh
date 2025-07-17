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
#   1. Kinker dataset (tabular format)
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
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00

# Activate Conda environment
source /software/miniconda3/etc/profile.d/conda.sh
conda activate tp53
echo "Activated Conda environment: $CONDA_DEFAULT_ENV"

############################################################
# SECTION 1: Kinker Dataset (tabular format)
############################################################
# Uncomment this block to process the Kinker dataset

BASE_DIR=$(pwd)
EXPRESSION_FILE="${BASE_DIR}/data/Kinker/UMIcount_data.txt.gz"  #FOR LOCAL USE
#EXPRESSION_FILE="${BASE_DIR}/data/Kinker/UMIcount_data.txt"  #FOR SLURM USE
METADATA_FILE="${BASE_DIR}/data/Kinker/Metadata.txt"
OUTPUT_DIR="${BASE_DIR}/output"
OUTPUT_FILE="${OUTPUT_DIR}/expression_matrix_kinker.csv"
CHUNKSIZE_GENES=1000

mkdir -p "${OUTPUT_DIR}"

echo "Starting Python script: generate_expression_matrix.py (Kinker)"
python src/generate_expression_matrix.py \
    --data kinker \
    --expression_file "${EXPRESSION_FILE}" \
    --metadata_file "${METADATA_FILE}" \
    --output_file "${OUTPUT_FILE}" \
    --chunksize_genes ${CHUNKSIZE_GENES}

if [ $? -eq 0 ]; then
    echo "Python script completed successfully."
else
    echo "Python script failed. Check error logs: generate_expression_matrix_*.err and generate_expression_matrix_*.out"
    exit 1
fi

echo "Job finished."

############################################################
# SECTION 2: Gambardella Dataset (10x Genomics format)
############################################################
# Uncomment this block to process the Gambardella dataset

# BASE_DIR=$(pwd)
# INPUT_DIRECTORY="${BASE_DIR}/data/Gambardella"
# OUTPUT_DIR="${BASE_DIR}/output"
# OUTPUT_FILE="${OUTPUT_DIR}/expression_matrix_gambardella.csv"

# mkdir -p "${OUTPUT_DIR}"

# echo "Starting Python script: generate_expression_matrix.py (Gambardella)"
# python src/generate_expression_matrix.py \
#     --data gambardella \
#     --input_dir "${INPUT_DIRECTORY}" \
#     --output_dir "${OUTPUT_DIR}" \
#     --output_file "${OUTPUT_FILE}"

# if [ $? -eq 0 ]; then
#     echo "Python script completed successfully."
# else
#     echo "Python script failed. Check error logs: generate_expression_matrix_*.err and generate_expression_matrix_*.out"
#     exit 1
# fi

# echo "Job finished."
