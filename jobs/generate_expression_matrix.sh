#!/bin/bash



#SBATCH --job-name=generate_expression_matrix
#SBATCH --output=generate_expression_matrix_%j.out  # Standard output and error log (%j expands to jobID)
#SBATCH --error=generate_expression_matrix_%j.err   # Separate error log (optional, can be merged with output)
#SBATCH --partition=defq             # Queue/Partition name 
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks (usually 1 for non-MPI jobs)
#SBATCH --cpus-per-task=4            # Number of CPUs per task (pandas can use multiple cores)
#SBATCH --mem=128G                   # Memory requested 
#SBATCH --time=02:00:00              # Wall time limit 

source /software/miniconda3/etc/profile.d/conda.sh
conda activate tp53
echo "Activated Conda environment: $CONDA_DEFAULT_ENV"
######################

# # VERSION FOR KINKER DATA
# # Define file paths
# BASE_DIR=$(pwd) # Assumes you submit from the directory containing 'data/'
# EXPRESSION_FILE="${BASE_DIR}/data/Kinker/UMIcount_data.txt"
# METADATA_FILE="${BASE_DIR}/data/Kinker/Metadata.txt"
# OUTPUT_DIR="${BASE_DIR}/output" # Directory to save the processed data
# OUTPUT_FILE="${OUTPUT_DIR}/expression_matrix_kinker.csv" # Or .parquet
# CHUNKSIZE_GENES=1000 # Default chunksize, can be overridden here or as script default

# mkdir -p ${OUTPUT_DIR}

# # Run the Python script
# echo "Starting Python script: generate_expression_matrix.py"
# python src/generate_expression_matrix.py \
#     --data kinker \
#     --expression_file "${EXPRESSION_FILE}" \
#     --metadata_file "${METADATA_FILE}" \
#     --output_file "${OUTPUT_FILE}" \
#     --chunksize_genes ${CHUNKSIZE_GENES}
# if [ $? -eq 0 ]; then
#     echo "Python script completed successfully."
# else
#     echo "Python script failed. Check error logs: process_scRNA_%j.err and process_scRNA_%j.out"
#     exit 1
# fi

# echo "Job finished."

#######################

# VERSION FOR GAMBARDELLA DATA
BASE_DIR=$(pwd)
INPUT_DIRECTORY="${BASE_DIR}/data/Gambardella"
OUTPUT_DIR="${BASE_DIR}/output"
OUTPUT_FILE="${OUTPUT_DIR}/expression_matrix_gambardella.csv" 

mkdir -p "${OUTPUT_DIR}"

python src/generate_expression_matrix.py \
    --data gambardella \
    --input_dir "${INPUT_DIRECTORY}" \
    --output_dir "${OUTPUT_DIR}" \
    --output_file "${OUTPUT_FILE}" 


if [ $? -eq 0 ]; then
    echo "Python script completed successfully."
else
    echo "Python script failed. Check error logs: process_scRNA_%j.err and process_scRNA_%j.out"
    exit 1
fi

echo "Job finished."
