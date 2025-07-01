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


# 1. Define file paths
#    Adjust these paths if your script/data are located elsewhere relative to where you submit.
#    Assumes data is in a 'data' subdirectory from where the job is submitted.
BASE_DIR=$(pwd) # Assumes you submit from the directory containing 'process_data_cluster.py' and 'data/'
EXPRESSION_FILE="${BASE_DIR}/data/GSE157220/UMIcount_data.txt"
METADATA_FILE="${BASE_DIR}/data/GSE157220/Metadata.txt"
OUTPUT_DIR="${BASE_DIR}/output" # Directory to save the processed data
OUTPUT_FILE="${OUTPUT_DIR}/expression_matrix.csv" # Or .parquet
CHUNKSIZE_GENES=1000 # Default chunksize, can be overridden here or as script default

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# 2. Run the Python script
echo "Starting Python script: generate_expression_matrix.py"
python src/generate_expression_matrix.py \
    --expression_file "${EXPRESSION_FILE}" \
    --metadata_file "${METADATA_FILE}" \
    --output_file "${OUTPUT_FILE}" \
    --chunksize_genes ${CHUNKSIZE_GENES}

# Check Python script exit status
if [ $? -eq 0 ]; then
    echo "Python script completed successfully."
else
    echo "Python script failed. Check error logs: process_scRNA_%j.err and process_scRNA_%j.out"
    exit 1
fi

echo "Job finished."
