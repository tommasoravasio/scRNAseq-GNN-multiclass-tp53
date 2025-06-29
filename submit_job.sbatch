#!/bin/bash

#SBATCH --job-name=process_scRNA
#SBATCH --output=process_scRNA_%j.out  # Standard output and error log (%j expands to jobID)
#SBATCH --error=process_scRNA_%j.err   # Separate error log (optional, can be merged with output)

# --- Resource Requests ---
#SBATCH --partition=defq             # Queue/Partition name (please verify 'defq' is a CPU queue or specify another)
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks=1                   # Number of tasks (usually 1 for non-MPI jobs)
#SBATCH --cpus-per-task=4            # Number of CPUs per task (pandas can use multiple cores)
#SBATCH --mem=32G                    # Memory requested (e.g., 32GB; adjust as needed)
#SBATCH --time=02:00:00              # Wall time limit (e.g., 2 hours; adjust as needed)

# --- Job Steps ---

# 1. Load Conda module (if required by your cluster environment)
#    Uncomment and modify the line below if your cluster uses environment modules for Conda
# module load anaconda/latest  # Or specific version e.g., anaconda3/2023.09
# module load miniconda3/latest

# 2. Activate your Conda environment
#    Replace 'your_conda_env_name' with the actual name of your Conda environment
CONDA_ENV_NAME="your_conda_env_name"
source $(conda info --base)/etc/profile.d/conda.sh # Ensures conda command is found
conda activate ${CONDA_ENV_NAME}

# Check if Conda activation was successful
if [ $? -ne 0 ]; then
    echo "Conda environment '${CONDA_ENV_NAME}' activation failed. Exiting."
    exit 1
fi
echo "Conda environment activated: $(which python)"

# 3. Define file paths
#    Adjust these paths if your script/data are located elsewhere relative to where you submit.
#    Assumes data is in a 'data' subdirectory from where the job is submitted.
BASE_DIR=$(pwd) # Assumes you submit from the directory containing 'process_data_cluster.py' and 'data/'
EXPRESSION_FILE="${BASE_DIR}/data/GSE157220/UMIcount_data.txt"
METADATA_FILE="${BASE_DIR}/data/GSE157220/Metadata.txt"
OUTPUT_DIR="${BASE_DIR}/output" # Directory to save the processed data
OUTPUT_FILE="${OUTPUT_DIR}/final_processed_data.csv" # Or .parquet
CHUNKSIZE_GENES=1000 # Default chunksize, can be overridden here or as script default

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# 4. Run the Python script
echo "Starting Python script: process_data_cluster.py"
python process_data_cluster.py \
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

# 5. Deactivate Conda environment (optional, good practice)
conda deactivate

echo "Job finished."
