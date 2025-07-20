#!/bin/bash

# =============================================================================
# ADD MUTATION STATUS SCRIPT
# =============================================================================
# 
# USAGE: Comment/uncomment the section for the dataset you want to process
# 
# To run: sbatch jobs/add_mutation_status.sh
# To run locally: bash jobs/add_mutation_status.sh
# =============================================================================

# SLURM configuration
#SBATCH --job-name=add_mut_status
#SBATCH --output=add_mut_status_%j.out 
#SBATCH --error=add_mut_status_%j.err 
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=defq

# Create logs directory if it doesn't exist
mkdir -p logs

# Redirect output and error to files with a pseudo-job ID if not running under SLURM
if [ -z "$SLURM_JOB_ID" ]; then
  JOB_ID=$(date +%s)
  exec > add_mutation_status_${JOB_ID}.out 2> add_mutation_status_${JOB_ID}.err
fi

# =============================================================================
# GAMBARDELLA DATASET
# =============================================================================
# Uncomment the section below to process Gambardella dataset
# Comment out other dataset sections

# DATASET="gambardella"
# EXPRESSION_H5AD="output/expression_matrix_gambardella.h5ad"
# MUTATION_FILES=(
#     "data/ccle_broad_2019/data_mutations.txt"
#     "data/cellline_ccle_broad/data_mutations.txt"
# )
# OUTPUT_H5AD="output/expression_matrix_gambardella_with_tp53_status.h5ad"
# CELL_LINE_COLUMN="Cell_line"

# =============================================================================
# KINKER DATASET  
# =============================================================================
# Uncomment the section below to process Kinker dataset
# Comment out other dataset sections

DATASET="kinker"
EXPRESSION_H5AD="output/expression_matrix_kinker.h5ad"
MUTATION_FILES=(
    "data/ccle_broad_2019/data_mutations.txt"
    "data/cellline_ccle_broad/data_mutations.txt"
)
OUTPUT_H5AD="output/expression_matrix_kinker_with_tp53_status.h5ad"
CELL_LINE_COLUMN="Cell_line"

# =============================================================================
# VALIDATION AND EXECUTION
# =============================================================================

# Check if a dataset is selected
if [[ -z "$DATASET" ]]; then
    echo "ERROR: No dataset selected!"
    echo "Please uncomment one of the dataset sections above."
    echo ""
    echo "Available datasets:"
    echo "  - Gambardella: Uncomment the GAMBARDELLA DATASET section"
    echo "  - Kinker: Uncomment the KINKER DATASET section"
    exit 1
fi

# Check if all required variables are set
if [[ -z "$EXPRESSION_H5AD" ]]; then
    echo "ERROR: EXPRESSION_H5AD not set for dataset: $DATASET"
    exit 1
fi

if [[ ${#MUTATION_FILES[@]} -eq 0 ]]; then
    echo "ERROR: MUTATION_FILES not set for dataset: $DATASET"
    exit 1
fi

if [[ -z "$OUTPUT_H5AD" ]]; then
    echo "ERROR: OUTPUT_H5AD not set for dataset: $DATASET"
    exit 1
fi

if [[ -z "$CELL_LINE_COLUMN" ]]; then
    echo "ERROR: CELL_LINE_COLUMN not set for dataset: $DATASET"
    exit 1
fi

# Build the command
CMD="python src/add_mutation_status.py --dataset '$DATASET' --expression-h5ad '$EXPRESSION_H5AD' --output-h5ad '$OUTPUT_H5AD' --cell-line-column '$CELL_LINE_COLUMN' --mutation-files"
for file in "${MUTATION_FILES[@]}"; do
    CMD="$CMD '$file'"
done

# Display configuration
echo "============================================================================="
echo "ADD MUTATION STATUS - DATASET: $DATASET"
echo "============================================================================="
echo "Expression AnnData: $EXPRESSION_H5AD"
echo "Mutation files: ${MUTATION_FILES[*]}"
echo "Output AnnData: $OUTPUT_H5AD"
echo "Cell line column: $CELL_LINE_COLUMN"
echo "============================================================================="
echo "Command: $CMD"
echo "Starting at: $(date)"
echo "============================================================================="

# Execute the command
eval $CMD

echo "============================================================================="
echo "Done at: $(date)"
echo "============================================================================="