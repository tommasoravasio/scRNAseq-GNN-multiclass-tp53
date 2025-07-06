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

# =============================================================================
# GAMBARDELLA DATASET
# =============================================================================
# Uncomment the section below to process Gambardella dataset
# Comment out other dataset sections

DATASET="gambardella"
EXPRESSION_MATRIX="output/expression_matrix_gambardella.csv"
MUTATION_FILES=(
    "data/ccle_broad_2019/data_mutations.txt"
    "data/cellline_ccle_broad/data_mutations.txt"
)
OUTPUT_FILE="output/expression_matrix_with_tp53_status_gambardella.csv"
CELL_LINE_COLUMN="Cell_line"

# =============================================================================
# KINKER DATASET  
# =============================================================================
# Uncomment the section below to process Kinker dataset
# Comment out other dataset sections

# DATASET="kinker"
# EXPRESSION_MATRIX="output/expression_matrix_kinker.csv"
# MUTATION_FILES=(
#     "data/ccle_broad_2019/data_mutations.txt"
#     "data/cellline_ccle_broad/data_mutations.txt"
# )
# OUTPUT_FILE="output/expression_matrix_with_tp53_status_kinker.csv"
# CELL_LINE_COLUMN="Cell_line"

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
if [[ -z "$EXPRESSION_MATRIX" ]]; then
    echo "ERROR: EXPRESSION_MATRIX not set for dataset: $DATASET"
    exit 1
fi

if [[ ${#MUTATION_FILES[@]} -eq 0 ]]; then
    echo "ERROR: MUTATION_FILES not set for dataset: $DATASET"
    exit 1
fi

if [[ -z "$OUTPUT_FILE" ]]; then
    echo "ERROR: OUTPUT_FILE not set for dataset: $DATASET"
    exit 1
fi

if [[ -z "$CELL_LINE_COLUMN" ]]; then
    echo "ERROR: CELL_LINE_COLUMN not set for dataset: $DATASET"
    exit 1
fi

# Build the command
CMD="python src/add_mutation_status.py --dataset '$DATASET' --expression-matrix '$EXPRESSION_MATRIX' --output-file '$OUTPUT_FILE' --cell-line-column '$CELL_LINE_COLUMN' --mutation-files"
for file in "${MUTATION_FILES[@]}"; do
    CMD="$CMD '$file'"
done

# Display configuration
echo "============================================================================="
echo "ADD MUTATION STATUS - DATASET: $DATASET"
echo "============================================================================="
echo "Expression matrix: $EXPRESSION_MATRIX"
echo "Mutation files: ${MUTATION_FILES[*]}"
echo "Output file: $OUTPUT_FILE"
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