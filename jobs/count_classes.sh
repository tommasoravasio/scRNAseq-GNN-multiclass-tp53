#!/bin/bash

# =============================================================================
# COUNT CLASSES SCRIPT
# =============================================================================
# 
# USAGE: Comment/uncomment the section for the dataset you want to process
# 
# To run: sbatch jobs/count_classes.sh
# To run locally: bash jobs/count_classes.sh
# =============================================================================

# SLURM configuration
#SBATCH --job-name=count_classes
#SBATCH --output=logs/count_classes_%j.out
#SBATCH --error=logs/count_classes_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Create logs directory if it doesn't exist
mkdir -p logs

# =============================================================================
# GAMBARDELLA DATASET
# =============================================================================
# Uncomment the section below to process Gambardella dataset
# Comment out other dataset sections

DATASET="gambardella"
INPUT_FILE="output/expression_matrix_with_tp53_status_gambardella.csv"
CELL_LINE_COLUMN="Cell_line"

# =============================================================================
# KINKER DATASET  
# =============================================================================
# Uncomment the section below to process Kinker dataset
# Comment out other dataset sections

# DATASET="kinker"
# INPUT_FILE="output/expression_matrix_with_tp53_status_kinker.csv"
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

# Check if input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    echo "Please ensure the file exists before running this script."
    exit 1
fi

# Build the command
CMD="python src/count_classes.py --data '$DATASET' --input-file '$INPUT_FILE' --cell-line-column '$CELL_LINE_COLUMN'"

# Display configuration
echo "============================================================================="
echo "COUNT CLASSES - DATASET: $DATASET"
echo "============================================================================="
echo "Input file: $INPUT_FILE"
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