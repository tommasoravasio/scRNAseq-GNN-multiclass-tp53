#!/bin/bash

# =============================================================================
# UNZIP UMI DATA SCRIPT
# =============================================================================
# 
# USAGE: This script unzips the UMI count data file for the Kinker dataset
# 
# To run: sbatch jobs/unzip_umi.sh
# To run locally: bash jobs/unzip_umi.sh
# =============================================================================

#SBATCH --job-name=unzip_umi
#SBATCH --output=unzip_umi.out
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --partition=defq

cd /home/3192281/tp53_mul/data/Kinker

echo "Unzipping UMIcount_data.txt.gz..."
gunzip UMIcount_data.txt.gz
echo "Job finished."
