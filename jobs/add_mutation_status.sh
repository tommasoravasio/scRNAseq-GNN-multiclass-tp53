#!/bin/bash
#SBATCH --job-name=add_mut_status
#SBATCH --output=add_mut_status.out
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=defq

cd /home/3192281/tp53_mul

echo "Running add_mutation_status.py..."
python src/add_mutation_status.py \
    --expression_matrix output/final_processed_data.csv \
    --mutation_data data/ccle_broad_2019_clinical_data.tsv \
    --output_file output/final_labeled_matrix.csv
echo "Done!"