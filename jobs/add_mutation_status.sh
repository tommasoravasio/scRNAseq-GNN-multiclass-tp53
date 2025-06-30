#!/bin/bash
#SBATCH --job-name=add_mut_status
#SBATCH --output=add_mut_status_%j.out 
#SBATCH --error=add_mut_status_%j.err 
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=defq

cd /home/3192281/tp53_mul

echo "Running add_mutation_status.py..."
python src/add_mutation_status.py 
echo "Done!"