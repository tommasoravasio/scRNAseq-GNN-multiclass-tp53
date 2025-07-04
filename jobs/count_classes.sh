#!/bin/bash
#SBATCH --job-name=count_classes
#SBATCH --output=count_classes.out
#SBATCH --error=count_classes.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1


cd /home/3192281/tp53_mul
echo "Counting different values per class..."
python src/count_classes.py
echo "Job finished."