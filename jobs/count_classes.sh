#!/bin/bash
#SBATCH --job-name=count_tp53
#SBATCH --output=logs/count_tp53.out
#SBATCH --error=logs/count_tp53.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1


cd /home/3192281/tp53_mul
echo "Counting different values per class..."
python count_tp53.py
echo "Job finished."