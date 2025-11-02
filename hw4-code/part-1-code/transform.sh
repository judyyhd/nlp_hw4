#!/bin/bash
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00

python3 main.py --eval_transformed
