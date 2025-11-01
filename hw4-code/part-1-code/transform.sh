#!/bin/bash
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00

conda init bash
source ~/.bashrc
conda activate hw4-part-1-nlp
python3 main.py --eval_transformed --debug_transformation