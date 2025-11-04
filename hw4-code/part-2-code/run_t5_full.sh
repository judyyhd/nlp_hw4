#!/bin/bash
#SBATCH --job-name=t5_finetune
#SBATCH --output=t5_train_%j.log
#SBATCH --error=t5_train_%j.err
#SBATCH --partition=a100_2
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hy1331@nyu.edu


python3 train_t5.py \
    --finetune \
    --auto_resume \
    --batch_size 16 \
    --test_batch_size 16 \
    --learning_rate 1e-4 \
    --max_n_epochs 15 \
    --patience_epochs 5 \
    --experiment_name ft_experiment