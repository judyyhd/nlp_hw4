#!/bin/bash
#SBATCH --job-name=t5_scratch
#SBATCH --output=t5_scratch_%j.log
#SBATCH --error=t5_scratch_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hy1331@nyu.edu


python3 train_t5.py \
    --auto_resume \
    --batch_size 16 \
    --test_batch_size 16 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --scheduler_type cosine \
    --num_warmup_epochs 5 \
    --max_n_epochs 80 \
    --patience_epochs 15 \
    --experiment_name scratch_experiment_v2