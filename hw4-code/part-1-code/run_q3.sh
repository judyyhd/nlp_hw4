#!/bin/bash

#SBATCH --job-name=hw4_q3
#SBATCH #SBATCH --output=q3_output_%j.log
#SBATCH --error=q3_error_%j.log
#SBATCH --time=03:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
# Email notifications
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hy1331@nyu.edu

# Load your conda environment
source ~/.bashrc
conda activate hw4-part-1-nlp

# Navigate to code directory
cd ~/nlp_hw4/hw4-code/part-1-code

echo "=========================================="
echo "Starting Q3: Data Augmentation Experiments"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Step 1: Train model on augmented data and evaluate on transformed test set
echo ""
echo "Step 1: Training on augmented data..."
python3 main.py --train_augmented --eval_transformed

# Step 2: Evaluate augmented model on original test set
echo ""
echo "Step 2: Evaluating augmented model on original test set..."
python3 main.py --eval --model_dir out_augmented

# Step 3: Evaluate augmented model on transformed test set
echo ""
echo "Step 3: Evaluating augmented model on transformed test set..."
python3 main.py --eval_transformed --model_dir out_augmented

echo ""
echo "=========================================="
echo "Q3 Experiments Complete!"
echo "=========================================="
echo "Results files generated:"
echo "  - out_augmented_original.txt"
echo "  - out_augmented_transformed.txt"
