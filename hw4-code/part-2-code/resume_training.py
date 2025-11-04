#!/usr/bin/env python3
"""
Helper script to resume training from checkpoints
"""
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Resume T5 training from checkpoint')
    parser.add_argument('--experiment_name', type=str, default='ft_experiment',
                        help='Name of the experiment to resume')
    parser.add_argument('--model_type', type=str, default='ft', choices=['ft', 'scr'],
                        help='Model type (ft for finetune, scr for scratch)')
    parser.add_argument('--list_checkpoints', action='store_true',
                        help='List available checkpoints')
    
    args = parser.parse_args()
    
    if args.list_checkpoints:
        checkpoint_base = 'checkpoints'
        if os.path.exists(checkpoint_base):
            print("Available checkpoints:")
            for model_type in ['ft', 'scr']:
                exp_dir = os.path.join(checkpoint_base, f'{model_type}_experiments')
                if os.path.exists(exp_dir):
                    print(f"\n{model_type.upper()} experiments:")
                    for exp in os.listdir(exp_dir):
                        exp_path = os.path.join(exp_dir, exp)
                        if os.path.isdir(exp_path):
                            has_state = os.path.exists(os.path.join(exp_path, 'training_state.pt'))
                            has_best = os.path.exists(os.path.join(exp_path, 'best_model'))
                            has_last = os.path.exists(os.path.join(exp_path, 'last_model'))
                            status = []
                            if has_state:
                                status.append("training_state")
                            if has_best:
                                status.append("best_model")
                            if has_last:
                                status.append("last_model")
                            print(f"  - {exp} ({', '.join(status)})")
        return
    
    # Find checkpoint path
    checkpoint_path = os.path.join('checkpoints', f'{args.model_type}_experiments', args.experiment_name)
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Use --list_checkpoints to see available checkpoints")
        return
    
    if not os.path.exists(os.path.join(checkpoint_path, 'training_state.pt')):
        print(f"No training state found in {checkpoint_path}")
        print("Cannot resume training without training_state.pt")
        return
    
    # Generate command to resume training
    finetune_flag = '--finetune' if args.model_type == 'ft' else ''
    
    command = f"""python3 train_t5.py \\
    {finetune_flag} \\
    --resume_from_checkpoint {checkpoint_path} \\
    --experiment_name {args.experiment_name} \\
    --learning_rate 1e-4 \\
    --max_n_epochs 15 \\
    --patience_epochs 5 \\
    --batch_size 16 \\
    --test_batch_size 16"""
    
    print("Command to resume training:")
    print(command)
    print("\nOr use auto-resume:")
    print(f"python3 train_t5.py {finetune_flag} --auto_resume --experiment_name {args.experiment_name} --learning_rate 1e-4 --max_n_epochs 15 --patience_epochs 5")

if __name__ == "__main__":
    main()