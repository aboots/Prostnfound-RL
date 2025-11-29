#!/bin/bash
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time 8:00:00
#SBATCH -c 16 
#SBATCH --output=slurm-%A-%a.log
#SBATCH --open-mode=append
#SBATCH --partition=rtx6000
#SBATCH --qos=m
#SBATCH --array=0-4

WANDB_RUN_GROUP=kfold \
    srun python train_patch_classification.py \
    --fold $SLURM_ARRAY_TASK_ID \
    --exclude_benign \
    --cohort_selection_mode kfold \
    --involvement_threshold_pct 40 \
    --undersample_benign_ratio 6 \
    --batch_size 128 --use_aug