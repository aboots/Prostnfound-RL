#!/bin/bash

#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1
#SBATCH --time 8:00:00
#SBATCH -c 16 
#SBATCH --qos=m2
#SBATCH --output=slurm-%j.log
#SBATCH --open-mode=append

# send this batch script a SIGUSR1 240 seconds
# before we hit our time limit
#SBATCH --signal=B:USR1@240

CENTER=UVA
EXP_NAME=${CENTER}_age_psa_prompts
EXP_DIR=experiments/${EXP_NAME}/$SLURM_JOB_ID
CKPT_DIR=/checkpoint/$USER/$SLURM_JOB_ID

# Set environment variables for training
export TQDM_MININTERVAL=30
export WANDB_RUN_ID=$SLURM_JOB_ID
export WANDB_RESUME=allow

# Create experiment directory
echo "EXP_DIR: $EXP_DIR"
mkdir -p $EXP_DIR
# Symbolic link to checkpoint directory
# so it is easier to find them
echo "CKPT_DIR: $CKPT_DIR"
ln -s $CKPT_DIR $(realpath $EXP_DIR)/checkpoints

# Kill training process and resubmit job if it receives a SIGUSR1
handle_timeout_or_preemption() {
  date +"%Y-%m-%d %T"
  echo "Caught timeout or preemption signal"
  echo "Sending SIGINT to child process"
  scancel $SLURM_JOB_ID --signal=SIGINT
  wait $child_pid
  echo "Job step terminated gracefully"
  echo $(date +"%Y-%m-%d %T") "Resubmitting job"
  scontrol requeue $SLURM_JOB_ID
  exit 0
}
trap handle_timeout_or_preemption SIGUSR1

# Run training script
srun python train_medsam.py \
  --test_center $CENTER \
  --undersample_benign_ratio 3 \
  --min_involvement_train 40 \
  --augmentations translate \
  --remove_benign_cores_from_positive_patients \
  --lr 0.00001 \
  --encoder_lr 0.00001 \
  --warmup_lr 0.0001 \
  --warmup_epochs 1 \
  --wd 0 \
  --prompts age psa \
  --epochs 30 \
  --test_every_epoch \
  --base_loss ce \
  --device cuda \
  --accumulate_grad_steps 8 \
  --exp_dir $EXP_DIR \
  --checkpoint_dir $CKPT_DIR \
  --project miccai2024 \
  --name $EXP_NAME \
  --log_images \
  --seed 42 & wait

