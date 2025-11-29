#!/bin/bash
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time 8:00:00
#SBATCH -c 16 
#SBATCH --output=slurm-%j.log
#SBATCH --open-mode=append
#SBATCH --partition=rtx6000
#SBATCH --qos=m

# send this batch script a SIGUSR1 240 seconds
# before we hit our time limit
#SBATCH --signal=B:USR1@240

RUN_ID=$SLURM_JOB_ID
CKPT_DIR=/checkpoint/$USER/$RUN_ID

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

export TQDM_MININTERVAL=30
export WANDB_RUN_ID=$RUN_ID
export WANDB_RESUME=allow
loss=needle_region_ce
# loss=needle_region_ce


center=PCC
ssl_adapt="false"
export WANDB_TAGS="test-$CENTER"

CHECKPOINT=$CKPT_DIR srun -u python train.py \
  --splits_file test_center_${center}_undersample_benign.json \
  --augmentations none \
  --model unet_small \
  --image_size 224 \
  --mask_size 224 \
  --batch_size 64 \
  --no-run_val \
  --loss ${loss} \
  --lr 1e-4 \
  --encoder_lr 1e-4 \
  --warmup_lr 3e-4 \
  --warmup_epochs 0 \
  --epochs 35 \
  --wd 0 \
  --device cuda \
  --use_amp \
  --project miccai2024_comparison \
  --seed 30 \
  $@ \
  & 

child_pid=$!
wait $child_pid

