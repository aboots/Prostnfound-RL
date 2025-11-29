#!/bin/bash
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time 16:00:00
#SBATCH -c 16 
#SBATCH --output=slurm-%j.log
#SBATCH --open-mode=append
##SBATCH --account=deadline
##SBATCH --qos=deadline

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
loss=mil_prop_bce_entropy_reg # needle_region_ce, 

center=PCC

CHECKPOINT=$CKPT_DIR srun -u python train.py \
  --splits_file test_center_${center}.json \
  --augmentations translate,random_crop,gamma,contrast \
  --model prostnfound_adapter_medsam_legacy_ssl_adapt \
  --model_kw adapter_dim=64 \
  --model_kw prompts=[age,psa,approx_psa_density,mid_lateral_encoding,family_history] \
  --model_kw enc_ckpt=/h/pwilson/projects/medAI/checkpoints/ibot/test_center_${center}.pt \
  --image_size 224 \
  --mask_size 56 \
  --batch_size 8 \
  --loss ${loss} \
  --lr 1e-5 \
  --encoder_lr 1e-5 \
  --warmup_lr 1e-4 \
  --warmup_epochs 0 \
  --epochs 35 \
  --wd 0 \
  --device cuda \
  --use_amp \
  --project miccai2024_comparison \
  --seed 30 \
  --log_images \
  --log_images_every 20 \
  & 

child_pid=$!
wait $child_pid

