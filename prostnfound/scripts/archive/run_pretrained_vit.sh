#!/bin/bash
#SBATCH --mem=16G
#SBATCH --gres=gpu:a40:1
#SBATCH --time 16:00:00
#SBATCH -c 16 
#SBATCH --output=slurm-%j.log
#SBATCH --open-mode=append
#SBATCH --account=deadline
#SBATCH --qos=deadline

# send this batch script a SIGUSR1 240 seconds
# before we hit our time limit
#SBATCH --signal=B:USR1@240

CENTER=PCC
MODEL_SETUP=unetr_vit_lora
EXP_NAME=${CENTER}_ProFound_${MODEL_SETUP}
RUN_ID=$SLURM_JOB_ID
EXP_DIR=experiments/${EXP_NAME}/$RUN_ID
#CKPT_DIR=/checkpoint/$USER/$RUN_ID
CKPT_DIR=${SCRATCH_DIR}/SLURM/${SLURM_JOB_ID}
mkdir -p $CKPT_DIR

# Set environment variables for training
export TQDM_MININTERVAL=30
export WANDB_RUN_ID=$RUN_ID
export WANDB_RESUME=allow
export PYTHONUNBUFFERED=1

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


# model and backbone configs

if [ "$MODEL_SETUP" == "unetr_vit" ]; then
    MODEL=unetr_vit
    BACKBONE="vit_small.medibot-v0"
    IMAGE_SIZE=224
    MASK_SIZE=224
    WARMUP_EPOCHS=100
elif [ "$MODEL_SETUP" == "unetr_vit_lora" ]; then
    MODEL=unetr_vit
    BACKBONE="vit_small.medibot-v0:lora_r=2"
    IMAGE_SIZE=224
    MASK_SIZE=224
    WARMUP_EPOCHS=15
else
    echo "Unknown model: $MODEL"
    exit 1
fi


# Run training script
srun -u python train_profound.py \
  --test_center $CENTER \
  --min_involvement_train 40 \
  --augmentations translate \
  --batch_size 16 \
  --lr 1e-5 \
  --encoder_lr 1e-5 \
  --warmup_lr 1e-4 \
  --warmup_epochs $WARMUP_EPOCHS \
  --model $MODEL \
  --backbone $BACKBONE \
  --image_size $IMAGE_SIZE \
  --mask_size $MASK_SIZE \
  --epochs 100 \
  --wd 0 \
  --test_every_epoch \
  --loss_0_name valid_region \
  --loss_0_base_loss_name ce \
  --loss_0_pos_weight 2 \
  --loss_0_prostate_mask True \
  --loss_0_needle_mask True \
  --loss_0_weight 1 \
  --device cuda \
  --accumulate_grad_steps 2 \
  --exp_dir $EXP_DIR \
  --checkpoint_dir $CKPT_DIR \
  --name $EXP_NAME \
  --use_amp \
  --project miccai2024_comparisons \
  --seed 42 \
  $@ \
  & 

child_pid=$!
wait $child_pid

