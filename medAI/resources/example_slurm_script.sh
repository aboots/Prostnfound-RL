#!/bin/bash
#SBATCH --job-name=ibot
#SBATCH --output=%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --mem=128G
#SBATCH --qos=m2
#SBATCH --partition=a40,rtx6000,t4v2
#SBATCH --time=4:00:00
#SBATCH --signal=B:USR1@240
#SBATCH --open-mode=append

# send this batch script a SIGUSR1 240 seconds
# before we hit our time limit
#SBATCH --signal=B:USR1@240


MY_EXP_DIR_NAME=logs/$(date +"%Y-%m-%d/%H-%M-%S")


# load prior checkpoint if it exists for this job
CHECKPOINT_DIR=/checkpoint/$USER/$SLURM_JOB_ID
if [ -f "${CHECKPOINT_DIR}/.slurm_setup.sh" ]
then
    source ${CHECKPOINT_DIR}/.slurm_setup.sh
else
    echo "export EXPERIMENT_DIR=$MY_EXP_DIR_NAME" > ${CHECKPOINT_DIR}/.slurm_setup.sh
    source ${CHECKPOINT_DIR}/.slurm_setup.sh
fi
echo "Experiment directory: $EXPERIMENT_DIR"


# symbolic link to checkpoint directory
mkdir -p $EXPERIMENT_DIR
if [ $SLURM_LOCALID = 0 ]
then
    if [ ! -d "${EXPERIMENT_DIR}/checkpoint" ] 
    then
        ln -s $CHECKPOINT_DIR $EXPERIMENT_DIR/checkpoint
    fi
fi


# Kill training process and resubmit job if it receives a SIGUSR1
handle_timeout_or_preemption() {
  
  echo $(date +"%Y-%m-%d %T") "Caught timeout or preemption signal"
  echo "Sending SIGINT to child process"
  scancel $SLURM_JOB_ID --signal=SIGINT
  wait $child_pid
  echo "Job step terminated gracefully"
  echo $(date +"%Y-%m-%d %T") "Resubmitting job"
  scontrol requeue $SLURM_JOB_ID
  exit 0
}
trap handle_timeout_or_preemption SIGUSR1

# Set environment variables for training - these are useful examples
export TQDM_MININTERVAL=30
export WANDB_RUN_ID=$SLURM_JOB_ID
export WANDB_RESUME=allow

# =====================================================================
# Run training script - this is where you would put your training script
echo "Running training script"

COMMAND="python -c 'import time; time.sleep(1000)'"
srun -u -o $EXPERIMENT_DIR/%j-%t.log $COMMAND & 

# =====================================================================

child_pid=$!
wait $child_pid

echo "Job step terminated successfully."
