#! /bin/bash
# This script is used to submit a training job for the DINOv3 model.
#SBATCH --nodes=1
#SBATCH --job-name=dinov3-ultrasound
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --account=aip-medilab
#SBATCH --time=1:00:00 
#SBATCH --output=logs/%x-%j.log
#SBATCH --mem=64G # request all available memory

source ~/.venv/bin/activate

srun python train.py -c $1