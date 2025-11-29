#!/bin/bash
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time 8:00:00
#SBATCH -c 16 
#SBATCH --output=slurm-%j.log
#SBATCH --open-mode=append
#SBATCH --partition=rtx6000
#SBATCH --qos=m

srun $@