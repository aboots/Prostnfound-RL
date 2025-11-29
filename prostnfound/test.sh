#!/bin/bash
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time 8:00:00
#SBATCH -c 16 
#SBATCH --partition rtx6000
#SBATCH --qos m2
#SBATCH --output=slurm-%j.log
#SBATCH --open-mode=append
##SBATCH --account=deadline
##SBATCH --qos=deadline

# send this batch script a SIGUSR1 240 seconds
# before we hit our time limit
#SBATCH --signal=B:USR1@240

if [ -z $1 ]; 
then return 1; 
fi 

checkpoint_dir=$1
split=val
dataset=nct2013

id=$(basename $checkpoint_dir)
output_dir=$(pwd)/logs/$split/$id
mkdir -p $output_dir

srun -o $output_dir/out.log -u python test.py \
    --checkpoint "${checkpoint_dir}/experiment_state.pth" \
    --output_dir ${output_dir} \
    --split $split \
    --save_heatmaps \

