#!/bin/bash -l

#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -J test
#SBATCH -A m3623_g
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -c 32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1

srun ./agent inputs.ca &> run_out.txt
