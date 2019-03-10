#!/bin/bash
#SBATCH --ntasks-per-node=1 # number of tasks per node
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-05:00 # Runtime in D-HH:MM
#SBATCH -p gpu # gpu partition
#SBATCH --gres=gpu:1 # number of gpus


module load cuda/8.0
module load anaconda3/5.0.1
python3 RLO1.py   
