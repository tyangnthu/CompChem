#!/bin/bash
#SBATCH --job-name=traing_model    ## job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1            ##number of GPU
#SBATCH --cpus-per-task=4
##SBATCH --time=00:30:00    
#SBATCH --account=MST111483
#SBATCH --partition=gp4d
source ~/.bashrc
. ~/anaconda3/etc/profile.d/conda.sh
conda activate CompChem

python ~/CompChem/Practice/Week12/Exercise2/train.py
