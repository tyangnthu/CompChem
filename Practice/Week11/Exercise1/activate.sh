#!/bin/sh

# setting environments
module load cuda/11.7
module load openmpi4/4.1.4
module load intel/2020
export PATH=$HOME/software/gromacs/2022.2/bin:$PATH
#module load gcc/11.2.0
#alias gmx=/home/nickyang69/software/gromacs/2022.2/bin/gmx_mpi
