#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4 
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --account=MST111483
#SBATCH -p gp1d

. ~/scripts/gromacs/activate.sh

# Set some environment variables 
FREE_ENERGY=$(pwd)
echo "Free energy home directory set to $FREE_ENERGY"
MDP=$FREE_ENERGY
echo ".mdp files are stored in $MDP"

# Change to the location of your GROMACS-2018 installation
GMX=$HOME/software/gromacs/2022.2/bin

###  NVT  ###
$GMX/gmx grompp -f nvt.mdp -c em.gro -p topol.top -r em.gro -o nvt.tpr 

$GMX/gmx mdrun -deffnm nvt

###  NPT  ###
$GMX/gmx grompp -f npt.mdp -c nvt.gro -p topol.top -r nvt.gro -o npt.tpr

$GMX/gmx mdrun -deffnm npt

###  MD  ###

$GMX/gmx grompp -f md_pull.mdp -c npt.gro -p topol.top -r npt.gro -n index.ndx -t npt.cpt -o pull.tpr

$GMX/gmx mdrun -deffnm pull -pf pullf.xvsg -px pullx.xvg
