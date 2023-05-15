#!/bin/bash

# check sys argument
if [ $# -lt 1 ];
then
  echo 'Make sure that you specify the gro file as the first argument.'
  exit
else
  grof=$1
fi

if [ $# -lt 2 ];
then
  echo 'Make sure that you specify the number of RE as the second argument.'
  exit
else
  num_re=$2
fi

if [ $# -lt 3 ];
then
  echo 'Make sure that you specify the lower Temp as the third argument.'
  exit
else
  temp_i=$3
fi

if [ $# -lt 4 ];
then
  echo 'Make sure that you specify the higher Temp as the fourth argument.'
  exit
else
  temp_f=$4
fi

if [ $# -eq 5 ];
then
  if [[ $5 == 'random' ]] || [[ $5 == 'equal' ]];
  then
    option=$5
    echo "Will do $option generation of temperatures"
  else
    echo "The fifth argument was specified not as one of the options: random/equal. Please re-enter with one of the two options"
    exit
  fi
fi

# check for files
if [ $(find ./ -name "md*" | wc -l ) -eq 0 ];
then
  echo 'md files are not found. Exiting.'
  exit 1
fi
if [ $(find ./ -name "posre*" | wc -l ) -eq 0 ];
then
  echo 'posre files are not found. Exiting.'
  exit 1
fi
if [ $(find ./ -name "topol*" | wc -l ) -eq 0 ];
then
  echo 'topol files are not found. Exiting.'
  exit 1
fi
if [ $(find ./ -name "lig*" | wc -l ) -eq 0 ];
then
  echo 'lig files are not found. Exiting.'
  exit 1
fi

if [ $(find ./ -name "index*" | wc -l ) -eq 0 ];
then
  echo 'index files are not found. Exiting.'
  exit 1
fi

# modify remd*mdp files
temp_d=$(echo "$temp_f - $temp_i" | bc -l)
if [[ $option == 'random' ]];
then
  temps=($(for i in $(seq 1 1 $num_re); do echo $(($RANDOM * $temp_d  / 32767 + $temp_i)); done | sort -n))
else
  temps=($(seq $temp_i $(($temp_d / ($num_re - 1))) $temp_f))
fi
for (( i=0; i<$num_re; i++ ));
do
  cp remd.mdp remd_${i}.mdp
  cp nvt.mdp nvt_${i}.mdp
  cp npt.mdp npt_${i}.mdp
  temp=${temps[$i]}
  sed -i "s/ temp/ $temp/g" nvt_${i}.mdp
  sed -i "s/ temp/ $temp/g" npt_${i}.mdp
  sed -i "s/ temp/ $temp/g" remd_${i}.mdp
done

# define the number of jobs to be submitted
gmx_dir=/home/nickyang69/software/gromacs/2022.2/bin
gmx_mpi_dir=/home/nickyang69/software/gromacs/2022.2-mpi/bin

# generate submit files and submit them
cat > remd.sub << EOF 
#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4 
#SBATCH --ntasks-per-node=$num_re
#SBATCH --gres=gpu:$num_re
#SBATCH --account=MST111483
#SBATCH -p gp1d
#SBATCH -t 24:00:00

export GMX_ENABLE_DIRECT_GPU_COMM=true

. ~/scripts/gromacs/activate.sh

# Set some environment variables 
REPLICA_EXCHANGE=\$(pwd)
echo "Replica exchange home directory set to \$REPLICA_EXCHANGE"
MDP=\$REPLICA_EXCHANGE
echo ".mdp files are stored in \$MDP"
workspace=\$(pwd)

# Change to the location of your GROMACS-2018 installation
GMX=$gmx_dir
GMX_MPI=$gmx_mpi_dir

nvt () {
  echo "Starting constant volume equilibration for RE\$1..."

  \$GMX/gmx grompp -n \$REPLICA_EXCHANGE/index.ndx -r \$REPLICA_EXCHANGE/$grof \
    -c \$REPLICA_EXCHANGE/$grof -f \$MDP/nvt_\$1.mdp -p \$REPLICA_EXCHANGE/topol.top -o nvt\$1.tpr

  \$GMX/gmx mdrun -deffnm nvt\$1 -ntomp 4 -ntmpi 1 -update gpu

  echo "Constant volume equilibration complete."

  sleep 10
}

npt () {
  echo "Starting constant pressure equilibration for RE\$1..."

  \$GMX/gmx grompp -n \$REPLICA_EXCHANGE/index.ndx -r nvt\$1.gro -f \$MDP/npt_\$1.mdp \
    -c nvt\$1.gro -p \$REPLICA_EXCHANGE/topol.top -t nvt\$1.cpt -o npt\$1.tpr

  \$GMX/gmx mdrun -deffnm npt\$1 -ntomp 4 -ntmpi 1 -update gpu

  echo "Constant pressure equilibration complete."

  sleep 10
}

remd () {
  \$GMX/gmx grompp -n \$REPLICA_EXCHANGE/index.ndx -f \$REPLICA_EXCHANGE/remd_\$1.mdp -c npt\$1.gro \
      -r npt\$1.gro -p \$REPLICA_EXCHANGE/topol.top -t npt\$1.cpt -o remd.tpr
    echo "REMD tpr generation complete."

  sleep 10
}

for (( i=0; i<$num_re; i++ ))
do
  (
    RE=\$i
    mkdir \$workspace/RE\$RE
    cd \$workspace/RE\$RE
    nvt \$RE
    npt \$RE
    remd \$RE
  ) &
done

wait

cd \$workspace

. ~/scripts/gromacs/activate_mpi.sh
mpirun -np $num_re \$GMX_MPI/gmx_mpi mdrun -s remd -multidir \$(ls -d RE*/ | sort -n | xargs echo) \
  -ntomp 4 -update gpu -replex 1000 -nstlist 200

EOF

sbatch remd.sub
