#!/bin/sh

module load orca/4.2
#export PATH=/home/u4/nickyang69/software/orca_5_0_2_linux_x86-64_shared_openmpi411:$PATH
#export LD_LIBRARY_PATH=/home/u4/nickyang69/software/orca_5_0_2_linux_x86-64_shared_openmpi411:$LD_LIBRARY_PATH

for i in $(ls *inp | sort -n);
do
  orca $i > ${i%.???}.out
done

