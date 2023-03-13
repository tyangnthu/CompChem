#!/bin/sh

export PATH=/home/u4/nickyang69/software/orca_5_0_2_linux_x86-64_shared_openmpi411:$PATH
export LD_LIBRARY_PATH=/home/u4/nickyang69/software/orca_5_0_2_linux_x86-64_shared_openmpi411:$LD_LIBRARY_PATH

for i in *inp;
do
  /home/u4/nickyang69/software/orca_5_0_2_linux_x86-64_shared_openmpi411/orca $i > ${i%.???}.out
done

#for i in $(ls *R-nr* | sort -n); do grep 'FINAL SINGLE' $i | awk '{print $5}'; done
