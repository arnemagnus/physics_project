#!/bin/bash
#PBS -N advectionruns
#PBS -A ########
#PBS -l walltime=02:00:00
#PBS -l select=64:ncpus=32:mpiprocs=16

cd $PBS_O_WORKDIR

module load intelcomp/17.0.0 mpt/2.14 python/3.6.3

mpiexec -n 1024 python3 advect_script.py euler 0.000001 1.e-1 1.e-1
