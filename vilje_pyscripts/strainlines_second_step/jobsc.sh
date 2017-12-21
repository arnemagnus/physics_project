#!/bin/bash
#PBS -N strainlinesrun
#PBS -A #######
#PBS -l walltime=01:30:00
#PBS -l select=1:ncpus=32

cd $PBS_O_WORKDIR

module load intelcomp/17.0.0 python/3.6.3

python3 strainline_iteration_script.py rkdp87 0.1 1e-12 1e-12
