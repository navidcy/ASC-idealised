#!/bin/bash
#PBS -q gpuvolta
#PBS -P x77
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=190GB
#PBS -l jobfs=100GB
#PBS -l walltime=24:00:00
#PBS -l wd
#PBS -N asc
#PBS -W umask=027
#PBS -l storage=gdata/v45+gdata/hh5+scratch/v45+scratch/x77+gdata/x77

# Load modules.
export JULIA_DEPOT_PATH=/g/data/v45/nc3020/.julia:$JULIA_ROOT/share/julia/site/
export JULIA_LOAD_PATH="@":"@v#.#":"@stdlib":"@site"
export JULIA_CUDA_USE_BINARYBUILDER="false"
export JULIA_NUM_THREADS=48

# Run Julia
cd /g/data/v45/nc3020/ASC-idealised/
/g/data/v45/nc3020/julia/julia --check-bounds=no --color=yes --project asc.jl > $PBS_JOBID.log
