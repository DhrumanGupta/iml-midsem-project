#!/bin/bash
#PBS -N sir-diffusion
#PBS -o job-out.log
#PBS -e job-err.log
#PBS -l nodes=gpu-h100:ppn=30

cd $PBS_O_WORKDIR
source .venv/bin/activate

python -m models diffusion