#!/bin/bash
#PBS -N boosting
#PBS -o job-out.log
#PBS -e job-err.log
#PBS -l nodes=compute2:ppn=104
#PBS -q cpu

cd $PBS_O_WORKDIR
source .venv/bin/activate

python -m models xgboost --grid-search