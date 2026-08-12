#!/bin/bash

#SBATCH -t 0-72:00:00
#SBATCH --mem=2G
#SBATCH -J LAMPS
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH --output=test_%j_out.txt
#SBATCH --error=test_%j_err.txt

module load miniforge3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh

conda activate lamps

"$@"

conda deactivate
