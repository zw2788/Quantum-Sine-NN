#!/bin/bash
#SBATCH --partition=tiny
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=04:00:00
#SBATCH --output=log_N18_22L10.out

module load miniconda/python-3.9
source activate sinenv

python DSinNN_N18_22L10.py
