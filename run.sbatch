#!/bin/bash
#Here is a comment
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=TransPolymer
#SBATCH --account=rcc-staff
#SBATCH --output=TransPolymer-%j.out
#SBATCH --error=TransPolymer-%j.err
#SBATCH --mail-user=hyadav@uchicago.edu
#SBATCH --mail-type=BEGIN

module load python/anaconda-2022.05

source activate TransPolymer

python Downstream.py