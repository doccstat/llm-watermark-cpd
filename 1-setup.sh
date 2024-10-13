#!/bin/bash

#SBATCH --job-name=setup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-00:02:00
#SBATCH --partition=short,medium,long,xlong
#SBATCH --mem-per-cpu=1GB
#SBATCH --output=/home/anthony.li/out/setup.%j.out
#SBATCH --error=/home/anthony.li/out/setup.%j.err
##SBATCH --mail-type=ALL
##SBATCH --mail-user=anthony.li@tamu.edu

module purge
module load Python/3.11.5-GCCcore-13.2.0

cd /home/anthony.li/llm-watermark-cpd

export HF_HOME=/scratch/user/anthony.li/hf_cache

python 1-setup.py build_ext --inplace