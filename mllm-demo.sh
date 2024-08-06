#!/bin/bash

#SBATCH --job-name=mllm-demo
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=medium,long,xlong
#SBATCH --mem-per-cpu=1GB
#SBATCH --output=/home/anthony.li/out/mllm-demo.%A.%a.out
#SBATCH --error=/home/anthony.li/out/mllm-demo.%A.%a.err
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=anthony.li@tamu.edu

module purge
module load Python/3.11.5-GCCcore-13.2.0

cd /home/anthony.li/llm-watermark-cpd

echo "Starting job with ID ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"

export HF_HOME=/scratch/user/anthony.li/hf_cache

# Calculate fixed_i as SLURM_ARRAY_TASK_ID - 1
fixed_i=$(($SLURM_ARRAY_TASK_ID - 1))

python mllm-demo.py --token_file "results/ml3-mllm-gumbel.p" --model openai-community/gpt2 --seed 1 --detected_cpts "50,60" &

wait
