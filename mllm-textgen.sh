#!/bin/bash

#SBATCH --job-name=mllm-textgen
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1

#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu,xgpu
#SBATCH --gres=gpu:a30:3

#SBATCH --mem=128GB
#SBATCH --output=/home/anthony.li/out/mllm-textgen.%j.out
#SBATCH --error=/home/anthony.li/out/mllm-textgen.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anthony.li@tamu.edu

module purge
module load Python/3.11.5-GCCcore-13.2.0

cd /home/anthony.li/llm-watermark-cpd

mkdir -p results
mkdir -p log

echo "Starting job with ID ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"
echo $(which python)

export PATH="/home/anthony.li/.local/bin:$PATH"
export PYTHONPATH=".":$PYTHONPATH
export HF_HOME=/scratch/user/anthony.li/hf_cache

n=1000

python mllm-textgen.py \
  --save results/ml3-mllm-gumbel.p \
  --watermark_key_length $n \
  --tokens_count 300 \
  --model1 meta-llama/Meta-Llama-3-8B \
  --model2 mistralai/Mistral-Nemo-Instruct-2407 \
  --seed 1 \
  --k 20 \
  --method gumbel
