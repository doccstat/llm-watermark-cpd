#!/bin/bash

#SBATCH --job-name=mllm-textgen
#SBATCH --nodes=1
#SBATCH --ntasks=100
#SBATCH --ntasks-per-node=100
#SBATCH --cpus-per-task=1

#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu,xgpu
#SBATCH --gres=gpu:a30:3

#SBATCH --mem=100GB
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
  --save results/ml3-mllm-1-gumbel.p \
  --watermark_key_length $n \
  --tokens_count 300 \
  --model1 meta-llama/Meta-Llama-3-8B \
  --model2 openai-community/gpt2 \
  --seed 1 \
  --k 20 \
  --buffer_tokens 0 \
  --method gumbel

python mllm-textgen.py \
  --save results/ml3-mllm-2-gumbel.p \
  --watermark_key_length $n \
  --tokens_count 300 \
  --model1 meta-llama/Meta-Llama-3-8B \
  --model2 openai-community/gpt2 \
  --seed 1 \
  --k 20 \
  --buffer_tokens 0 \
  --method gumbel \
  --skip 100

python mllm-textgen.py \
  --save results/ml3-mllm-3-gumbel.p \
  --watermark_key_length $n \
  --tokens_count 300 \
  --model1 meta-llama/Meta-Llama-3-8B \
  --model2 openai-community/gpt2 \
  --seed 1 \
  --k 20 \
  --buffer_tokens 0 \
  --method gumbel \
  --skip 200

python mllm-textgen.py \
  --save results/ml3-mllm-4-gumbel.p \
  --watermark_key_length $n \
  --tokens_count 300 \
  --model1 meta-llama/Meta-Llama-3-8B \
  --model2 openai-community/gpt2 \
  --seed 1 \
  --k 20 \
  --buffer_tokens 0 \
  --method gumbel \
  --skip 300

python mllm-textgen.py \
  --save results/ml3-mllm-5-gumbel.p \
  --watermark_key_length $n \
  --tokens_count 300 \
  --model1 meta-llama/Meta-Llama-3-8B \
  --model2 openai-community/gpt2 \
  --seed 1 \
  --k 20 \
  --buffer_tokens 0 \
  --method gumbel \
  --skip 400
