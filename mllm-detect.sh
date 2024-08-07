#!/bin/bash

#SBATCH --job-name=mllm-detect
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=medium,long,xlong
#SBATCH --mem-per-cpu=1GB
#SBATCH --output=/home/anthony.li/out/mllm-detect.%A.%a.out
#SBATCH --error=/home/anthony.li/out/mllm-detect.%A.%a.err
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=anthony.li@tamu.edu
#SBATCH --array=1-300

module purge
module load Python/3.11.5-GCCcore-13.2.0

cd /home/anthony.li/llm-watermark-cpd

echo "Starting job with ID ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"

export HF_HOME=/scratch/user/anthony.li/hf_cache

# Calculate fixed_i as SLURM_ARRAY_TASK_ID - 1
fixed_i=$(($SLURM_ARRAY_TASK_ID - 1))

python mllm-detect.py --token_file "results/ml3-mllm-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python mllm-detect.py --token_file "results/ml3-mllm-gumbel.p" --n 1000 --model openai-community/gpt2 --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &

python mllm-detect.py --token_file "results/ml3-mllm-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 1 --k 20 --method gumbel --fixed_i ${fixed_i} &
python mllm-detect.py --token_file "results/ml3-mllm-gumbel.p" --n 1000 --model openai-community/gpt2 --seed 1 --Tindex 1 --k 20 --method gumbel --fixed_i ${fixed_i} &

python mllm-detect.py --token_file "results/ml3-mllm-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 2 --k 20 --method gumbel --fixed_i ${fixed_i} &
python mllm-detect.py --token_file "results/ml3-mllm-gumbel.p" --n 1000 --model openai-community/gpt2 --seed 1 --Tindex 2 --k 20 --method gumbel --fixed_i ${fixed_i} &

python mllm-detect.py --token_file "results/ml3-mllm-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 3 --k 20 --method gumbel --fixed_i ${fixed_i} &
python mllm-detect.py --token_file "results/ml3-mllm-gumbel.p" --n 1000 --model openai-community/gpt2 --seed 1 --Tindex 3 --k 20 --method gumbel --fixed_i ${fixed_i} &

python mllm-detect.py --token_file "results/ml3-mllm-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 4 --k 20 --method gumbel --fixed_i ${fixed_i} &
python mllm-detect.py --token_file "results/ml3-mllm-gumbel.p" --n 1000 --model openai-community/gpt2 --seed 1 --Tindex 4 --k 20 --method gumbel --fixed_i ${fixed_i} &

wait
