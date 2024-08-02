#!/bin/bash

#SBATCH --job-name=textgen
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1

#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu,xgpu
#SBATCH --gres=gpu:a30:2

#SBATCH --mem=128GB
#SBATCH --output=/home/anthony.li/out/textgen.%j.out
#SBATCH --error=/home/anthony.li/out/textgen.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anthony.li@tamu.edu

module purge
module load JupyterLab/4.0.5-GCCcore-12.3.0

cd /home/anthony.li/llm-watermark-cpd

mkdir -p results
mkdir -p log

echo "Starting job with ID ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"
echo $(which python)

export PATH="/home/anthony.li/.local/bin:$PATH"
export PYTHONPATH=".":$PYTHONPATH
export HF_HOME=/scratch/user/anthony.li/hf_cache

n=1000

# for method in gumbel transform; do
#   # 500 tokens with 0 change points
#   python textgen.py \
#     --save results/opt-0changepoints-$method.p \
#     --watermark_key_length $n \
#     --batch_size 25 \
#     --tokens_count 500 \
#     --model facebook/opt-1.3b \
#     --seed 1 \
#     --T 500 \
#     --k 20 \
#     --method $method
#   # 500 tokens with 1 change point
#   python textgen.py \
#     --save results/opt-1changepoints-$method.p \
#     --watermark_key_length $n \
#     --batch_size 25 \
#     --tokens_count 250 \
#     --model facebook/opt-1.3b \
#     --seed 1 \
#     --T 500 \
#     --k 20 \
#     --method $method \
#     --insertion_blocks_start 250 \
#     --insertion_blocks_length 250
#   # 500 tokens with 2 change points
#   python textgen.py \
#     --save results/opt-2changepoints-$method.p \
#     --watermark_key_length $n \
#     --batch_size 25 \
#     --tokens_count 500 \
#     --model facebook/opt-1.3b \
#     --seed 1 \
#     --T 500 \
#     --k 20 \
#     --method $method \
#     --substitution_blocks_start 200 \
#     --substitution_blocks_end 300
#   # 500 tokens with 4 change points
#   python textgen.py \
#     --save results/opt-4changepoints-$method.p \
#     --watermark_key_length $n \
#     --batch_size 25 \
#     --tokens_count 400 \
#     --model facebook/opt-1.3b \
#     --seed 1 \
#     --T 500 \
#     --k 20 \
#     --method $method \
#     --substitution_blocks_start 100 \
#     --substitution_blocks_end 200 \
#     --insertion_blocks_start 300 \
#     --insertion_blocks_length 100
# done

# for method in gumbel transform; do
#   # 500 tokens with 0 change points
#   python textgen.py \
#     --save results/gpt-0changepoints-$method.p \
#     --watermark_key_length $n \
#     --batch_size 25 \
#     --tokens_count 500 \
#     --model openai-community/gpt2 \
#     --seed 1 \
#     --T 500 \
#     --k 20 \
#     --method $method
#   # 500 tokens with 1 change point
#   python textgen.py \
#     --save results/gpt-1changepoints-$method.p \
#     --watermark_key_length $n \
#     --batch_size 25 \
#     --tokens_count 250 \
#     --model openai-community/gpt2 \
#     --seed 1 \
#     --T 500 \
#     --k 20 \
#     --method $method \
#     --insertion_blocks_start 250 \
#     --insertion_blocks_length 250
#   # 500 tokens with 2 change points
#   python textgen.py \
#     --save results/gpt-2changepoints-$method.p \
#     --watermark_key_length $n \
#     --batch_size 25 \
#     --tokens_count 500 \
#     --model openai-community/gpt2 \
#     --seed 1 \
#     --T 500 \
#     --k 20 \
#     --method $method \
#     --substitution_blocks_start 200 \
#     --substitution_blocks_end 300
#   # 500 tokens with 4 change points
#   python textgen.py \
#     --save results/gpt-4changepoints-$method.p \
#     --watermark_key_length $n \
#     --batch_size 25 \
#     --tokens_count 400 \
#     --model openai-community/gpt2 \
#     --seed 1 \
#     --T 500 \
#     --k 20 \
#     --method $method \
#     --substitution_blocks_start 100 \
#     --substitution_blocks_end 200 \
#     --insertion_blocks_start 300 \
#     --insertion_blocks_length 100
# done

for method in gumbel transform; do
  # 500 tokens with 0 change points
  python textgen.py \
    --save results/ml3-0changepoints-$method.p \
    --watermark_key_length $n \
    --batch_size 10 \
    --tokens_count 500 \
    --model meta-llama/Meta-Llama-3-8B \
    --seed 1 \
    --T 10 \
    --k 20 \
    --method $method
  # 500 tokens with 1 change point
  # 1-250: watermark, 250: insert
  python textgen.py \
    --save results/ml3-1changepoints-$method.p \
    --watermark_key_length $n \
    --batch_size 10 \
    --tokens_count 250 \
    --model meta-llama/Meta-Llama-3-8B \
    --seed 1 \
    --T 10 \
    --k 20 \
    --method $method \
    --insertion_blocks_start 250 \
    --insertion_blocks_length 250
  # 500 tokens with 2 change points
  # 1-200: watermark
  # 201-300: substitute
  # 301-500: watermark
  python textgen.py \
    --save results/ml3-2changepoints-$method.p \
    --watermark_key_length $n \
    --batch_size 10 \
    --tokens_count 500 \
    --model meta-llama/Meta-Llama-3-8B \
    --seed 1 \
    --T 10 \
    --k 20 \
    --method $method \
    --substitution_blocks_start 200 \
    --substitution_blocks_end 300
  # 500 tokens with 4 change points
  # 1-100: watermark
  # 101-200: substitute
  # 201-300: watermark, 100: insert
  # 301-400: watermark
  python textgen.py \
    --save results/ml3-4changepoints-$method.p \
    --watermark_key_length $n \
    --batch_size 10 \
    --tokens_count 400 \
    --model meta-llama/Meta-Llama-3-8B \
    --seed 1 \
    --T 10 \
    --k 20 \
    --method $method \
    --substitution_blocks_start 100 \
    --substitution_blocks_end 200 \
    --insertion_blocks_start 300 \
    --insertion_blocks_length 100
  # 500 tokens with 9 change points
  # 1-50: watermark
  # 51-100: substitute
  # 101-150: watermark, 50: insert
  # 150-200: watermark
  # 201-250: substitute
  # 251-300: watermark, 50: insert
  # 301-350: watermark
  # 351-400: substitute
  python textgen.py \
    --save results/ml3-9changepoints-$method.p \
    --watermark_key_length $n \
    --batch_size 10 \
    --tokens_count 400 \
    --model meta-llama/Meta-Llama-3-8B \
    --seed 1 \
    --T 10 \
    --k 20 \
    --method $method \
    --substitution_blocks_start 50,200,350 \
    --substitution_blocks_end 100,250,400 \
    --insertion_blocks_start 150,300 \
    --insertion_blocks_length 50,50
  # 500 tokens with 19 change points
  # 1-25: watermark
  # 26-50: substitute
  # 51-75: watermark, 25: insert
  # 76-100: watermark
  # 101-125: substitute
  # 126-150: watermark, 25: insert
  # 151-175: watermark
  # 176-200: substitute
  # 201-225: watermark
  # 226-250: substitute
  # 251-275: watermark
  # 276-300: substitute
  # 301-325: watermark, 25: insert
  # 326-350: watermark, 25: insert
  # 351-375: watermark
  # 376-400: substitute
  python textgen.py \
    --save results/ml3-19changepoints-$method.p \
    --watermark_key_length $n \
    --batch_size 10 \
    --tokens_count 400 \
    --model meta-llama/Meta-Llama-3-8B \
    --seed 1 \
    --T 10 \
    --k 20 \
    --method $method \
    --substitution_blocks_start 25,100,175,225,275,375 \
    --substitution_blocks_end 50,125,200,250,300,400 \
    --insertion_blocks_start 75,150,325,350 \
    --insertion_blocks_length 25,25,25,25
done
