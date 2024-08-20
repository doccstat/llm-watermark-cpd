#!/bin/bash

#SBATCH --job-name=extra-textgen
#SBATCH --nodes=1
#SBATCH --ntasks=100
#SBATCH --ntasks-per-node=100
#SBATCH --cpus-per-task=1

#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu,xgpu
#SBATCH --gres=gpu:a30:2

#SBATCH --mem=100GB
#SBATCH --output=/home/anthony.li/out/extra-textgen.%j.out
#SBATCH --error=/home/anthony.li/out/extra-textgen.%j.err
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

python extra-textgen.py \
  --save results/ml3-random-3cpts-text1-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 3 \
  --npseed 1

python extra-textgen.py \
  --save results/ml3-random-3cpts-text2-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 3 \
  --npseed 2

python extra-textgen.py \
  --save results/ml3-random-3cpts-text3-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 3 \
  --npseed 3

python extra-textgen.py \
  --save results/ml3-random-3cpts-text4-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 3 \
  --npseed 4

python extra-textgen.py \
  --save results/ml3-random-3cpts-text5-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 3 \
  --npseed 5

################################################################################

python extra-textgen.py \
  --save results/ml3-random-4cpts-text1-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 4 \
  --npseed 1

python extra-textgen.py \
  --save results/ml3-random-4cpts-text2-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 4 \
  --npseed 2

python extra-textgen.py \
  --save results/ml3-random-4cpts-text3-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 4 \
  --npseed 3

python extra-textgen.py \
  --save results/ml3-random-4cpts-text4-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 4 \
  --npseed 4

python extra-textgen.py \
  --save results/ml3-random-4cpts-text5-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 4 \
  --npseed 5

################################################################################

python extra-textgen.py \
  --save results/ml3-random-6cpts-text1-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 6 \
  --npseed 1

python extra-textgen.py \
  --save results/ml3-random-6cpts-text2-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 6 \
  --npseed 2

python extra-textgen.py \
  --save results/ml3-random-6cpts-text3-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 6 \
  --npseed 3

python extra-textgen.py \
  --save results/ml3-random-6cpts-text4-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 6 \
  --npseed 4

python extra-textgen.py \
  --save results/ml3-random-6cpts-text5-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 6 \
  --npseed 5

################################################################################

python extra-textgen.py \
  --save results/ml3-random-8cpts-text1-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 8 \
  --npseed 1

python extra-textgen.py \
  --save results/ml3-random-8cpts-text2-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 8 \
  --npseed 2

python extra-textgen.py \
  --save results/ml3-random-8cpts-text3-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 8 \
  --npseed 3

python extra-textgen.py \
  --save results/ml3-random-8cpts-text4-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 8 \
  --npseed 4

python extra-textgen.py \
  --save results/ml3-random-8cpts-text5-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 8 \
  --npseed 5

################################################################################

python extra-textgen.py \
  --save results/ml3-random-9cpts-text1-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 9 \
  --npseed 1

python extra-textgen.py \
  --save results/ml3-random-9cpts-text2-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 9 \
  --npseed 2

python extra-textgen.py \
  --save results/ml3-random-9cpts-text3-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 9 \
  --npseed 3

python extra-textgen.py \
  --save results/ml3-random-9cpts-text4-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 9 \
  --npseed 4

python extra-textgen.py \
  --save results/ml3-random-9cpts-text5-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 9 \
  --npseed 5

################################################################################

python extra-textgen.py \
  --save results/ml3-random-12cpts-text1-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 12 \
  --npseed 1

python extra-textgen.py \
  --save results/ml3-random-12cpts-text2-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 12 \
  --npseed 2

python extra-textgen.py \
  --save results/ml3-random-12cpts-text3-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 12 \
  --npseed 3

python extra-textgen.py \
  --save results/ml3-random-12cpts-text4-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 12 \
  --npseed 4

python extra-textgen.py \
  --save results/ml3-random-12cpts-text5-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --cpts 12 \
  --npseed 5

################################################################################

python extra-textgen.py \
  --save results/ml3-concat-text5-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --meaningful \
  --buffer_tokens 0

python extra-textgen.py \
  --save results/ml3-concat-french-text5-gumbel.p \
  --watermark_key_length $n \
  --batch_size 10 \
  --tokens_count 500 \
  --model meta-llama/Meta-Llama-3-8B \
  --seed 1 \
  --T 1 \
  --k 20 \
  --method gumbel \
  --meaningful \
  --rt_translate \
  --language french \
  --buffer_tokens 0
