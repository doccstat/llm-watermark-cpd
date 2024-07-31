#!/bin/bash

#SBATCH --job-name=textgen
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1

#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a30:1

#SBATCH --mem=128GB
#SBATCH --output=/home/anthony.li/out/textgen.%j
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anthony.li@tamu.edu

module purge
module load JupyterLab/4.0.5-GCCcore-12.3.0

cd /home/anthony.li/llm-watermark-cpd

mkdir -p results
mkdir -p log

export PYTHONPATH=".":$PYTHONPATH
export HF_HOME=/scratch/user/anthony.li/hf_cache

n=1000

for method in gumbel transform; do
  python textgen.py --save results/opt-watermark500-$method.p --watermark_key_length $n --batch_size 25 --tokens_count 500 --model facebook/opt-1.3b --seed 1 --T 500 --k 20 --method $method
  python textgen.py --save results/opt-watermark250-nowatermark250-$method.p --watermark_key_length $n --batch_size 25 --tokens_count 250 --model facebook/opt-1.3b --seed 1 --T 500 --k 20 --method $method --insertion_blocks_start 250 --insertion_blocks_length 250
  python textgen.py --save results/opt-watermark200-nowatermark100-watermark200-$method.p --watermark_key_length $n --batch_size 25 --tokens_count 500 --model facebook/opt-1.3b --seed 1 --T 500 --k 20 --method $method --substitution_blocks_start 200 --substitution_blocks_end 300
  python textgen.py --save results/opt-watermark100-nowatermark100-watermark100-nowatermark100-watermark100-$method.p --watermark_key_length $n --batch_size 25 --tokens_count 400 --model facebook/opt-1.3b --seed 1 --T 500 --k 20 --method $method --substitution_blocks_start 100 --substitution_blocks_end 200 --insertion_blocks_start 300 --insertion_blocks_length 100
done

for method in gumbel transform; do
  python textgen.py --save results/gpt-watermark500-$method.p --watermark_key_length $n --batch_size 25 --tokens_count 500 --model openai-community/gpt2 --seed 1 --T 500 --k 20 --method $method
  python textgen.py --save results/gpt-watermark250-nowatermark250-$method.p --watermark_key_length $n --batch_size 25 --tokens_count 250 --model openai-community/gpt2 --seed 1 --T 500 --k 20 --method $method --insertion_blocks_start 250 --insertion_blocks_length 250
  python textgen.py --save results/gpt-watermark200-nowatermark100-watermark200-$method.p --watermark_key_length $n --batch_size 25 --tokens_count 500 --model openai-community/gpt2 --seed 1 --T 500 --k 20 --method $method --substitution_blocks_start 200 --substitution_blocks_end 300
  python textgen.py --save results/gpt-watermark100-nowatermark100-watermark100-nowatermark100-watermark100-$method.p --watermark_key_length $n --batch_size 25 --tokens_count 400 --model openai-community/gpt2 --seed 1 --T 500 --k 20 --method $method --substitution_blocks_start 100 --substitution_blocks_end 200 --insertion_blocks_start 300 --insertion_blocks_length 100
done
