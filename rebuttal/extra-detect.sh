#!/bin/bash

#SBATCH --job-name=extra-detect
#SBATCH --ntasks=33
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=medium,long,xlong
#SBATCH --mem-per-cpu=1GB
#SBATCH --output=/home/anthony.li/out/extra-detect.%A.%a.out
#SBATCH --error=/home/anthony.li/out/extra-detect.%A.%a.err
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=anthony.li@tamu.edu
#SBATCH --array=1-500

module purge
module load Python/3.11.5-GCCcore-13.2.0

cd /home/anthony.li/llm-watermark-cpd

echo "Starting job with ID ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"

export HF_HOME=/scratch/user/anthony.li/hf_cache

# Calculate fixed_i as SLURM_ARRAY_TASK_ID - 1
fixed_i=$(($SLURM_ARRAY_TASK_ID - 1))

python extra-detect.py --token_file "results/ml3-random-3cpts-text1-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-3cpts-text2-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-3cpts-text3-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-3cpts-text4-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-3cpts-text5-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &

python extra-detect.py --token_file "results/ml3-random-4cpts-text1-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-4cpts-text2-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-4cpts-text3-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-4cpts-text4-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-4cpts-text5-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &

python extra-detect.py --token_file "results/ml3-random-6cpts-text1-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-6cpts-text2-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-6cpts-text3-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-6cpts-text4-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-6cpts-text5-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &

python extra-detect.py --token_file "results/ml3-random-8cpts-text1-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-8cpts-text2-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-8cpts-text3-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-8cpts-text4-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-8cpts-text5-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &

python extra-detect.py --token_file "results/ml3-random-9cpts-text1-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-9cpts-text2-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-9cpts-text3-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-9cpts-text4-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-9cpts-text5-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &

python extra-detect.py --token_file "results/ml3-random-12cpts-text1-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-12cpts-text2-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-12cpts-text3-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-12cpts-text4-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-random-12cpts-text5-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &

python extra-detect.py --token_file "results/ml3-concat-text5-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &
python extra-detect.py --token_file "results/ml3-concat-french-text5-gumbel.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method gumbel --fixed_i ${fixed_i} &

python extra-detect.py --token_file "results/ml3-random-3cpts-text1-kirchenbauer.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex 0 --k 20 --method kirchenbauer --fixed_i ${fixed_i} --kirch_delta 1.5

wait
