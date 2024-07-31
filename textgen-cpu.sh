#!/bin/bash

#SBATCH --job-name=textgen-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1

#SBATCH --time=7-00:00:00
#SBATCH --partition=long,xlong

#SBATCH --mem=128GB
#SBATCH --output=/home/anthony.li/out/textgen-cpu.%j.out
#SBATCH --error=/home/anthony.li/out/textgen-cpu.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anthony.li@tamu.edu

module purge
module load JupyterLab/4.0.5-GCCcore-12.3.0

cd /home/anthony.li/llm-watermark-cpd

mkdir -p results
mkdir -p log

echo "Starting job with ID ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"
echo $(which python)

export PYTHONPATH=".":$PYTHONPATH
export HF_HOME=/scratch/user/anthony.li/hf_cache

n=1000

for method in gumbel transform; do
  python textgen.py --save results/ml3-watermark500-$method.p --n $n --batch_size 25 --m 500 --model meta-llama/Meta-Llama-3-8B --seed 1 --T 10 --k 20 --method $method
  python textgen.py --save results/ml3-watermark250-nowatermark250-$method.p --n $n --batch_size 25 --m 250 --model meta-llama/Meta-Llama-3-8B --seed 1 --T 10 --k 20 --method $method --insertion_blocks_start 250 --insertion_blocks_length 250
  python textgen.py --save results/ml3-watermark200-nowatermark100-watermark200-$method.p --n $n --batch_size 25 --m 500 --model meta-llama/Meta-Llama-3-8B --seed 1 --T 10 --k 20 --method $method --substitution_blocks_start 200 --substitution_blocks_end 300
  python textgen.py --save results/ml3-watermark100-nowatermark100-watermark100-nowatermark100-watermark100-$method.p --n $n --batch_size 25 --m 400 --model meta-llama/Meta-Llama-3-8B --seed 1 --T 10 --k 20 --method $method --substitution_blocks_start 100 --substitution_blocks_end 200 --insertion_blocks_start 300 --insertion_blocks_length 100
done
