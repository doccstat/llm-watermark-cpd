#!/bin/bash

#SBATCH --job-name=detect
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0-02:00:00
#SBATCH --partition=short,medium,long,xlong
#SBATCH --mem-per-cpu=1GB
#SBATCH --output=/home/anthony.li/out/detect.%A.%a.out
#SBATCH --error=/home/anthony.li/out/detect.%A.%a.err
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=anthony.li@tamu.edu
#SBATCH --array=1-1000

module purge
module load JupyterLab/4.0.5-GCCcore-12.3.0

cd /home/anthony.li/llm-watermark-cpd

echo "Starting job with ID ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"

export HF_HOME=/scratch/user/anthony.li/hf_cache

# Calculate the starting and ending line numbers for the commands
start_command=$(( (${SLURM_ARRAY_TASK_ID} - 1) * 60 + 1 ))
end_command=$((${SLURM_ARRAY_TASK_ID} * 60))

echo "Running tasks for commands from $start_command to $end_command"

# Loop over the designated commands for this job
for i in $(seq $start_command $end_command); do
    command=$(sed -n "${i}p" detect-commands.sh)
    echo "Executing command $i: $command"
    eval "$command"
done
