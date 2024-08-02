#!/bin/bash

#SBATCH --job-name=seedbs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --partition=medium,long,xlong
#SBATCH --mem-per-cpu=1GB
#SBATCH --output=/home/anthony.li/out/seedbs.%A.%a.out
#SBATCH --error=/home/anthony.li/out/seedbs.%A.%a.err
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=anthony.li@tamu.edu
#SBATCH --array=1-1000

module purge
module load JupyterLab/4.0.5-GCCcore-12.3.0

cd /home/anthony.li/llm-watermark-cpd

echo "Starting job with ID ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"

# Total number of commands and jobs
total_commands=11280
total_jobs=1000

# Calculate the number of commands per job (minimum)
commands_per_job=$((total_commands / total_jobs))

# Calculate the number of jobs that need to process an extra command
extra_commands=$((total_commands % total_jobs))

# Determine the start and end command index for this particular job
if [ ${SLURM_ARRAY_TASK_ID} -le $extra_commands ]; then
    start_command=$(( (${SLURM_ARRAY_TASK_ID} - 1) * (commands_per_job + 1) + 1 ))
    end_command=$(( ${SLURM_ARRAY_TASK_ID} * (commands_per_job + 1) ))
else
    start_command=$(( extra_commands * (commands_per_job + 1) + (${SLURM_ARRAY_TASK_ID} - extra_commands - 1) * commands_per_job + 1 ))
    end_command=$(( extra_commands * (commands_per_job + 1) + (${SLURM_ARRAY_TASK_ID} - extra_commands) * commands_per_job ))
fi

echo "Running tasks for commands from $start_command to $end_command"

# Loop over the designated commands for this job
for i in $(seq $start_command $end_command); do
    command=$(sed -n "${i}p" seedbs-commands.sh)
    echo "Executing command $i: $command"
    eval "$command"
done
