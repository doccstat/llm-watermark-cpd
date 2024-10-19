#!/bin/bash

#SBATCH --job-name=textgen
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --time=0-00:40:00
#SBATCH --partition=gpu,xgpu
#SBATCH --gres=gpu:a30:2

#SBATCH --mem=25GB
#SBATCH --output=/home/anthony.li/llm-watermark-cpd/log/textgen.%A.%a.out
#SBATCH --error=/home/anthony.li/llm-watermark-cpd/log/textgen.%A.%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=anthony.li@tamu.edu
#SBATCH --array=1-38

module purge
module load Python/3.11.5-GCCcore-13.2.0

cd /home/anthony.li/llm-watermark-cpd

echo "Starting job with ID ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"

export PATH="/home/anthony.li/.local/bin:$PATH"
export PYTHONPATH=".":$PYTHONPATH
export HF_HOME=/scratch/user/anthony.li/hf_cache

# Determine the total number of commands by counting lines in 2-textgen-commands.sh
total_commands=$(wc -l < 2-textgen-commands.sh)

# The total number of concurrently running jobs. This number should be the
# smaller of the maximum array id and the total number of commands.
total_jobs=38

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
  command=$(sed -n "${i}p" 2-textgen-commands.sh)
  echo "Executing command $i: $command"
  eval "$command"
done
