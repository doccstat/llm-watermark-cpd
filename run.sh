#!/bin/bash

# Provide a name for your job, so it may be recognized in the output of squeue
# SBATCH --job-name=watermark

# Define how many nodes this job needs.
# This example uses one 1 node.  Recall that each node has 128 CPU cores.
#SBATCH --nodes=1

#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1

# Define a maximum amount of time the job will run in real time. This is a hard
# upper bound, meaning that if the job runs longer than what is written here, it
# will be terminated by the server.
#              d-hh:mm:ss
#SBATCH --time=1-00:00:00

# Define the partition on which the job shall run.
#SBATCH --partition=gpu

# Define how much memory you need. Choose one of the following:
# --mem will define memory per node and
# --mem-per-cpu will define memory per CPU/core.
##SBATCH --mem-per-cpu=1024MB
#SBATCH --mem=500GB        # The double hash means that this one is not in effect

# Define any general resources required by this job.  In this example 1 "a30"
# GPU is requested per node.  Note that gpu:1 would request any gpu type, if
# available.  This cluster currenlty only contains NVIDIA A30 GPUs.
#SBATCH --gres=gpu:a30:1

# Define the destination file name(s) for this batch scripts output.
# The use of '%j' here uses the job ID as part of the filename.
#SBATCH --output=/home/anthony.li/out/watermark.%j

# Turn on mail notification. There are many possible values, and more than one
# may be specified (using comma separated values):
# NONE, BEGIN, END, FAIL, REQUEUE, ALL, INVALID_DEPEND, STAGE_OUT, TIME_LIMIT,
# TIME_LIMIT_90, TIME_LIMIT_80, TIME_LIMIT_50 - See "man sbatch" or the slurm
# website for more values (https://slurm.schedmd.com/sbatch.html).
#SBATCH --mail-type=ALL

# The email address to which emails should be sent.
#SBATCH --mail-user=anthony.li@tamu.edu

# All commands should follow the last SBATCH directive.

# Define or set any necessary environment variables for this job.
# Note that several environment variables have been defined for you, and two
# of particular interest are:
#   SCRATCH=/scratch/user/NetID  # This folder is accessible from any node.
#   TMPDIR=/tmp/job.%j  # This folder is automatically created / destroyed for
#                       # you at the start / end of each job. This folder exists
#                       # locally on a compute node using a fast local disk.  It
#                       # is not directly accessible from any other node.

# As an example, if your application requires the loading of many files it may
# be faster, and certainly more efficient, to first copy those files to TMPDIR.
# Doing so ensures that the files are copied across the network once, and are
# accessible to the application locally on each node using a fast disk.
# cp watermark/demo/* ${TMPDIR}

# Load any modules that are required.  Note that while the system does provide a
# default set of basic tools, it does not include all of the software you will
# need for your job.  As such you should specify the modules for the software
# packages and versions that your job needs here.
module purge
# module load Anaconda3/2024.02-1
# module load CUDA/12.4.0
module load JupyterLab/4.0.5-GCCcore-12.3.0
# module load PyTorch/2.1.2-foss-2023b
module load R/4.3.2-gfbf-2023a

# get unused socket per https://unix.stackexchange.com/a/132524
readonly PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# This is where the actual work is done.  Execute your application, passing
# any parameters requried.

cat 1>&2 <<END
1. SSH tunnel from your workstation using the following command:

   ssh -L ${PORT}:${HOSTNAME}:${PORT} -N ${USER}@arseven.stat.tamu.edu

2. Use the URL returned by Jupyter that looks similar to the following:

   http://127.0.0.1:${PORT}/lab?token=b16726df7fbb0f05142df6cb40ea279c517fc86c8ee4a86c

When done using Jupyter, terminate the job by:

1. Issue the following command on the login node:

      scancel -f ${SLURM_JOB_ID}
END

# jupyter-lab --no-browser --ip ${HOSTNAME} --port ${PORT}

cd /home/anthony.li/llm-watermark-cpd

python setup.py build_ext --inplace

mkdir -p results

export PYTHONPATH=".":$PYTHONPATH

for method in gumbel transform; do
  python textgen.py --save results/opt-watermark500-$method.p --n 1000 --batch_size 25 --m 500 --model facebook/opt-1.3b --seed 1 --T 500 --k 20 --method $method
  python textgen.py --save results/opt-watermark250-nowatermark250-$method.p --n 1000 --batch_size 25 --m 250 --model facebook/opt-1.3b --seed 1 --T 500 --k 20 --method $method --insertion_blocks_start 250 --insertion_blocks_length 250
  python textgen.py --save results/opt-watermark200-nowatermark100-watermark200-$method.p --n 1000 --batch_size 25 --m 500 --model facebook/opt-1.3b --seed 1 --T 500 --k 20 --method $method --substitution_blocks_start 200 --substitution_blocks_end 300
  python textgen.py --save results/opt-watermark100-nowatermark100-watermark100-nowatermark100-watermark100-$method.p --n 1000 --batch_size 25 --m 400 --model facebook/opt-1.3b --seed 1 --T 500 --k 20 --method $method --substitution_blocks_start 100 --substitution_blocks_end 200 --insertion_blocks_start 300 --insertion_blocks_length 100
done

for method in gumbel transform; do
  python textgen.py --save results/gpt-watermark500-$method.p --n 1000 --batch_size 25 --m 500 --model openai-community/gpt2 --seed 1 --T 500 --k 20 --method $method
  python textgen.py --save results/gpt-watermark250-nowatermark250-$method.p --n 1000 --batch_size 25 --m 250 --model openai-community/gpt2 --seed 1 --T 500 --k 20 --method $method --insertion_blocks_start 250 --insertion_blocks_length 250
  python textgen.py --save results/gpt-watermark200-nowatermark100-watermark200-$method.p --n 1000 --batch_size 25 --m 500 --model openai-community/gpt2 --seed 1 --T 500 --k 20 --method $method --substitution_blocks_start 200 --substitution_blocks_end 300
  python textgen.py --save results/gpt-watermark100-nowatermark100-watermark100-nowatermark100-watermark100-$method.p --n 1000 --batch_size 25 --m 400 --model openai-community/gpt2 --seed 1 --T 500 --k 20 --method $method --substitution_blocks_start 100 --substitution_blocks_end 200 --insertion_blocks_start 300 --insertion_blocks_length 100
done
