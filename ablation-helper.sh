#!/bin/bash

export PYTHONPATH=".":$PYTHONPATH

method=$1
Tindex=$2
cpts=$3
k=$4
fixed_i=$5
n_runs=$6

n=1000

python ablation.py --token_file "results/ml3-${cpts}changepoints-${method}.p" --n ${n} --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex ${Tindex} --k ${k} --method ${method} --fixed_i ${fixed_i} --n_runs ${n_runs}
