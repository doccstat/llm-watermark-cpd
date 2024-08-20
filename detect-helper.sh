#!/bin/bash

export PYTHONPATH=".":$PYTHONPATH

method=$1
Tindex=$2
cpts=$3
k=$4
fixed_i=$5

n=1000

# python detect.py --token_file "results/opt-${cpts}changepoints-${method}.p" --n ${n} --model facebook/opt-1.3b --seed 1 --Tindex ${Tindex} --k ${k} --method ${method} --fixed_i ${fixed_i}
# python detect.py --token_file "results/gpt-${cpts}changepoints-${method}.p" --n ${n} --model openai-community/gpt2 --seed 1 --Tindex ${Tindex} --k ${k} --method ${method} --fixed_i ${fixed_i}
python detect.py --token_file "results/ml3-${cpts}changepoints-${method}.p" --n ${n} --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex ${Tindex} --k ${k} --method ${method} --fixed_i ${fixed_i}
# python detect.py --token_file "results/ml3-comment-${method}-${k}.p" --n ${n} --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex ${Tindex} --k ${k} --method ${method} --fixed_i ${fixed_i}
