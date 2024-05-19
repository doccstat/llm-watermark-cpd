#!/bin/bash

export PYTHONPATH=".":$PYTHONPATH

method=$1
Tindex=$2
setting=$3

python detect.py --token_file "results/opt-${setting}-${method}.pfacebook.opt-1.3b${method}" --n 256 --model facebook/opt-1.3b --seed 1 --Tindex ${Tindex} --k 20 --method ${method}
python detect.py --token_file "results/gpt-${setting}-${method}.popenai-community.gpt2${method}" --n 256 --model openai-community/gpt2 --seed 1 --Tindex ${Tindex} --k 20 --method ${method}
