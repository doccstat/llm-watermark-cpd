#!/bin/bash

watermark_key_length=1000
permutation_count=999
seed=1

rm -f 3-detect-commands.sh

for method in gumbel transform; do
  for cpts in 0 1 2 4 9 19; do
    for model_prefix in opt gpt ml3; do
      if [ "$model_prefix" = "opt" ]; then
        model="facebook/opt-1.3b"
      elif [ "$model_prefix" = "gpt" ]; then
        model="openai-community/gpt2"
      else
        model="meta-llama/Meta-Llama-3-8B"
      fi

      for rolling_window_size in 20; do
        # rm -rf results/$model_prefix-$method-$watermark_key_length-$cpts-$rolling_window_size-$permutation_count-detect
        mkdir -p results/$model_prefix-$method-$watermark_key_length-$cpts-$rolling_window_size-$permutation_count-detect

        for prompt_index in $(seq 0 499); do
          for rolling_window_index in $(seq 0 499); do
            echo "python 3-detect.py --token_file results/$model_prefix-$method-$watermark_key_length-$cpts --model $model --method $method --watermark_key_length $watermark_key_length --rolling_window_size $rolling_window_size --permutation_count $permutation_count --seed $seed --Tindex $prompt_index --rolling_window_index $rolling_window_index" >> 3-detect-commands.sh
          done
        done
      done
    done
  done
done
