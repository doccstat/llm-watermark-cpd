#!/bin/bash

watermark_key_length=1000
number_of_experiments=500
seed=1
batch_size=25

rm 2-textgen-commands.sh

for model_prefix in ml3; do
  for method in gumbel transform; do
    if [ "$model_prefix" = "opt" ]; then
      model=facebook/opt-1.3b
    elif [ "$model_prefix" = "gpt" ]; then
      model=openai-community/gpt2
    else
      model=meta-llama/Meta-Llama-3-8B
    fi

    # 500 tokens with 0 change points
    echo "python 2-textgen.py --save results/$model_prefix-$method-$watermark_key_length-0 --model $model --method $method --watermark_key_length $watermark_key_length --T $number_of_experiments --seed $seed --batch_size $batch_size --tokens_count 500" >> 2-textgen-commands.sh
    # 500 tokens with 1 change point
    # 1-250: watermark, 250: insert
    echo "python 2-textgen.py --save results/$model_prefix-$method-$watermark_key_length-1 --model $model --method $method --watermark_key_length $watermark_key_length --T $number_of_experiments --seed $seed --batch_size $batch_size --tokens_count 250 --insertion_blocks_start 250 --insertion_blocks_length 250" >> 2-textgen-commands.sh
    # 500 tokens with 2 change points
    # 1-200: watermark
    # 201-300: substitute
    # 301-500: watermark
    echo "python 2-textgen.py --save results/$model_prefix-$method-$watermark_key_length-2 --model $model --method $method --watermark_key_length $watermark_key_length --T $number_of_experiments --seed $seed --batch_size $batch_size --tokens_count 500 --substitution_blocks_start 200 --substitution_blocks_end 300" >> 2-textgen-commands.sh
    # 500 tokens with 4 change points
    # 1-100: watermark
    # 101-200: substitute
    # 201-300: watermark, 100: insert
    # 301-400: watermark
    echo "python 2-textgen.py --save results/$model_prefix-$method-$watermark_key_length-4 --model $model --method $method --watermark_key_length $watermark_key_length --T $number_of_experiments --seed $seed --batch_size $batch_size --tokens_count 400 --substitution_blocks_start 100 --substitution_blocks_end 200 --insertion_blocks_start 300 --insertion_blocks_length 100" >> 2-textgen-commands.sh
    # 500 tokens with 9 change points
    # 1-50: watermark
    # 51-100: substitute
    # 101-150: watermark, 50: insert
    # 150-200: watermark
    # 201-250: substitute
    # 251-300: watermark, 50: insert
    # 301-350: watermark
    # 351-400: substitute
    echo "python 2-textgen.py --save results/$model_prefix-$method-$watermark_key_length-9 --model $model --method $method --watermark_key_length $watermark_key_length --T $number_of_experiments --seed $seed --batch_size $batch_size --tokens_count 400 --substitution_blocks_start 50,200,350 --substitution_blocks_end 100,250,400 --insertion_blocks_start 150,300 --insertion_blocks_length 50,50" >> 2-textgen-commands.sh
    # 500 tokens with 19 change points
    # 1-25: watermark
    # 26-50: substitute
    # 51-75: watermark, 25: insert
    # 76-100: watermark
    # 101-125: substitute
    # 126-150: watermark, 25: insert
    # 151-175: watermark
    # 176-200: substitute
    # 201-225: watermark
    # 226-250: substitute
    # 251-275: watermark
    # 276-300: substitute
    # 301-325: watermark, 25: insert
    # 326-350: watermark, 25: insert
    # 351-375: watermark
    # 376-400: substitute
    echo "python 2-textgen.py --save results/$model_prefix-$method-$watermark_key_length-19 --model $model --method $method --watermark_key_length $watermark_key_length --T $number_of_experiments --seed $seed --batch_size $batch_size --tokens_count 400 --substitution_blocks_start 25,100,175,225,275,375 --substitution_blocks_end 50,125,200,250,300,400 --insertion_blocks_start 75,150,325,350 --insertion_blocks_length 25,25,25,25" >> 2-textgen-commands.sh
  done
done

model_prefix=ml3
method=gumbel
model=meta-llama/Meta-Llama-3-8B
# Additional command used during rebuttal. Different segment lengths.
echo "python 2-textgen.py --save results/$model_prefix-$method-$watermark_key_length-comment --model $model --method $method --watermark_key_length $watermark_key_length --T $number_of_experiments --seed $seed --batch_size $batch_size --tokens_count 1300 --substitution_blocks_start 1,150,650 --substitution_blocks_end 50,400,950" >> 2-textgen-commands.sh

# Additional command used during rebuttal. Experiment on rewriting attacks.
# 500 tokens with 4 change points
# 1-100: watermark
# 101-200: substitute
# 201-300: watermark, 100: insert
# 301-400: watermark
echo "python 2-textgen.py --save results/$model_prefix-$method-$watermark_key_length-rewrite --model $model --method $method --watermark_key_length $watermark_key_length --T $number_of_experiments --seed $seed --batch_size $batch_size --gpt_rewrite_key '' --tokens_count 400 --substitution_blocks_start 100 --substitution_blocks_end 200 --insertion_blocks_start 300 --insertion_blocks_length 100" >> 2-textgen-commands.sh
