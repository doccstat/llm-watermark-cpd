#!/bin/bash

mkdir -p seedbs-not-textgen

export PYTHONPATH=".":$PYTHONPATH

for method in gumbel transform; do
  python textgen.py --save seedbs-not-textgen/opt-watermark200-$method.p --n 256 --batch_size 25 --m 500 --model facebook/opt-1.3b --seed 1 --T 500 --k 20 --method $method
  python textgen.py --save seedbs-not-textgen/opt-watermark100-nowatermark100-$method.p --n 256 --batch_size 25 --m 250 --model facebook/opt-1.3b --seed 1 --T 500 --k 20 --method $method --insertion_blocks_start 250 --insertion_blocks_length 250
  python textgen.py --save seedbs-not-textgen/opt-watermark80-nowatermark60-watermark60-$method.p --n 256 --batch_size 25 --m 500 --model facebook/opt-1.3b --seed 1 --T 500 --k 20 --method $method --substitution_blocks_start 200 --substitution_blocks_end 300
  python textgen.py --save seedbs-not-textgen/opt-watermark40-nowatermark50-watermark60-nowatermark20-watermark30-$method.p --n 256 --batch_size 25 --m 400 --model facebook/opt-1.3b --seed 1 --T 500 --k 20 --method $method --substitution_blocks_start 100 --substitution_blocks_end 200 --insertion_blocks_start 300 --insertion_blocks_length 100
done

for method in gumbel transform; do
  python textgen.py --save seedbs-not-textgen/gpt-watermark200-$method.p --n 256 --batch_size 25 --m 500 --model openai-community/gpt2 --seed 1 --T 500 --k 20 --method $method
  python textgen.py --save seedbs-not-textgen/gpt-watermark100-nowatermark100-$method.p --n 256 --batch_size 25 --m 250 --model openai-community/gpt2 --seed 1 --T 500 --k 20 --method $method --insertion_blocks_start 250 --insertion_blocks_length 250
  python textgen.py --save seedbs-not-textgen/gpt-watermark80-nowatermark60-watermark60-$method.p --n 256 --batch_size 25 --m 500 --model openai-community/gpt2 --seed 1 --T 500 --k 20 --method $method --substitution_blocks_start 200 --substitution_blocks_end 300
  python textgen.py --save seedbs-not-textgen/gpt-watermark40-nowatermark50-watermark60-nowatermark20-watermark30-$method.p --n 256 --batch_size 25 --m 400 --model openai-community/gpt2 --seed 1 --T 500 --k 20 --method $method --substitution_blocks_start 100 --substitution_blocks_end 200 --insertion_blocks_start 300 --insertion_blocks_length 100
done
