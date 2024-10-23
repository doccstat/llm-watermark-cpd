#!/bin/bash

rm -f 4-seedbs-commands.sh

for template_index in $(seq 1 8); do
  for prompt_index in $(seq 0 99); do
    for seeded_interval_index in $(seq 1 47); do
      echo "Rscript 4-seedbs.R $template_index $prompt_index $seeded_interval_index" >> 4-seedbs-commands.sh
    done
  done
done
