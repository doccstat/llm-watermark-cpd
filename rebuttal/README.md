# Segmenting Watermarked Texts From Language Models

Scripts used during the rebuttal phase of the submission.

## Instructions

Move the contents of this directory to the root of the repository and run the
following commands.

### Extra experiments

```shell
sbatch extra-textgen.sh
for cpts in 3 4 6 8 9 12; do
  for texts in 1 2 3 4 5; do
    mkdir -p results/ml3-random-${cpts}cpts-text${texts}-gumbel.p-detect
  done
done
mkdir -p results/ml3-concat-text5-gumbel.p-detect
mkdir -p results/ml3-concat-french-text5-gumbel.p-detect

sbatch extra-detect.sh

rm -f extra-seedbs-commands.sh
for template_index in $(seq 1 32); do
  for prompt_index in 0; do
    for seeded_interval_index in $(seq 1 47); do
      echo "Rscript extra-seedbs.R $template_index $prompt_index $seeded_interval_index" >> extra-seedbs-commands.sh
    done
  done
done

sbatch extra-seedbs.sh
```

### Multiple LLM experiments

```shell
rm -f mllm-seedbs-commands.sh
for template_index in 1 2 3 4 5; do
  mkdir -p results/ml3-mllm-${template_index}-gumbel.p-detect
  for prompt_index in 0; do
    for seeded_interval_index in $(seq 1 29); do
      for llm in gpt ml3; do
        echo "Rscript mllm-seedbs.R $template_index $prompt_index $seeded_interval_index $llm" >> mllm-seedbs-commands.sh
      done
    done
  done
done
```
