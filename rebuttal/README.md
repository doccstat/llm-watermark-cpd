# Segmenting Watermarked Texts From Language Models

Scripts used during the rebuttal phase of the submission.

## Instructions

Move the contents of this directory to the root of the repository and run the
following commands.

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
