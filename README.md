# llm-watermark-cpd

## Prerequisites

<details closed>
<summary>Python environments</summary>

-   Cython==3.0.10
-   datasets==2.19.1
-   huggingface_hub==0.23.0
-   nltk==3.8.1
-   numpy==1.26.4
-   sacremoses==0.0.53
-   scipy==1.13.0
-   sentencepiece==0.2.0
-   tokenizers==0.19.1
-   torch==2.3.0.post100
-   torchaudio==2.3.0
-   torchvision==0.18.0
-   tqdm==4.66.4
-   transformers==4.40.2

</details>

### Set up environments

#### Python

> [!NOTE]
> Refer to https://pytorch.org for PyTorch installation on other platforms

```shell
# conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::transformers
conda install cython scipy nltk sentencepiece sacremoses
```

#### R

> [!NOTE]
> R is used for change point detection. Refer to https://www.r-project.org for
> installation instructions.

```r
install.packages(c("doParallel", "reshape2", "ggplot2", "fossil"))
```

## Instruction

To reproduce the results, follow the instructions below or use the attached
results directly using `Rscript analyze.R 1 3200`.

### Set up pyx

```shell
python setup.py build_ext --inplace
```

#### Expected running time

Less than 1 minute on a single core CPU machine.

#### Expected memory usage

Less than 1 GB.

### Generate watermarked tokens

```shell
sbatch textgen.sh
```

#### Expected running time

Less than 4 hours on 1 compute node with 1 NVIDIA A30 GPU and 128 CPU cores.

#### Expected memory usage

Less than 128 GB.

### Calculate p-values for sliding windows

```shell
rm -f detect-commands.sh
for method in gumbel transform; do
  for cpts in 0 1 2 4 9 19; do
    mkdir -p results/ml3-${cpts}changepoints-$method.p-detect
    for Tindex in $(seq 0 9); do
      for k in 20; do
        for fixed_i in $(seq 0 499); do
          echo "bash ./detect-helper.sh $method $Tindex $cpts $k $fixed_i" >> detect-commands.sh
        done
      done
    done
  done
done

sbatch detect.sh
```

#### Expected running time

Less than 24 hours on 8 compute nodes with no GPU and 28 CPU cores each.

#### Expected memory usage

Less than 10 GB per compute node.

### Change point analysis

```shell
rm -f seedbs-commands.sh
for template_index in $(seq 1 24); do
  for prompt_index in $(seq 0 9); do
    for seeded_interval_index in $(seq 1 47); do
      echo "Rscript seedbs.R $template_index $prompt_index $seeded_interval_index" >> seedbs-commands.sh
    done
  done
done

sbatch seedbs.sh
```

```shell
parallel -j 8 --progress Rscript analyze.R {1} {2} ::: $(seq 1 400 2801) ::: $(seq 400 400 3200)
Rscript analyze.R 1 3200
```

#### Expected running time

Less than 12 hours on 8 compute nodes with no GPU and 28 CPU cores each.

##### Running time test

The following command should run in less than 10 minutes on 1 compute node
with no GPU and 28 CPU cores.

```shell
Rscript analyze.R 1 5
```

#### Expected memory usage

Less than 10 GB per compute node.

### Ablation study

```shell
rm -f ablation-commands.sh
for method in gumbel; do
  for cpts in 4; do
    mkdir -p results/ml3-${cpts}changepoints-$method.p-detect
    for Tindex in $(seq 0 4); do
      # Loop to handle the k values and their corresponding n_runs
      for k in 10 20 30 40 50; do
        if [ $k -eq 20 ]; then
          # Specific n_runs values for k = 20
          n_runs_array=(99 249 499 749 999)
        else
          # Default n_runs value for other k values
          n_runs_array=(999)
        fi
        # Loop over each n_runs in the array
        for n_runs in "${n_runs_array[@]}"; do
          for fixed_i in $(seq 0 499); do
            echo "bash ./ablation-helper.sh $method $Tindex $cpts $k $fixed_i $n_runs" >> ablation-commands.sh
          done
        done
      done
    done
  done
done

sbatch ablation.sh

rm -f ablation-seedbs-commands.sh
for template_index in $(seq 1 9); do
  for prompt_index in $(seq 0 4); do
    for seeded_interval_index in $(seq 1 47); do
      echo "Rscript ablation-seedbs.R $template_index $prompt_index $seeded_interval_index" >> ablation-seedbs-commands.sh
    done
  done
done

sbatch ablation-seedbs.sh
```

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
mkdir -p results/ml3-mllm-1-gumbel.p-detect
mkdir -p results/ml3-mllm-2-gumbel.p-detect
mkdir -p results/ml3-mllm-3-gumbel.p-detect
mkdir -p results/ml3-mllm-4-gumbel.p-detect
mkdir -p results/ml3-mllm-5-gumbel.p-detect

rm -f mllm-seedbs-commands.sh
for template_index in 1; do
  for prompt_index in 0 1 2 3 4; do
    for seeded_interval_index in $(seq 1 29); do
      for llm in gpt ml3; do
        echo "Rscript mllm-seedbs.R $template_index $prompt_index $seeded_interval_index $llm" >> mllm-seedbs-commands.sh
      done
    done
  done
done
```
