# Segmenting Watermarked Texts From Language Models

Implementation of the methods described in "Segmenting Watermarked Texts From Language Models" by [Xingchi Li](https://xingchi.li), [Guanxun Li](https://guanxun.li), [Xianyang Zhang](https://zhangxiany-tamu.github.io).

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

> [!NOTE]
> Refer to https://pytorch.org for PyTorch installation on other platforms

```shell
# conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::transformers
conda install cython scipy nltk sentencepiece sacremoses
```

## Instructions

All experiments are conducted using Slurm workload manager. Expected running
time and memory usage are provided in the corresponding sbatch scripts.

> [!NOTE]
> Please modify the paths, uncomment Slurm mail options and adjust the GPU
> resources in the sbatch scripts before running the experiments.

### Setup pyx.

```shell
sbatch 1-setup.sh
```

### Text generation.

```shell
bash 2-textgen-helper.sh
jobid=$(sbatch --parsable 2-textgen.sh)
sacct -j $jobid --format=JobID,JobName,State,ExitCode --noheader | grep textgen
```

### Rolling window watermark detection.

```shell
bash 3-detect-helper.sh
jobid=$(sbatch --parsable 3-detect.sh)
sacct -j $jobid --format=JobID,JobName,State,ExitCode --noheader | grep detect
sacct -j $jobid --format=JobID,JobName,State,ExitCode --parsable2 | awk -F'|' '
  /detect/ {
    if ($3 == "NODE_FAIL") { node_fail++ }
    if ($3 == "PENDING") { pending++ }
    if ($3 == "COMPLETED") { completed++ }
    if ($3 == "RUNNING" ) { running++ }
  }
  END {
    print "Node fail:", node_fail
    print "Pending:", pending
    print "Completed:", completed
    print "Running:", running
  }'
```

### Change point analysis

```shell
bash 4-seedbs-helper.sh
jobid=$(sbatch --parsable 4-seedbs.sh)
sacct -j $jobid --format=JobID,JobName,State,ExitCode --noheader | grep seedbs
sacct -j $jobid --format=JobID,JobName,State,ExitCode --parsable2 | awk -F'|' '
  /seedbs/ {
    if ($3 == "NODE_FAIL") { node_fail++ }
    if ($3 == "PENDING") { pending++ }
    if ($3 == "COMPLETED") { completed++ }
    if ($3 == "RUNNING" ) { running++ }
  }
  END {
    print "Node fail:", node_fail
    print "Pending:", pending
    print "Completed:", completed
    print "Running:", running
  }'
```

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
            echo "python ablation.py --token_file "results/ml3-${cpts}changepoints-${method}.p" --n 1000 --model meta-llama/Meta-Llama-3-8B --seed 1 --Tindex ${Tindex} --k ${k} --method ${method} --fixed_i ${fixed_i} --n_runs ${n_runs}" >> ablation-commands.sh
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

# Citation

TBD

```bibtex
@inproceedings{
  anonymous2024segmenting,
  title={Segmenting Watermarked Texts From Language Models},
  author={Anonymous},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=FAuFpGeLmx}
}
```
