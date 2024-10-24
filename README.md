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
sbatch 2-textgen.sh
```

### Rolling window watermark detection.

```shell
bash 3-detect-helper.sh
sbatch 3-detect.sh
```

### Change point analysis

```shell
bash 4-seedbs-helper.sh
sbatch 4-seedbs.sh
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
