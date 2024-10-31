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

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::transformers
conda install cython scipy nltk sentencepiece sacremoses
```

## Instructions

All experiments are conducted using Slurm workload manager. Expected running
time and memory usage are provided in the corresponding sbatch scripts.

> [!IMPORTANT]
> Please modify the paths, Slurm mail options and adjust the GPU resources in
> the sbatch scripts before running the experiments.

> [!CAUTION]
> The Python SeedBS script is modified based on the R version. The output is not
> guaranteed to be the same.

```shell
# Setup pyx.
sbatch 1-setup.sh

# Text generation.
bash 2-textgen-helper.sh
sbatch 2-textgen.sh

# Rolling window watermark detection.
bash 3-detect-helper.sh
sbatch 3-detect.sh

# Change point analysis.
bash 4-seedbs-helper.sh
sbatch 4-seedbs.sh
# OR
bash 4.1-seedbs-helper.sh
sbatch 4.1-seedbs.sh

# Result analysis and ploting. (NOT)
Rscript 5-not.R
```

> [!TIP]
> The implementation of NOT can be found in the [5-not.R](./5-not.R) script from
> [line 348 to 371](https://github.com/doccstat/llm-watermark-cpd/blob/main/5-not.R#L348-L371).

## Citation

```bibtex
@misc{li2024segmentingwatermarkedtextslanguage,
  title={Segmenting Watermarked Texts From Language Models},
  author={Xingchi Li and Guanxun Li and Xianyang Zhang},
  year={2024},
  eprint={2410.20670},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2410.20670},
}
```
