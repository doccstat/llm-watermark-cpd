# llm-watermark-cpd

## Set up environments

```bash
conda create --name watermark
conda activate watermark

# Refer to https://pytorch.org for PyTorch installation on other platforms
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::transformers
conda install cython scipy nltk sentencepiece sacremoses

python setup.py build_ext --inplace
```

> [!NOTE]
> R is used for change point detection. Refer to https://www.r-project.org for
> installation instructions.

## Usage

### Generate watermarked tokens

```bash
sh textgen.sh
```

### Calculate p-values for sliding windows

```bash
sh detect-run.sh
```

### Change point analysis

```bash
Rscript analyze.R 1 3200
```
