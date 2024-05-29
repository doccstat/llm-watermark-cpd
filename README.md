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
mkdir -p results

export PYTHONPATH=".":$PYTHONPATH

for method in gumbel transform; do
  python textgen.py --save results/opt-watermark500-$method.p --n 1000 --batch_size 25 --m 500 --model facebook/opt-1.3b --seed 1 --T 500 --k 20 --method $method
  python textgen.py --save results/opt-watermark250-nowatermark250-$method.p --n 1000 --batch_size 25 --m 250 --model facebook/opt-1.3b --seed 1 --T 500 --k 20 --method $method --insertion_blocks_start 250 --insertion_blocks_length 250
  python textgen.py --save results/opt-watermark200-nowatermark100-watermark200-$method.p --n 1000 --batch_size 25 --m 500 --model facebook/opt-1.3b --seed 1 --T 500 --k 20 --method $method --substitution_blocks_start 200 --substitution_blocks_end 300
  python textgen.py --save results/opt-watermark100-nowatermark100-watermark100-nowatermark100-watermark100-$method.p --n 1000 --batch_size 25 --m 400 --model facebook/opt-1.3b --seed 1 --T 500 --k 20 --method $method --substitution_blocks_start 100 --substitution_blocks_end 200 --insertion_blocks_start 300 --insertion_blocks_length 100
done

for method in gumbel transform; do
  python textgen.py --save results/gpt-watermark500-$method.p --n 1000 --batch_size 25 --m 500 --model openai-community/gpt2 --seed 1 --T 500 --k 20 --method $method
  python textgen.py --save results/gpt-watermark250-nowatermark250-$method.p --n 1000 --batch_size 25 --m 250 --model openai-community/gpt2 --seed 1 --T 500 --k 20 --method $method --insertion_blocks_start 250 --insertion_blocks_length 250
  python textgen.py --save results/gpt-watermark200-nowatermark100-watermark200-$method.p --n 1000 --batch_size 25 --m 500 --model openai-community/gpt2 --seed 1 --T 500 --k 20 --method $method --substitution_blocks_start 200 --substitution_blocks_end 300
  python textgen.py --save results/gpt-watermark100-nowatermark100-watermark100-nowatermark100-watermark100-$method.p --n 1000 --batch_size 25 --m 400 --model openai-community/gpt2 --seed 1 --T 500 --k 20 --method $method --substitution_blocks_start 100 --substitution_blocks_end 200 --insertion_blocks_start 300 --insertion_blocks_length 100
done
```

#### Expected running time

Less than 4 hours on 1 compute node with 1 NVIDIA A30 GPU and 128 CPU cores.

#### Expected memory usage

Less than 128 GB.

### Calculate p-values for sliding windows

```shell
for method in gumbel transform; do
  mkdir -p results/opt-watermark500-$method.pfacebook.opt-1.3b$method
  mkdir -p results/opt-watermark250-nowatermark250-$method.pfacebook.opt-1.3b$method
  mkdir -p results/opt-watermark200-nowatermark100-watermark200-$method.pfacebook.opt-1.3b$method
  mkdir -p results/opt-watermark100-nowatermark100-watermark100-nowatermark100-watermark100-$method.pfacebook.opt-1.3b$method
done

for method in gumbel transform; do
  mkdir -p results/gpt-watermark500-$method.popenai-community.gpt2$method
  mkdir -p results/gpt-watermark250-nowatermark250-$method.popenai-community.gpt2$method
  mkdir -p results/gpt-watermark200-nowatermark100-watermark200-$method.popenai-community.gpt2$method
  mkdir -p results/gpt-watermark100-nowatermark100-watermark100-nowatermark100-watermark100-$method.popenai-community.gpt2$method
done

chmod +x ./detect.sh
parallel -j 50 --progress ./detect.sh {1} {2} {3} ::: gumbel transform ::: $(seq 1 100) ::: watermark500 watermark250-nowatermark250 watermark200-nowatermark100-watermark200 watermark100-nowatermark100-watermark100-nowatermark100-watermark100
```

#### Expected running time

Less than 24 hours on 8 compute nodes with no GPU and 28 CPU cores each.

#### Expected memory usage

Less than 10 GB per compute node.

### Change point analysis

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
