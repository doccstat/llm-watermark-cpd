# llm-watermark-cpd

## Set up environments

```shell
conda create --name watermark
conda activate watermark

# Refer to https://pytorch.org for PyTorch installation on other platforms
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::transformers
conda install cython scipy nltk sentencepiece sacremoses
```

> [!NOTE]
> R is used for change point detection. Refer to https://www.r-project.org for
> installation instructions.

## Usage

### Set up pyx

```shell
python setup.py build_ext --inplace
```

### Generate watermarked tokens

```shell
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
```

### Calculate p-values for sliding windows

```shell
for method in gumbel transform; do
  mkdir -p seedbs-not-textgen/opt-watermark200-$method.pfacebook.opt-1.3b$method
  mkdir -p seedbs-not-textgen/opt-watermark100-nowatermark100-$method.pfacebook.opt-1.3b$method
  mkdir -p seedbs-not-textgen/opt-watermark80-nowatermark60-watermark60-$method.pfacebook.opt-1.3b$method
  mkdir -p seedbs-not-textgen/opt-watermark40-nowatermark50-watermark60-nowatermark20-watermark30-$method.pfacebook.opt-1.3b$method
done

for method in gumbel transform; do
  mkdir -p seedbs-not-textgen/gpt-watermark200-$method.popenai-community.gpt2$method
  mkdir -p seedbs-not-textgen/gpt-watermark100-nowatermark100-$method.popenai-community.gpt2$method
  mkdir -p seedbs-not-textgen/gpt-watermark80-nowatermark60-watermark60-$method.popenai-community.gpt2$method
  mkdir -p seedbs-not-textgen/gpt-watermark40-nowatermark50-watermark60-nowatermark20-watermark30-$method.popenai-community.gpt2$method
done

chmod +x ./detect.sh
parallel -j 50 --progress ./detect.sh {1} {2} {3} ::: gumbel transform ::: $(seq 1 100) ::: watermark200 watermark100-nowatermark100 watermark80-nowatermark60-watermark60 watermark40-nowatermark50-watermark60-nowatermark20-watermark30
```

### Change point analysis

```shell
Rscript analyze.R 1 3200
```
