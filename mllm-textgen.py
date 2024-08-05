from time import time

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
from transformers import MarianMTModel, MarianTokenizer

from datasets import load_dataset, load_from_disk

from tqdm import tqdm
from collections import defaultdict
import pickle
import copy

import numpy as np

from watermarking.generation import generate, generate_rnd
from watermarking.attacks import insertion_block_attack, substitution_block_attack, gpt_rewrite

from watermarking.transform.sampler import transform_sampling
from watermarking.transform.key import transform_key_func

from watermarking.gumbel.sampler import gumbel_sampling
from watermarking.gumbel.key import gumbel_key_func

from watermarking.kirchenbauer.watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

import argparse

import csv

results = defaultdict(dict)

parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--method', default="transform", type=str)

parser.add_argument('--model1', default="facebook/opt-1.3b", type=str)
parser.add_argument('--model2', default="facebook/opt-1.3b", type=str)
parser.add_argument('--save', default="", type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--batch_size', default=1, type=int)

parser.add_argument('--tokens_count', default=80, type=int)
parser.add_argument('--k', default=0, type=int)
parser.add_argument('--watermark_key_length', default=256, type=int)

parser.add_argument('--prompt_tokens', default=50, type=int)
parser.add_argument('--buffer_tokens', default=20, type=int)
parser.add_argument('--n_runs', default=5000, type=int)
parser.add_argument('--max_seed', default=100000, type=int)
parser.add_argument('--offset', action='store_true')

parser.add_argument('--gamma', default=0.4, type=float)
parser.add_argument('--nowatermark', action='store_true')

# comma separated values
parser.add_argument('--substitution_blocks_start', default="0", type=str)
parser.add_argument('--substitution_blocks_end', default="0", type=str)
parser.add_argument('--insertion_blocks_start', default="0", type=str)
parser.add_argument('--insertion_blocks_length', default="0", type=str)

parser.add_argument('--kirch_gamma', default=0.25, type=float)
parser.add_argument('--kirch_delta', default=1.0, type=float)

parser.add_argument('--truncate_vocab', default=8, type=int)

args = parser.parse_args()

results['args'] = copy.deepcopy(args)

log_file = open('log/textgen.log', 'w')
log_file.write(str(args) + '\n')
log_file.flush()

# fix the random seed for reproducibility
t0 = time()
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer1 = AutoTokenizer.from_pretrained(
        "/scratch/user/anthony.li/models/" + args.model1 + "/tokenizer")
    model1 = AutoModelForCausalLM.from_pretrained(
        "/scratch/user/anthony.li/models/" + args.model1 + "/model",
        device_map='auto'
    )

    log_file.write(f'Loaded the local model\n')
except:
    tokenizer1 = AutoTokenizer.from_pretrained(args.model1)
    model1 = AutoModelForCausalLM.from_pretrained(args.model1).to(device)
    log_file.write(f'Loaded the model\n')

try:
    tokenizer2 = AutoTokenizer.from_pretrained(
        "/scratch/user/anthony.li/models/" + args.model2 + "/tokenizer")
    model2 = AutoModelForCausalLM.from_pretrained(
        "/scratch/user/anthony.li/models/" + args.model2 + "/model",
        device_map='auto'
    )

    log_file.write(f'Loaded the local model\n')
except:
    tokenizer2 = AutoTokenizer.from_pretrained(args.model2)
    model2 = AutoModelForCausalLM.from_pretrained(args.model2).to(device)
    log_file.write(f'Loaded the model\n')

log_file.flush()

vocab_size1 = model1.get_output_embeddings().weight.shape[0]
eff_vocab_size1 = vocab_size1 - args.truncate_vocab

vocab_size2 = model2.get_output_embeddings().weight.shape[0]
eff_vocab_size2 = vocab_size2 - args.truncate_vocab

log_file.write(f'Loaded the model (t = {time()-t0} seconds)\n')
log_file.flush()

try:
    dataset = load_from_disk(
        '/scratch/user/anthony.li/datasets/allenai/c4/realnewslike/train'
    )
except:
    dataset = load_dataset("allenai/c4", "realnewslike",
                           split="train", streaming=True)

T = 1                  # number of prompts/generations
n_batches = int(np.ceil(T / args.batch_size))  # number of batches
prompt_tokens = args.prompt_tokens      # minimum prompt length
new_tokens = args.tokens_count
buffer_tokens = args.buffer_tokens
if args.k == 0:
    k = args.tokens_count  # k is the block size (= number of tokens)
else:
    k = args.k
n = args.watermark_key_length

# this is the "key" for the watermark
# for now each generation gets its own key
seeds = torch.randint(2**32, (T,))
seeds_save = open(args.save + '-seeds.csv', 'w')
seeds_writer = csv.writer(seeds_save, delimiter=",")
# seeds_writer.writerow(np.asarray(seeds.squeeze().numpy()))
seeds_list = np.asarray(seeds.squeeze().numpy()).tolist()
if not isinstance(seeds_list, list):
    seeds_list = [seeds_list]
seeds_save.close()


def generate_watermark1(prompt, seed, nt):
    return generate(
        model1,
        prompt,
        vocab_size1,
        n,
        nt+buffer_tokens,
        seed,
        gumbel_key_func,
        gumbel_sampling,
        random_offset=args.offset
    )


def generate_watermark2(prompt, seed, nt):
    return generate(
        model2,
        prompt,
        vocab_size2,
        n,
        nt+buffer_tokens,
        seed,
        gumbel_key_func,
        gumbel_sampling,
        random_offset=args.offset
    )


ds_iterator = iter(dataset)

t1 = time()

# Iterate through the dataset to get the prompts
prompt_save = open(args.save + '-prompt.csv', 'w')
prompt_writer = csv.writer(prompt_save, delimiter=",")
prompts = []
itm = 0
pbar = tqdm(total=T)
while itm < T:
    example = next(ds_iterator)
    text = example['text']

    tokens = tokenizer1.encode(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=2048-buffer_tokens
    )[0]
    if len(tokens) < prompt_tokens + new_tokens:
        continue
    prompt = tokens[-(new_tokens+prompt_tokens):-new_tokens]
    prompts.append(prompt)
    prompt_writer.writerow(np.asarray(prompt.numpy()))

    itm += 1
    pbar.update(1)
pbar.close()
prompt_save.close()
prompts = torch.vstack(prompts)

null_samples = []
watermarked_samples = []

t1 = time()
pbar = tqdm(total=n_batches)
for batch in range(n_batches):
    idx = torch.arange(batch * args.batch_size,
                       min(T, (batch + 1) * args.batch_size))

    null_samples.append(generate_rnd(
        prompts[idx], 100+buffer_tokens, model1))
    watermarked_samples = generate_watermark1(
        null_samples[-1][idx], seeds[idx], 100)
    watermarked_samples = tokenizer1.decode(
        watermarked_samples[0, :], skip_special_tokens=True
    )
    watermarked_samples = tokenizer2.encode(watermarked_samples,
                                            return_tensors='pt',
                                            truncation=True,
                                            max_length=2048)[0]
    watermarked_samples = [generate_watermark2(
        torch.vstack([watermarked_samples]), seeds[idx], 100)[:, prompt_tokens:]]

    pbar.update(1)
    log_file.write(f'Generated batch 0 in (t = {time()-t1} seconds)\n')
    log_file.flush()
    t1 = time()
pbar.close()
null_samples = torch.vstack(null_samples)
watermarked_samples = torch.vstack(watermarked_samples)

results['watermark']['tokens'] = copy.deepcopy(watermarked_samples)
results['null']['tokens'] = copy.deepcopy(null_samples)

null_samples = torch.clip(null_samples, max=eff_vocab_size2-1)
watermarked_samples = torch.clip(watermarked_samples, max=eff_vocab_size2-1)

if args.nowatermark:
    watermarked_samples = null_samples

log_file.write(f'Generated samples in (t = {time()-t1} seconds)\n')
log_file.flush()

# Save the text/tokens before attack and NTP for each token in the watermark
# texts with true and empty prompt.
t1 = time()
tokens_before_attack_save = open(args.save + '-tokens-before-attack.csv', "w")
tokens_before_attack_writer = csv.writer(
    tokens_before_attack_save, delimiter=",")
pbar = tqdm(total=len(watermarked_samples))
for tokens in watermarked_samples:
    tokens_before_attack_writer.writerow(np.asarray(tokens.numpy()))
    pbar.update(1)
pbar.close()
tokens_before_attack_save.close()
log_file.write(
    f'Saved text/tokens before attack and probs in (t = {time()-t1} seconds)\n')
log_file.flush()

t1 = time()
null_tokens_save = open(args.save + '-null.csv', 'w')
null_tokens_writer = csv.writer(null_tokens_save, delimiter=",")
for tokens in null_samples:
    null_tokens_writer.writerow(np.asarray(tokens.numpy()))
null_tokens_save.close()
log_file.write(
    f'Saved null samples and probs in (t = {time()-t1} seconds)\n')
log_file.flush()

# Attack the watermarked texts and store a copy appended with the
# prompt-extracting prompt in `icl_samples`.
attacked_tokens_save = open(
    args.save + "-attacked-tokens.csv", "w")
attacked_tokens_writer = csv.writer(attacked_tokens_save, delimiter=",")
pi_save = None
pi_writer = None
if args.method == "transform":
    pi_save = open(args.save + "-pi.csv", "w")
    pi_writer = csv.writer(pi_save, delimiter=",")

pbar = tqdm(total=T)
for itm in range(T):
    watermarked_sample = watermarked_samples[itm]
    watermarked_sample = tokenizer2.decode(
        watermarked_sample, skip_special_tokens=True)
    log_file.write(
        f'Attacked the sample {itm} with text: {watermarked_sample}\n'
    )
    log_file.flush()
    watermarked_sample = tokenizer2.encode(watermarked_sample,
                                           return_tensors='pt',
                                           truncation=True,
                                           max_length=2048)[0]
    if len(watermarked_sample) < new_tokens + 1:
        watermarked_sample = torch.nn.functional.pad(
            watermarked_sample, (new_tokens-len(watermarked_sample), 0),
            "constant", 0
        )
    else:
        watermarked_sample = watermarked_sample[1:new_tokens+1+sum(
            list(map(int, args.insertion_blocks_length.split(','))))]
    attacked_tokens_writer.writerow(np.asarray(watermarked_sample.numpy()))

    pbar.update(1)

pbar.close()
log_file.write(f'Attacked the samples in (t = {time()-t1} seconds)\n')
log_file.flush()
log_file.close()
attacked_tokens_save.close()

pickle.dump(results, open(args.save, "wb"))
