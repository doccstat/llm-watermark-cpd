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

from watermarking.gumbel.sampler import gumbel_sampling
from watermarking.gumbel.key import gumbel_key_func

from watermarking.kirchenbauer.watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

import argparse

import csv

results = defaultdict(dict)

parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--method', default="transform", type=str)

parser.add_argument('--model', default="facebook/opt-1.3b", type=str)
parser.add_argument('--save', default="", type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--batch_size', default=1, type=int)

parser.add_argument('--tokens_count', default=80, type=int)
parser.add_argument('--k', default=0, type=int)
parser.add_argument('--watermark_key_length', default=256, type=int)
parser.add_argument('--T', default=500, type=int)

parser.add_argument('--prompt_tokens', default=50, type=int)
parser.add_argument('--buffer_tokens', default=20, type=int)
parser.add_argument('--n_runs', default=5000, type=int)
parser.add_argument('--max_seed', default=100000, type=int)
parser.add_argument('--offset', action='store_true')

parser.add_argument('--gamma', default=0.4, type=float)
parser.add_argument('--nowatermark', action='store_true')

parser.add_argument('--cpts', default=0, type=int)
parser.add_argument('--npseed', default=0, type=int)
parser.add_argument('--meaningful', action='store_true')

# comma separated values
parser.add_argument('--substitution_blocks_start', default="0", type=str)
parser.add_argument('--substitution_blocks_end', default="0", type=str)
parser.add_argument('--insertion_blocks_start', default="0", type=str)
parser.add_argument('--insertion_blocks_length', default="0", type=str)

parser.add_argument('--gpt_rewrite_key', default='', type=str)

parser.add_argument('--kirch_gamma', default=0.25, type=float)
parser.add_argument('--kirch_delta', default=1.0, type=float)

parser.add_argument('--rt_translate', action='store_true')
parser.add_argument('--language', default="french", type=str)

parser.add_argument('--truncate_vocab', default=8, type=int)

args = parser.parse_args()

np.random.seed(args.npseed)

if args.cpts > 0:
    cpts = []
    while len(cpts) < args.cpts:
        newcpt = np.random.randint(20, 479)
        while any([abs(newcpt - cpt) < 20 for cpt in cpts]):
            newcpt = np.random.randint(20, 479)
        cpts.append(newcpt)
    cpts = sorted(cpts)
    # odd indices are the start of the block, even indices are the end of the block
    # if args.cpts is odd, the last block is from the last odd index to the end of the text
    args.substitution_blocks_start = ','.join(
        [str(cpt) for cpt in cpts[::2]])
    args.substitution_blocks_end = ','.join([str(cpt) for cpt in cpts[1::2]])
    if len(cpts) % 2 == 1:
        args.substitution_blocks_end += ',' + str(args.tokens_count)

    cpts_save = open(args.save + '-cpts.csv', "w")
    cpts_writer = csv.writer(cpts_save, delimiter=",")
    cpts_writer.writerow(cpts)
    cpts_save.close()

results['args'] = copy.deepcopy(args)

log_file = open('log/textgen.log', 'w')
log_file.write(str(args) + '\n')
log_file.flush()

# fix the random seed for reproducibility
t0 = time()
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        "/scratch/user/anthony.li/models/" + args.model + "/tokenizer")
    model = AutoModelForCausalLM.from_pretrained(
        "/scratch/user/anthony.li/models/" + args.model + "/model",
        device_map='auto'
    )

    log_file.write(f'Loaded the local model\n')
except:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    log_file.write(f'Loaded the model\n')

log_file.flush()

vocab_size = model.get_output_embeddings().weight.shape[0]
eff_vocab_size = vocab_size - args.truncate_vocab
log_file.write(f'Loaded the model (t = {time()-t0} seconds)\n')
log_file.flush()

try:
    dataset = load_from_disk(
        '/scratch/user/anthony.li/datasets/allenai/c4/realnewslike/train'
    )
except:
    dataset = load_dataset("allenai/c4", "realnewslike",
                           split="train", streaming=True)


def corrupt(tokens):
    tokens = substitution_block_attack(tokens, list(map(int, args.substitution_blocks_start.split(
        ','))), list(map(int, args.substitution_blocks_end.split(','))), eff_vocab_size)
    tokens = insertion_block_attack(tokens, list(map(int, args.insertion_blocks_start.split(
        ','))), list(map(int, args.insertion_blocks_length.split(','))), eff_vocab_size)

    return tokens


T = args.T                  # number of prompts/generations
n_batches = int(np.ceil(T / args.batch_size))  # number of batches
prompt_tokens = args.prompt_tokens      # minimum prompt length
new_tokens = args.tokens_count
buffer_tokens = args.buffer_tokens
if args.k == 0:
    k = args.tokens_count  # k is the block size (= number of tokens)
else:
    k = args.k
n = args.watermark_key_length

if args.rt_translate:
    if args.language == "french":
        en_ne_model_name = "Helsinki-NLP/opus-mt-tc-big-en-fr"
        en_ne_tokenizer = MarianTokenizer.from_pretrained(en_ne_model_name)
        en_ne_model = MarianMTModel.from_pretrained(
            en_ne_model_name).to(device)

        ne_en_model_name = "Helsinki-NLP/opus-mt-tc-big-fr-en"
        ne_en_tokenizer = MarianTokenizer.from_pretrained(ne_en_model_name)
        ne_en_model = MarianMTModel.from_pretrained(
            ne_en_model_name).to(device)
    elif args.language == "russian":
        en_ne_model_name = "Helsinki-NLP/opus-mt-en-ru"
        en_ne_tokenizer = MarianTokenizer.from_pretrained(en_ne_model_name)
        en_ne_model = MarianMTModel.from_pretrained(
            en_ne_model_name).to(device)

        ne_en_model_name = "Helsinki-NLP/opus-mt-ru-en"
        ne_en_tokenizer = MarianTokenizer.from_pretrained(ne_en_model_name)
        ne_en_model = MarianMTModel.from_pretrained(
            ne_en_model_name).to(device)
    else:
        raise

    def rt_translate(text):
        try:
            tokens = en_ne_tokenizer(text.split(
                '. '), return_tensors="pt", padding=True).to(device)
            tokens = en_ne_model.generate(**tokens, max_new_tokens=52)
            french_text = ' '.join([en_ne_tokenizer.decode(
                t, skip_special_tokens=True) for t in tokens])

            tokens = ne_en_tokenizer(french_text.split(
                '. '), return_tensors="pt", padding=True).to(device)
            tokens = ne_en_model.generate(**tokens, max_new_tokens=512)
            roundtrip_text = ' '.join([ne_en_tokenizer.decode(
                t, skip_special_tokens=True) for t in tokens])
        except:
            roundtrip_text = ""
        return roundtrip_text

# this is the "key" for the watermark
# for now each generation gets its own key
seeds = torch.randint(2**32, (T,))
seeds_save = open(args.save + '-seeds.csv', 'w')
seeds_writer = csv.writer(seeds_save, delimiter=",")
seeds_writer.writerow(np.asarray(seeds.numpy()))
seeds_save.close()

if args.method == "gumbel":
    def generate_watermark(prompt, seed, nt):
        return generate(
            model,
            prompt,
            vocab_size,
            n,
            nt+buffer_tokens,
            seed,
            gumbel_key_func,
            gumbel_sampling,
            random_offset=args.offset
        )

elif args.method == "kirchenbauer":
    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                   gamma=args.kirch_gamma,
                                                   delta=args.kirch_delta,
                                                   seeding_scheme="simple_1")

    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                           gamma=args.kirch_gamma,  # should match original setting
                                           seeding_scheme="simple_1",  # should match original setting
                                           device=model.device,  # must match the original rng device type
                                           tokenizer=tokenizer,
                                           z_threshold=1.5,
                                           normalizers=[],
                                           ignore_repeated_bigrams=False)

    def generate_watermark(prompt, seed=None, nt = None): return model.generate(
        prompt.to(model.device),
        do_sample=True,
        max_new_tokens=nt+buffer_tokens,
        min_new_tokens=nt+buffer_tokens,
        top_k=0,
        logits_processor=LogitsProcessorList([watermark_processor])).cpu()
else:
    raise

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

    tokens = tokenizer.encode(
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

t1 = time()
pbar = tqdm(total=n_batches)
for batch in range(n_batches):
    idx = torch.arange(batch * args.batch_size,
                       min(T, (batch + 1) * args.batch_size))

    if not args.meaningful:
        null_samples.append(generate_rnd(model,
            prompts[idx], new_tokens+buffer_tokens)[:, prompt_tokens:])
        watermarked_samples = [
            generate_watermark(
                prompts[idx], seeds[idx], new_tokens)[:, prompt_tokens:]
        ]
    else:
        null_samples.append(generate_rnd(model,
            prompts[idx], 100+buffer_tokens))
        watermarked_samples = generate_watermark(
            null_samples[-1][idx], seeds[idx], 100)
        watermarked_samples = generate_rnd(model,
            watermarked_samples[idx], 100+buffer_tokens)
        watermarked_samples = generate_watermark(
            watermarked_samples[idx], seeds[idx], 100)
        watermarked_samples = [
            generate_rnd(model,
                watermarked_samples[idx], 100+buffer_tokens)[:, prompt_tokens:]
        ]

    pbar.update(1)
    log_file.write(f'Generated batch {batch} in (t = {time()-t1} seconds)\n')
    log_file.flush()
    t1 = time()
pbar.close()
null_samples = torch.vstack(null_samples)
watermarked_samples = torch.vstack(watermarked_samples)

results['watermark']['tokens'] = copy.deepcopy(watermarked_samples)
results['null']['tokens'] = copy.deepcopy(null_samples)

null_samples = torch.clip(null_samples, max=eff_vocab_size-1)
watermarked_samples = torch.clip(watermarked_samples, max=eff_vocab_size-1)

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
    if not args.meaningful:
        watermarked_sample = corrupt(watermarked_sample)
    watermarked_sample = tokenizer.decode(
        watermarked_sample, skip_special_tokens=True)
    if args.rt_translate:
        watermarked_sample = rt_translate(watermarked_sample)
        # watermarked_sample = "The biggest companies, which they already deem too profitable. Local electronics retailers complain that the deal gives Best Buy an unfair advantage. And citizen groups such as Recycle, Act Orange, and Green the Charlottesville Region have stated that they oppose the deal because it does little to promote capital expenditure to build the city's economy. Joy Phillips, a local businessman who fought so hard that he earned a column in the local newspaper all about him, summarized his opinion on the deal in a recent letter to the Daily Progress: 'Best Buy? We don't need another music store at Pantops, a music store at Pantops!' Phillips claimed, without a shred of evidence, that Best Buy was only coming to Charlottesville due to its proximity to the University of Virginia. He also said that Best Buy originally explored the site for an electronic recycling center but decided to turn it into a Best Buy store after a teenager from Richmond was convicted of dumping computers in the city's gutters. 'We are going bankrupt in the Fashion Square Mall parking lot,' he said during an interview. 'And all this for a company that would apparently be a Canadian energy waster,' he said, referring to the company's majority shareholder based in Calgary. Best Buy spokesperson, Todd Mitchell, admitted that the location of the two stores was 'a hurdle' to bringing Best Buy to Charlottesville. But he said the excellent location of the city, customers with money to spend, and the willingness of the Charlottesville Mall owner, Katon Square, to allow a temporary lease without obligation to pay rent made the site competitive. 'Congestion has no correlation with shopping, it's great to have these signs in place to channel traffic to the site,' he said. 'We couldn't ask for a better situation.' Mitchell said he considered the bankruptcy of the parking lot mainly as the result of a poor economy that has left many 'small retailers and vacations in stores,' rather than as a sign of very serious frustration among local consumers. 'This does not mean they don't have money to spend.' The pair of American Best Buy stores in Charlottesville represents an unprecedented intrusion into the Canadian market (for Best Buy at least). The Henry Schein Dental store sells to dentists, while HP does not have a non-Canadian online sales outlet. The Duke store is not considered an internet competitor for the nearest Costco."
    log_file.write(
        f'Attacked the sample {itm} with text: {watermarked_sample}\n'
    )
    log_file.flush()
    if args.gpt_rewrite_key:
        watermarked_sample = gpt_rewrite(
            watermarked_sample, args.gpt_rewrite_key
        )
    log_file.write(
        f'Attacked the sample {itm} with text rewrite: {watermarked_sample}\n'
    )
    log_file.flush()
    watermarked_sample = tokenizer.encode(watermarked_sample,
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
    if args.method == "transform":
        generator = torch.Generator()
        generator.manual_seed(int(seeds[itm]))
        pi = torch.randperm(vocab_size, generator=generator)
        pi_writer.writerow(np.asarray(pi.squeeze().numpy()))

    pbar.update(1)

pbar.close()
log_file.write(f'Attacked the samples in (t = {time()-t1} seconds)\n')
log_file.flush()
log_file.close()
attacked_tokens_save.close()

pickle.dump(results, open(args.save, "wb"))
