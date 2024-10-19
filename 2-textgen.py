from numpy import ceil

from argparse import ArgumentParser
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
)
from transformers import MarianMTModel, MarianTokenizer

from watermarking.kirchenbauer.watermark_processor import (
    WatermarkLogitsProcessor, WatermarkDetector
)

from csv import writer
from datasets import load_dataset, load_from_disk
from numpy import asarray

from watermarking.attacks import (
    insertion_block_attack, substitution_block_attack, gpt_rewrite
)
from watermarking.generation import generate
from watermarking.gumbel.key import gumbel_key_func
from watermarking.gumbel.sampler import gumbel_sampling
from watermarking.transform.key import transform_key_func
from watermarking.transform.sampler import transform_sampling

import torch

parser = ArgumentParser(description="Experiment Settings")

parser.add_argument('--save', default="", type=str)
parser.add_argument('--model', default="facebook/opt-1.3b", type=str)
parser.add_argument('--method', default="transform", type=str)
parser.add_argument('--watermark_key_length', default=256, type=int)
parser.add_argument('--number_of_experiments', default=500, type=int)

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--gpt_rewrite_key', default='', type=str)

parser.add_argument('--prompt_tokens', default=50, type=int)
parser.add_argument('--buffer_tokens', default=20, type=int)
parser.add_argument('--tokens_count', default=80, type=int)
parser.add_argument('--substitution_blocks_start', default="0", type=str)
parser.add_argument('--substitution_blocks_end', default="0", type=str)
parser.add_argument('--insertion_blocks_start', default="0", type=str)
parser.add_argument('--insertion_blocks_length', default="0", type=str)
parser.add_argument('--rt_translate', action='store_true')
parser.add_argument('--language', default="french", type=str)
parser.add_argument('--truncate_vocab', default=8, type=int)
parser.add_argument('--offset', action='store_true')
parser.add_argument('--kirch_gamma', default=0.25, type=float)
parser.add_argument('--kirch_delta', default=1.0, type=float)

args = parser.parse_args()

# fix the random seed for reproducibility
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(
    "/scratch/user/anthony.li/models/" + args.model + "/tokenizer")
model = AutoModelForCausalLM.from_pretrained(
    "/scratch/user/anthony.li/models/" + args.model + "/model",
    device_map='auto'
)

vocab_size = model.get_output_embeddings().weight.shape[0]
eff_vocab_size = vocab_size - args.truncate_vocab

try:
    dataset = load_from_disk(
        '/scratch/user/anthony.li/datasets/allenai/c4/realnewslike/train'
    )
except:
    dataset = load_dataset("allenai/c4", "realnewslike",
                           split="train", streaming=True)


def corrupt(tokens):
    tokens = substitution_block_attack(
        tokens,
        list(map(int, args.substitution_blocks_start.split(','))),
        list(map(int, args.substitution_blocks_end.split(','))),
        eff_vocab_size
    )
    tokens = insertion_block_attack(
        tokens,
        list(map(int, args.insertion_blocks_start.split(','))),
        list(map(int, args.insertion_blocks_length.split(','))),
        eff_vocab_size
    )

    return tokens


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
seeds = torch.randint(2**32, (args.number_of_experiments,))
seeds_save = open(args.save + '-seeds.csv', 'w')
seeds_writer = writer(seeds_save, delimiter=",")
seeds_writer.writerow(asarray(seeds.squeeze().numpy()))
seeds_save.close()

if args.method == "transform":
    def generate_watermark(prompt, seed):
        return generate(
            model,
            prompt,
            vocab_size,
            args.watermark_key_length,
            args.tokens_count+args.buffer_tokens,
            seed,
            transform_key_func,
            transform_sampling,
            random_offset=args.offset
        )

elif args.method == "gumbel":
    def generate_watermark(prompt, seed):
        return generate(
            model,
            prompt,
            vocab_size,
            args.watermark_key_length,
            args.tokens_count+args.buffer_tokens,
            seed,
            gumbel_key_func,
            gumbel_sampling,
            random_offset=args.offset
        )

elif args.method == "kirchenbauer":
    watermark_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.kirch_gamma,
        delta=args.kirch_delta,
        seeding_scheme="simple_1")

    watermark_detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.kirch_gamma,  # should match original setting
        seeding_scheme="simple_1",  # should match original setting
        device=model.device,  # must match the original rng device type
        tokenizer=tokenizer,
        z_threshold=1.5,
        normalizers=[],
        ignore_repeated_bigrams=False)

    def generate_watermark(prompt, seed=None): return model.generate(
        prompt.to(model.device),
        do_sample=True,
        max_new_tokens=args.tokens_count+args.buffer_tokens,
        min_new_tokens=args.tokens_count+args.buffer_tokens,
        top_k=0,
        logits_processor=LogitsProcessorList([watermark_processor])).cpu()
else:
    raise

ds_iterator = iter(dataset)

# Iterate through the dataset to get the prompts
prompt_save = open(args.save + '-prompt.csv', 'w')
prompt_writer = writer(prompt_save, delimiter=",")
prompts = []
itm = 0
pbar = tqdm(total=args.number_of_experiments)
while itm < args.number_of_experiments:
    example = next(ds_iterator)
    text = example['text']

    tokens = tokenizer.encode(
        text,
        return_tensors='pt',
        truncation=True,
        max_length=2048-args.buffer_tokens
    )[0]
    if len(tokens) < args.prompt_tokens + args.tokens_count:
        continue
    prompt = tokens[-(args.tokens_count+args.prompt_tokens):-args.tokens_count]
    prompts.append(prompt)
    prompt_writer.writerow(asarray(prompt.numpy()))

    itm += 1
    pbar.update(1)
pbar.close()
prompt_save.close()
prompts = torch.vstack(prompts)

watermarked_samples = []

batch_count = int(ceil(args.number_of_experiments / args.batch_size))
pbar = tqdm(total=batch_count)
for batch in range(batch_count):
    idx = torch.arange(
        batch * args.batch_size,
        min(args.number_of_experiments, (batch + 1) * args.batch_size))

    watermarked_samples.append(generate_watermark(
        prompts[idx], seeds[idx])[:, args.prompt_tokens:])

    pbar.update(1)
pbar.close()
watermarked_samples = torch.vstack(watermarked_samples)
watermarked_samples = torch.clip(watermarked_samples, max=eff_vocab_size-1)

# Save the text/tokens before attack and NTP for each token in the watermark
# texts with true and empty prompt.
tokens_before_attack_save = open(args.save + '-tokens-before-attack.csv', "w")
tokens_before_attack_writer = writer(
    tokens_before_attack_save, delimiter=",")
pbar = tqdm(total=len(watermarked_samples))
for tokens in watermarked_samples:
    tokens_before_attack_writer.writerow(asarray(tokens.numpy()))
    pbar.update(1)
pbar.close()
tokens_before_attack_save.close()

# Attack the watermarked texts and store a copy appended with the
# prompt-extracting prompt in `icl_samples`.
attacked_tokens_save = open(
    args.save + "-attacked-tokens.csv", "w")
attacked_tokens_writer = writer(attacked_tokens_save, delimiter=",")
pi_save = None
pi_writer = None
if args.method == "transform":
    pi_save = open(args.save + "-pi.csv", "w")
    pi_writer = writer(pi_save, delimiter=",")

pbar = tqdm(total=args.number_of_experiments)
for itm in range(args.number_of_experiments):
    watermarked_sample = watermarked_samples[itm]
    watermarked_sample = corrupt(watermarked_sample)
    watermarked_sample = tokenizer.decode(
        watermarked_sample, skip_special_tokens=True)
    if args.rt_translate:
        watermarked_sample = rt_translate(watermarked_sample)
    if args.gpt_rewrite_key:
        watermarked_sample = gpt_rewrite(
            watermarked_sample, args.gpt_rewrite_key
        )
    watermarked_sample = tokenizer.encode(watermarked_sample,
                                          return_tensors='pt',
                                          truncation=True,
                                          max_length=2048)[0]
    if len(watermarked_sample) < args.tokens_count + 1:
        watermarked_sample = torch.nn.functional.pad(
            watermarked_sample, (args.tokens_count-len(watermarked_sample), 0),
            "constant", 0
        )
    else:
        watermarked_sample = watermarked_sample[1:args.tokens_count+1+sum(
            list(map(int, args.insertion_blocks_length.split(','))))]
    attacked_tokens_writer.writerow(asarray(watermarked_sample.numpy()))
    if args.method == "transform":
        generator = torch.Generator()
        generator.manual_seed(int(seeds[itm]))
        pi = torch.randperm(vocab_size, generator=generator)
        pi_writer.writerow(asarray(pi.squeeze().numpy()))

    pbar.update(1)

pbar.close()
attacked_tokens_save.close()
