import torch

from collections import defaultdict
import copy

from transformers import AutoTokenizer, AutoModelForCausalLM

from numpy import genfromtxt

import argparse

results = defaultdict(dict)

parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--model', default="facebook/opt-1.3b", type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--token_file', default="", type=str)

# comma separated list of detected change points
parser.add_argument('--detected_cpts', default='', type=str)

args = parser.parse_args()
results['args'] = copy.deepcopy(args)

log_file = open(
    'log/' + args.token_file.split('results/')[1].split('.p')[0] + '-demo.log',
    'w'
)
log_file.write(str(args) + '\n')
log_file.flush()

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

prompt_tokens = genfromtxt(
    args.token_file + '-prompt.csv', delimiter=",")
prompt_text = tokenizer.decode(
        prompt_tokens, skip_special_tokens=True)

tokens_before_attack = genfromtxt(
    args.token_file + '-tokens-before-attack.csv', delimiter=",")
text_before_attack = tokenizer.decode(
    tokens_before_attack, skip_special_tokens=True)

attacked_tokens = genfromtxt(
    args.token_file + '-attacked-tokens.csv', delimiter=",")
attacked_text = tokenizer.decode(
    attacked_tokens, skip_special_tokens=True)

true_cpts = genfromtxt(
    args.token_file + '-cpts.csv', delimiter=",")
true_cpts = list(map(int, true_cpts))
true_cpts.append(len(attacked_tokens))
detected_cpts = list(map(int, args.detected_cpts.split(',')))
detected_cpts.append(len(attacked_tokens))

log_file.write(f'Prompt: {prompt_text}\n')
log_file.write(f'Text before attack: {text_before_attack}\n')
log_file.write(f'Attacked text: {attacked_text}\n')

# split the text_before_attack into parts by the true change points
for i in range(len(true_cpts) - 1):
    start = true_cpts[i]
    end = true_cpts[i + 1]
    tokens = tokens_before_attack[start:end]
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    log_file.write(f'Text before attack true change point {i}: {text}\n')

# split the text_before_attack into parts by the detected change points
for i in range(len(detected_cpts) - 1):
    start = detected_cpts[i]
    end = detected_cpts[i + 1]
    tokens = tokens_before_attack[start:end]
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    log_file.write(f'Text before attack detected change point {i}: {text}\n')

# split the attacked_text into parts by the true change points
for i in range(len(true_cpts) - 1):
    start = true_cpts[i]
    end = true_cpts[i + 1]
    tokens = attacked_tokens[start:end]
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    log_file.write(f'Attacked text true change point {i}: {text}\n')

# split the attacked_text into parts by the detected change points
for i in range(len(detected_cpts) - 1):
    start = detected_cpts[i]
    end = detected_cpts[i + 1]
    tokens = attacked_tokens[start:end]
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    log_file.write(f'Attacked text detected change point {i}: {text}\n')

log_file.close()
