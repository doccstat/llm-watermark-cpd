import time
import torch
from datasets import load_dataset

from collections import defaultdict
import copy

import numpy as np
from numpy import genfromtxt

from watermarking.transform.score import transform_score, transform_edit_score
from watermarking.transform.score import its_score, itsl_score
from watermarking.transform.key import transform_key_func

from watermarking.gumbel.score import gumbel_score, gumbel_edit_score
from watermarking.gumbel.score import ems_score, emsl_score
from watermarking.gumbel.key import gumbel_key_func

import argparse

import csv

results = defaultdict(dict)

parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--method', default="transform", type=str)

parser.add_argument('--model', default="facebook/opt-1.3b", type=str)
parser.add_argument('--token_file', default="", type=str)
parser.add_argument('--seed', default=0, type=int)

parser.add_argument('--k', default=0, type=int)
parser.add_argument('--n', default=256, type=int)
parser.add_argument('--Tindex', default=1, type=int)

parser.add_argument('--prompt_tokens', default=50, type=int)
parser.add_argument('--buffer_tokens', default=20, type=int)
parser.add_argument('--n_runs', default=5000, type=int)
parser.add_argument('--max_seed', default=100000, type=int)
parser.add_argument('--offset', action='store_true')

parser.add_argument('--norm', default=1, type=int)
parser.add_argument('--gamma', default=0.4, type=float)
# parser.add_argument('--edit', action='store_true')
parser.add_argument('--nowatermark', action='store_true')

parser.add_argument('--deletion', default=0.0, type=float)
parser.add_argument('--insertion', default=0.0, type=float)
parser.add_argument('--substitution', default=0.0, type=float)

parser.add_argument('--kirch_gamma', default=0.25, type=float)
parser.add_argument('--kirch_delta', default=1.0, type=float)

parser.add_argument('--truncate_vocab', default=8, type=int)

args = parser.parse_args()
results['args'] = copy.deepcopy(args)

t0 = time.time()

if args.model == "facebook/opt-1.3b":
    vocab_size = 50272
elif args.model == "openai-community/gpt2":
    vocab_size = 50257
else:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model).to(
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    print(model.get_output_embeddings().weight.shape[0])
    raise
eff_vocab_size = vocab_size - args.truncate_vocab
print(f'Loaded the model (t = {time.time()-t0} seconds)')

while True:
    try:
        dataset = load_dataset("allenai/c4", "realnewslike",
                               split="train", streaming=True)
        break
    except:
        time.sleep(3)

prompt_tokens = args.prompt_tokens      # minimum prompt length
buffer_tokens = args.buffer_tokens
k = args.k
n = args.n     # watermark key length

seeds = np.genfromtxt(args.token_file + '-seeds.txt',
                      delimiter=',', max_rows=1)

################################################################################


def sliding_permutation_test(
    tokens, vocab_size, n, k, seed, test_stats, n_runs=100, max_seed=100000
):
    pvalues = np.full((len(test_stats), len(tokens)), np.nan)
    for i in range(k // 2, len(tokens) - k // 2):
        pvalues[:, i] = permutation_test(
            tokens[(i - k // 2):(i + k // 2 + 1)
                   ], vocab_size, n, k, seed, test_stats, n_runs, max_seed
        )
    return pvalues


def permutation_test(
    tokens, vocab_size, n, k, seed, test_stats, n_runs=100, max_seed=100000
):
    generator = torch.Generator()

    test_results = []

    for test_stat in test_stats:
        generator.manual_seed(int(seed))
        test_result = test_stat(tokens=tokens,
                                n=n,
                                k=k,
                                generator=generator,
                                vocab_size=vocab_size)
        test_results.append(test_result)

    test_results = np.array(test_results)
    p_val = 0
    null_results = []
    for run in range(n_runs):
        null_results.append([])

        seed = torch.randint(high=max_seed, size=(1,)).item()
        for test_stat in test_stats:
            generator.manual_seed(int(seed))
            null_result = test_stat(tokens=tokens,
                                    n=n,
                                    k=k,
                                    generator=generator,
                                    vocab_size=vocab_size,
                                    null=True)
            null_results[-1].append(null_result)
        # assuming lower test values indicate presence of watermark
        p_val += (null_result <= test_result).float()
    null_results = np.array(null_results)

    return (np.sum(null_results <= test_results, axis=0) + 1.0) / (n_runs + 1.0)


def phi(
        tokens, n, k, generator, key_func, vocab_size, dist,
        null=False, normalize=False
):
    if null:
        tokens = torch.unique(torch.asarray(
            tokens), return_inverse=True, sorted=False)[1]
        eff_vocab_size = torch.max(tokens) + 1
    else:
        eff_vocab_size = vocab_size

    xi, pi = key_func(generator, n, vocab_size, eff_vocab_size)
    tokens = torch.argsort(pi)[tokens]
    if normalize:
        tokens = tokens.float() / vocab_size

    A = adjacency(tokens, xi, dist, k)
    closest = torch.min(A, axis=1)[0]

    return torch.min(closest)


def adjacency(tokens, xi, dist, k):
    m = len(tokens)
    n = len(xi)

    A = torch.empty(size=(m-(k-1), n))
    for i in range(m-(k-1)):
        for j in range(n):
            A[i][j] = dist(tokens[i:i+k], xi[(j+torch.arange(k)) % n])

    return A

################################################################################


if args.method == "transform":
    test_stats = []
    def dist1(x, y): return transform_edit_score(x, y, gamma=args.gamma)

    def test_stat1(tokens, n, k, generator, vocab_size, null=False): return phi(
        tokens=tokens,
        n=n,
        k=k,
        generator=generator,
        key_func=transform_key_func,
        vocab_size=vocab_size,
        dist=dist1,
        null=False,
        normalize=True
    )
    test_stats.append(test_stat1)
    def dist2(x, y): return transform_score(x, y)

    def test_stat2(tokens, n, k, generator, vocab_size, null=False): return phi(
        tokens=tokens,
        n=n,
        k=k,
        generator=generator,
        key_func=transform_key_func,
        vocab_size=vocab_size,
        dist=dist2,
        null=False,
        normalize=True
    )
    test_stats.append(test_stat2)
    def dist3(x, y): return its_score(x, y, vocab_size=vocab_size)

    def test_stat3(tokens, n, k, generator, vocab_size, null=False): return phi(
        tokens=tokens,
        n=n,
        k=k,
        generator=generator,
        key_func=transform_key_func,
        vocab_size=vocab_size,
        dist=dist3,
        null=False,
        normalize=True
    )
    test_stats.append(test_stat3)

    def dist4(x, y): return itsl_score(
        x, y, vocab_size=vocab_size, gamma=args.gamma)

    def test_stat4(tokens, n, k, generator, vocab_size, null=False): return phi(
        tokens=tokens,
        n=n,
        k=k,
        generator=generator,
        key_func=transform_key_func,
        vocab_size=vocab_size,
        dist=dist4,
        null=False,
        normalize=True
    )
    test_stats.append(test_stat4)


elif args.method == "gumbel":
    test_stats = []
    def dist1(x, y): return gumbel_edit_score(x, y, gamma=args.gamma)

    def test_stat1(tokens, n, k, generator, vocab_size, null=False): return phi(
        tokens=tokens,
        n=n,
        k=k,
        generator=generator,
        key_func=gumbel_key_func,
        vocab_size=vocab_size,
        dist=dist1,
        null=null,
        normalize=False
    )
    test_stats.append(test_stat1)
    def dist2(x, y): return gumbel_score(x, y)

    def test_stat2(tokens, n, k, generator, vocab_size, null=False): return phi(
        tokens=tokens,
        n=n,
        k=k,
        generator=generator,
        key_func=gumbel_key_func,
        vocab_size=vocab_size,
        dist=dist2,
        null=null,
        normalize=False
    )
    test_stats.append(test_stat2)
    def dist3(x, y): return ems_score(x, y)

    def test_stat3(tokens, n, k, generator, vocab_size, null=False): return phi(
        tokens=tokens,
        n=n,
        k=k,
        generator=generator,
        key_func=gumbel_key_func,
        vocab_size=vocab_size,
        dist=dist3,
        null=null,
        normalize=False
    )
    test_stats.append(test_stat3)
    def dist4(x, y): return emsl_score(x, y, gamma=args.gamma)

    def test_stat4(tokens, n, k, generator, vocab_size, null=False): return phi(
        tokens=tokens,
        n=n,
        k=k,
        generator=generator,
        key_func=gumbel_key_func,
        vocab_size=vocab_size,
        dist=dist4,
        null=null,
        normalize=False
    )
    test_stats.append(test_stat4)
else:
    raise

ds_iterator = iter(dataset)


def test(tokens, seed): return sliding_permutation_test(tokens,
                                                        vocab_size,
                                                        n,
                                                        k,
                                                        seed,
                                                        test_stats)


t1 = time.time()

watermarked_samples = genfromtxt(
    args.token_file + '-attacked-tokens.csv', delimiter=",")
Tindex = min(args.Tindex, watermarked_samples.shape[0])

csv_saves = []
csvWriters = []
if args.method == "transform":
    csv_saves.append(open(args.token_file + '/' +
                     str(args.Tindex) + '-transform-edit.csv', 'w'))
    csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    csv_saves.append(open(args.token_file + '/' +
                     str(args.Tindex) + '-transform.csv', 'w'))
    csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    csv_saves.append(open(args.token_file + '/' +
                     str(args.Tindex) + '-its.csv', 'w'))
    csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    csv_saves.append(open(args.token_file + '/' +
                          str(args.Tindex) + '-itsl.csv', 'w'))
    csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
elif args.method == "gumbel":
    csv_saves.append(open(args.token_file + '/' +
                     str(args.Tindex) + '-gumbel-edit.csv', 'w'))
    csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    csv_saves.append(open(args.token_file + '/' +
                     str(args.Tindex) + '-gumbel.csv', 'w'))
    csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    csv_saves.append(open(args.token_file + '/' +
                     str(args.Tindex) + '-ems.csv', 'w'))
    csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
    csv_saves.append(open(args.token_file + '/' +
                     str(args.Tindex) + '-emsl.csv', 'w'))
    csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
elif args.method == "kirchenbauer":
    csv_saves.append(open(args.token_file + '/' +
                     str(args.Tindex) + '-kirchenbauer.csv', 'w'))
    csvWriters.append(csv.writer(csv_saves[-1], delimiter=','))
else:
    raise

watermarked_sample = watermarked_samples[Tindex, :]
pval = test(watermarked_sample, seeds[Tindex])
for distance_index in range(len(test_stats)):
    csvWriters[distance_index].writerow(np.asarray(pval[distance_index, ]))
    csv_saves[distance_index].flush()

print(args.token_file + '/' + str(args.Tindex) + ' done')
print(f'Ran the experiment (t = {time.time()-t1} seconds)')

for csv_save in csv_saves:
    csv_save.close()
