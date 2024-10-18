from argparse import ArgumentParser

from csv import reader
from numpy import genfromtxt, savetxt
from time import perf_counter
from os.path import exists
from sys import exit

from watermarking.detection import sliding_permutation_test, phi
from watermarking.gumbel.key import gumbel_key_func
from watermarking.gumbel.score import gumbel_score, gumbel_edit_score
from watermarking.transform.key import transform_key_func
from watermarking.transform.score import transform_score, transform_edit_score

parser = ArgumentParser(description="Experiment Settings")

parser.add_argument('--token_file', default="", type=str)
parser.add_argument('--model', default="facebook/opt-1.3b", type=str)
parser.add_argument('--method', default="transform", type=str)
parser.add_argument('--watermark_key_length', default=256, type=int)
parser.add_argument('--rolling_window_size', default=0, type=int)
parser.add_argument('--permutation_count', default=999, type=int)

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--Tindex', default=1, type=int)
parser.add_argument('--rolling_window_index', default=-1, type=int)

parser.add_argument('--gamma', default=0.4, type=float)

args = parser.parse_args()

if args.model == "facebook/opt-1.3b":
    vocab_size = 50272
elif args.model == "openai-community/gpt2":
    vocab_size = 50257
elif args.model == "meta-llama/Meta-Llama-3-8B":
    vocab_size = 128256
else:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model)
    print(model.get_output_embeddings().weight.shape[0])
    raise

seeds = genfromtxt(args.token_file + '-seeds.csv', delimiter=',', max_rows=1)

watermarked_samples = genfromtxt(
    args.token_file + '-attacked-tokens.csv', delimiter=",")
Tindex = min(args.Tindex, watermarked_samples.shape[0])

if args.method == "transform":
    test_stats = []
    def dist1(x, y): return transform_edit_score(x, y, gamma=args.gamma)

    def test_stat1(
        tokens, watermark_key_length, rolling_window_size,
        generator, vocab_size, null=False
    ):
        return phi(
            tokens, watermark_key_length, rolling_window_size, generator,
            vocab_size, transform_key_func, dist1, null=False, normalize=True
        )
    test_stats.append(test_stat1)
    def dist2(x, y): return transform_score(x, y)

    def test_stat2(
        tokens, watermark_key_length, rolling_window_size,
        generator, vocab_size, null=False
    ):
        return phi(
            tokens, watermark_key_length, rolling_window_size, generator,
            vocab_size, transform_key_func, dist2, null=False, normalize=True
        )
    test_stats.append(test_stat2)

elif args.method == "gumbel":
    test_stats = []
    def dist1(x, y): return gumbel_edit_score(x, y, gamma=args.gamma)

    def test_stat1(
        tokens, watermark_key_length, rolling_window_size,
        generator, vocab_size, null=False
    ):
        return phi(
            tokens, watermark_key_length, rolling_window_size, generator,
            vocab_size, gumbel_key_func, dist1, null=null, normalize=False
        )
    test_stats.append(test_stat1)
    def dist2(x, y): return gumbel_score(x, y)

    def test_stat2(
        tokens, watermark_key_length, rolling_window_size,
        generator, vocab_size, null=False
    ):
        return phi(
            tokens, watermark_key_length, rolling_window_size, generator,
            vocab_size, gumbel_key_func, dist2, null=null, normalize=False
        )
    test_stats.append(test_stat2)
else:
    raise

# Don't forget to remove the folder following the helper file,
# if the experiment needs re-running.
if exists(
        args.token_file + '-' + str(args.rolling_window_size) + '-' +
        str(args.permutation_count) + '-detect/' + str(args.Tindex) + '-' +
        str(args.rolling_window_index) + '.csv'):
    with open(
            args.token_file + '-' + str(args.rolling_window_size) + '-' +
            str(args.permutation_count) + '-detect/' + str(args.Tindex) + '-' +
            str(args.rolling_window_index) + '.csv', 'r') as f:
        first_row = next(reader(f), None)
        if first_row is not None and len(first_row) == len(test_stats):
            exit()


def test(tokens, seed, test_stats):
    return sliding_permutation_test(tokens,
                                    vocab_size,
                                    args.watermark_key_length,
                                    args.rolling_window_size,
                                    args.permutation_count,
                                    seed,
                                    args.rolling_window_index,
                                    test_stats)


start_time = perf_counter()
pval = test(watermarked_samples[Tindex, :], seeds[Tindex], test_stats)
end_time = perf_counter()
savetxt(
    args.token_file + '-' + str(args.rolling_window_size) + '-' +
    str(args.permutation_count) + '-detect/' + str(args.Tindex) + '-' +
    str(args.rolling_window_index) + '.csv', pval, delimiter=','
)
