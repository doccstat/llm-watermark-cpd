import torch
import numpy as np


def sliding_permutation_test(
    tokens, vocab_size, watermark_key_length, rolling_window_size,
    permutation_count, seed, rolling_window_index, test_stats, max_seed=100000
):
    pvalues = np.full((1, len(test_stats)), np.nan)
    if (rolling_window_index < rolling_window_size // 2 or
            rolling_window_index >= len(tokens) - rolling_window_size // 2):
        return pvalues
    rolling_window_start = rolling_window_index - rolling_window_size // 2
    rolling_window_end = rolling_window_index + rolling_window_size // 2 + 1
    pvalues[0, :] = permutation_test(
        tokens[rolling_window_start:rolling_window_end],
        vocab_size,
        watermark_key_length,
        rolling_window_size,
        permutation_count,
        seed,
        test_stats,
        max_seed
    )
    return pvalues


def permutation_test(
    tokens, vocab_size, watermark_key_length, rolling_window_size,
    permutation_count, seed, test_stats, max_seed
):
    generator = torch.Generator()

    test_results = []

    for test_stat in test_stats:
        generator.manual_seed(int(seed))
        test_result = test_stat(tokens,
                                watermark_key_length,
                                rolling_window_size,
                                generator,
                                vocab_size)
        test_results.append(test_result)

    test_results = np.array(test_results)
    p_val = 0
    null_results = []
    for run in range(permutation_count):
        null_results.append([])

        seed = torch.randint(high=max_seed, size=(1,)).item()
        for test_stat in test_stats:
            generator.manual_seed(int(seed))
            null_result = test_stat(tokens,
                                    watermark_key_length,
                                    rolling_window_size,
                                    generator,
                                    vocab_size,
                                    null=True)
            null_results[-1].append(null_result)
        # assuming lower test values indicate presence of watermark
        p_val += (null_result <= test_result).float()
    null_results = np.array(null_results)

    numerator = np.sum(null_results <= test_results, axis=0) + 1.0
    denominator = permutation_count + 1.0

    return numerator / denominator


def phi(
        tokens, watermark_key_length, rolling_window_size, generator,
        vocab_size, key_func, dist, null=False, normalize=False
):
    if null:
        tokens = torch.unique(torch.asarray(
            tokens), return_inverse=True, sorted=False)[1]
        eff_vocab_size = torch.max(tokens) + 1
    else:
        eff_vocab_size = vocab_size

    xi, pi = key_func(generator, watermark_key_length,
                      vocab_size, eff_vocab_size)
    tokens = torch.argsort(pi)[tokens]
    if normalize:
        tokens = tokens.float() / vocab_size

    scanning_statistics = adjacency(tokens, xi, dist, rolling_window_size)
    closest = torch.min(scanning_statistics, axis=1)[0]

    return torch.min(closest)


def adjacency(tokens, xi, dist, rolling_window_size):
    tokens_count = len(tokens)
    watermark_key_length = len(xi)

    scanning_statistics = torch.empty(
        size=(tokens_count-rolling_window_size+1, watermark_key_length)
    )
    for i in range(tokens_count-rolling_window_size+1):
        for j in range(watermark_key_length):
            scanning_statistics[i][j] = dist(
                tokens[i:i+rolling_window_size],
                xi[
                    (j+torch.arange(rolling_window_size)) % watermark_key_length
                ]
            )

    return scanning_statistics
