import torch
import scipy
import numpy as np


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
        tokens = torch.unique(tokens, return_inverse=True, sorted=False)[1]
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
