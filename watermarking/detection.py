import torch
import time
import numpy as np


def sliding_permutation_test(
    tokens, vocab_size, n, k, seed, test_stats, log_file=None,
    n_runs=100, max_seed=100000, fixed_i=None
):
    if fixed_i is None:
        pvalues = np.full((len(test_stats), len(tokens)), np.nan)
        for i in range(k // 2, len(tokens) - k // 2):
            pvalues[:, i] = permutation_test(
                tokens[(i - k // 2):(i + k // 2 + 1)
                       ], vocab_size, n, k, seed, test_stats, n_runs, max_seed
            )
    else:
        pvalues = np.full((len(test_stats), 1), np.nan)
        if fixed_i < k // 2 or fixed_i >= len(tokens) - k // 2:
            return pvalues
        pvalues[:, 0] = permutation_test(
            tokens[(fixed_i - k // 2):(fixed_i + k // 2 + 1)
                   ], vocab_size, n, k, seed, test_stats, log_file,
                   n_runs, max_seed
        )
    return pvalues


def permutation_test(
    tokens, vocab_size, n, k, seed, test_stats, log_file=None,
    n_runs=100, max_seed=100000
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
    t0 = time.time()
    log_file.write(f'Begin {n_runs} permutation tests\n')
    log_file.flush()
    for run in range(n_runs):
        if run % 100 == 0:
            log_file.write(f'Run {run} (t = {time.time()-t0} seconds)\n')
            log_file.flush()
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
