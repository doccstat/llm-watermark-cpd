from torch import cumsum, gather, searchsorted

def transform_sampling(probs, pi, xi):
    cdf = cumsum(gather(probs, 1, pi), 1)
    return gather(pi, 1, searchsorted(cdf, xi))
