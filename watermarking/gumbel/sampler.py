from torch import argmax, gather


def gumbel_sampling(probs, pi, xi):
    return argmax(xi ** (1/gather(probs, 1, pi)), axis=1).unsqueeze(-1)
