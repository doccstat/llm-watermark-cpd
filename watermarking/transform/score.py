import sys
import numpy as np
from watermarking.transform.transform_levenshtein import transform_levenshtein
from watermarking.transform.its_levenshtein import its_levenshtein

import torch


def transform_score(tokens, xi):
    return torch.pow(torch.linalg.norm(tokens-xi.squeeze(), ord=1), 1)


def transform_edit_score(tokens, xi, gamma=1):
    return transform_levenshtein(tokens.numpy(), xi.squeeze().numpy(), gamma)


def its_score(tokens, xi, vocab_size):
    return -torch.inner(xi.squeeze() - 0.5, (tokens - 1) / (vocab_size - 1) - 0.5) / len(tokens)


def itsl_score(tokens, xi, vocab_size, gamma=1):
    return its_levenshtein(tokens.numpy(), xi.squeeze().numpy(), gamma, vocab_size)
