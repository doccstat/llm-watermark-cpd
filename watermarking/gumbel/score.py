import sys
import numpy as np
from watermarking.gumbel.gumbel_levenshtein import gumbel_levenshtein
from watermarking.gumbel.ems_levenshtein import ems_levenshtein

import torch


def gumbel_score(tokens, xi):
    xi_samp = torch.gather(xi, -1, tokens.unsqueeze(-1)).squeeze()
    return -torch.sum(torch.log(1/(1-xi_samp)))


def gumbel_edit_score(tokens, xi, gamma):
    return gumbel_levenshtein(tokens.numpy(), xi.numpy(), gamma)


def ems_score(tokens, xi):
    xi_samp = torch.gather(xi, -1, tokens.unsqueeze(-1)).squeeze()
    return -(1 + torch.mean(torch.log(xi_samp)))


def emsl_score(tokens, xi, gamma):
    return ems_levenshtein(tokens.numpy(), xi.numpy(), gamma)
