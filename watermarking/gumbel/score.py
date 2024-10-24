import torch

from watermarking.gumbel.gumbel_levenshtein import gumbel_levenshtein


def gumbel_score(tokens, xi):
    xi_samp = torch.gather(xi, -1, tokens.unsqueeze(-1)).squeeze()
    return -torch.sum(torch.log(1/(1-xi_samp)))


def gumbel_edit_score(tokens, xi, gamma):
    return gumbel_levenshtein(tokens.numpy(), xi.numpy(), gamma)
