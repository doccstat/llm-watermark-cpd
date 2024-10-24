from torch.linalg import norm

from torch import pow as torch_pow

from watermarking.transform.transform_levenshtein import transform_levenshtein


def transform_score(tokens, xi):
    return torch_pow(norm(tokens-xi.squeeze(), ord=1), 1)


def transform_edit_score(tokens, xi, gamma=1):
    return transform_levenshtein(tokens.numpy(), xi.squeeze().numpy(), gamma)
