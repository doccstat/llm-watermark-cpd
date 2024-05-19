import torch


def insertion_block_attack(tokens, starts, lengths, vocab_size, distribution=None):
    if distribution is None:
        def distribution(x): return torch.ones(
            size=(len(tokens), vocab_size)) / vocab_size
    if len(starts) != len(lengths):
        raise
    insertion_blocks = []
    for length in lengths:
        new_probs = distribution(tokens)
        samples = torch.multinomial(new_probs, 1).flatten()
        insertion_blocks.append(samples[:length])
    new_tokens = tokens[:starts[0]]
    for i in range(len(starts) - 1):
        new_tokens = torch.cat(
            [new_tokens, insertion_blocks[i], tokens[starts[i]:starts[i+1]]])
    new_tokens = torch.cat(
        [new_tokens, insertion_blocks[-1], tokens[starts[-1]:]])
    return new_tokens


def substitution_block_attack(tokens, starts, ends, vocab_size, distribution=None):
    if distribution is None:
        def distribution(x): return torch.ones(
            size=(len(tokens), vocab_size)) / vocab_size

    if len(starts) != len(ends):
        raise

    for start, end in zip(starts, ends):
        new_probs = distribution(tokens)
        samples = torch.multinomial(new_probs, 1).flatten()
        tokens[start:end] = samples[:(end-start)]

    return tokens
