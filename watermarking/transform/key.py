from torch import rand, randperm


def transform_key_func(generator, n, vocab_size, eff_vocab_size=None):
    pi = randperm(vocab_size, generator=generator)
    xi = rand((n, 1), generator=generator)

    return xi, pi
