from torch import int64

from torch import Generator, no_grad

from torch import arange, cat, multinomial, ones_like, randint, vstack, zeros
from torch.nn.functional import softmax


def generate(
        model, prompts, vocab_size, watermark_key_length, tokens_count, seeds,
        key_func, sampler, random_offset=True
):
    batch_size = len(prompts)

    generator = Generator()
    xis, pis = [], []
    for seed in seeds:
        generator.manual_seed(int(seed))
        xi, pi = key_func(generator, watermark_key_length, vocab_size)
        xis.append(xi.unsqueeze(0))
        pis.append(pi.unsqueeze(0))
    xis = vstack(xis)
    pis = vstack(pis)

    # deliberately not controlling this randomness with the generator
    if random_offset:
        offset = randint(watermark_key_length, size=(batch_size,))
    else:
        offset = zeros(size=(batch_size,), dtype=int64)
    inputs = prompts.to(model.device)
    attn = ones_like(inputs)
    past = None
    for i in range(tokens_count):
        with no_grad():
            if past:
                output = model(
                    inputs[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = softmax(output.logits[:, -1], dim=-1).cpu()
        tokens = sampler(probs, pis, xis[arange(
            batch_size), (offset.squeeze()+i) % watermark_key_length]).to(model.device)
        inputs = cat([inputs, tokens], dim=-1)

        past = output.past_key_values
        attn = cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return inputs.detach().cpu()


def generate_rnd(model, prompts, tokens_count):
    inputs = prompts.to(model.device)
    attn = ones_like(inputs)
    past = None
    for i in range(tokens_count):
        with no_grad():
            if past:
                output = model(
                    inputs[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = softmax(output.logits[:, -1], dim=-1)

        tokens = multinomial(probs, 1)
        inputs = cat([inputs, tokens], dim=1)

        past = output.past_key_values
        attn = cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return inputs.detach().cpu()
