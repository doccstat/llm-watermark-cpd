import torch


def generate(
        model, prompts, vocab_size, watermark_key_length, tokens_count, seeds,
        key_func, sampler, random_offset=True
):
    batch_size = len(prompts)

    generator = torch.Generator()
    xis, pis = [], []
    for seed in seeds:
        generator.manual_seed(int(seed))
        xi, pi = key_func(generator, watermark_key_length, vocab_size)
        xis.append(xi.unsqueeze(0))
        pis.append(pi.unsqueeze(0))
    xis = torch.vstack(xis)
    pis = torch.vstack(pis)

    # deliberately not controlling this randomness with the generator
    if random_offset:
        offset = torch.randint(watermark_key_length, size=(batch_size,))
    else:
        offset = torch.zeros(size=(batch_size,), dtype=torch.int64)
    inputs = prompts.to(model.device)
    attn = torch.ones_like(inputs)
    past = None
    for i in range(tokens_count):
        with torch.no_grad():
            if past:
                output = model(
                    inputs[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(output.logits[:, -1], dim=-1).cpu()
        tokens = sampler(probs, pis, xis[torch.arange(
            batch_size), (offset.squeeze()+i) % watermark_key_length]).to(model.device)
        inputs = torch.cat([inputs, tokens], dim=-1)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return inputs.detach().cpu()


def generate_rnd(model, prompts, tokens_count):
    inputs = prompts.to(model.device)
    attn = torch.ones_like(inputs)
    past = None
    for i in range(tokens_count):
        with torch.no_grad():
            if past:
                output = model(
                    inputs[:, -1:], past_key_values=past, attention_mask=attn)
            else:
                output = model(inputs)

        probs = torch.nn.functional.softmax(output.logits[:, -1], dim=-1)

        tokens = torch.multinomial(probs, 1)
        inputs = torch.cat([inputs, tokens], dim=1)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

    return inputs.detach().cpu()
