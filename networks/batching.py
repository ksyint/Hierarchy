import torch


def prompt_ids(tokenizer, prompt):
    if getattr(tokenizer, 'chat_template', None):
        return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                             tokenize=True, add_generation_prompt=True, enable_thinking=False)
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    return [bos] + tokenizer.encode(prompt, add_special_tokens=False)


def prepare_batch(tokenizer, prompts, responses, device, max_length=2048):
    sequences, masks = [], []
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    if bos is None or tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer must define EOS, PAD, and an initial token.')
    for prompt, response in zip(prompts, responses):
        prefix = prompt_ids(tokenizer, prompt)
        suffix = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(suffix) >= max_length:
            raise ValueError('Response exceeds max_length; raise limit instead of silently truncating preference data.')
        prefix = prefix[-(max_length - len(suffix)):]
        sequences.append(prefix + suffix)
        masks.append([0] * len(prefix) + [1] * len(suffix))
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    attention = torch.zeros_like(ids)
    for i, (seq, response_mask) in enumerate(zip(sequences, masks)):
        ids[i, :len(seq)] = torch.tensor(seq, device=device)
        mask[i, :len(seq)] = torch.tensor(response_mask, device=device)
        attention[i, :len(seq)] = 1
    return ids, attention, mask
