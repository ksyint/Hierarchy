import torch

from .batching import prepare_batch


def sequence_log_probs(logits, input_ids, completion_mask):
    """Sum only response-token log likelihood, with the autoregressive shift."""
    if logits.shape[:2] != input_ids.shape or completion_mask.shape != input_ids.shape:
        raise ValueError('Logits, token IDs and completion masks have incompatible shapes.')
    token_logp = logits[:, :-1].log_softmax(-1).gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    return (token_logp * completion_mask[:, 1:]).sum(-1)


def response_log_probs(model, tokenizer, prompts, responses, device, max_length=2048):
    ids, attention, mask = prepare_batch(tokenizer, prompts, responses, device, max_length)
    output = model(input_ids=ids, attention_mask=attention)
    return sequence_log_probs(output.logits, ids, mask)


def decision_logits(model, tokenizer, prompts, decision_tokens, device, max_length=2048):
    token_ids = [tokenizer.encode(token, add_special_tokens=False) for token in decision_tokens]
    if any(len(ids) != 1 for ids in token_ids):
        raise ValueError('Each decision verbalizer must encode to exactly one token.')
    probe_prompts = [prompt + '\nRefuse? ' for prompt in prompts]
    ids, attention, _ = prepare_batch(tokenizer, probe_prompts, [''] * len(prompts), device, max_length)
    # The appended EOS is a placeholder, not part of the probe input. The logits
    # immediately before it predict the one-token decision.
    positions = attention.sum(-1) - 2
    logits = model(input_ids=ids, attention_mask=attention).logits
    return logits[torch.arange(len(prompts), device=device), positions][:, [ids[0] for ids in token_ids]]
