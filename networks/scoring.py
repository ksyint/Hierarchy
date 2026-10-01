import torch

from .batching import prepare_batch


def sequence_log_probs(logits, input_ids, completion_mask):
    """Sum only response-token log likelihood, with the autoregressive shift."""
    if logits.shape[:2] != input_ids.shape or completion_mask.shape != input_ids.shape:
        raise ValueError('Logits, token IDs and completion masks have incompatible shapes.')
    scores = []
    for start in range(0, input_ids.shape[1] - 1, 128):
        stop = min(start + 128, input_ids.shape[1] - 1)
        logp = logits[:, start:stop].float().log_softmax(-1)
        selected = logp.gather(-1, input_ids[:, start + 1:stop + 1, None]).squeeze(-1)
        scores.append((selected * completion_mask[:, start + 1:stop + 1]).sum(-1))
    return torch.stack(scores).sum(0)


def response_log_probs(model, tokenizer, prompts, responses, device, max_length=2048):
    ids, attention, mask = prepare_batch(tokenizer, prompts, responses, device, max_length)
    output = model(input_ids=ids, attention_mask=attention)
    return sequence_log_probs(output.logits, ids, mask)


def decision_logits(model, tokenizer, prompts, decision_tokens, device, max_length=2048):
    probe_prompts = [prompt + '\nRefuse? ' for prompt in prompts]
    values = []
    for verbalizer in decision_tokens:
        ids, attention, mask = prepare_batch(tokenizer, probe_prompts, [verbalizer] * len(prompts), device, max_length)
        mask[torch.arange(len(prompts), device=device), attention.sum(-1) - 1] = 0
        values.append(sequence_log_probs(model(input_ids=ids, attention_mask=attention).logits, ids, mask))
    return torch.stack(values, dim=-1)
