from types import SimpleNamespace

import torch
from torch import nn

from utils.losses import sequence_log_probs


class ByteTokenizer:
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2

    def encode(self, text, add_special_tokens=False):
        return [b + 3 for b in text.encode('utf-8')]

    def decode(self, tokens, skip_special_tokens=True):
        return bytes([i - 3 for i in tokens if i >= 3]).decode('utf-8', errors='replace')


class TinyLanguageModel(nn.Module):
    """A real autoregressive CPU fixture, not a substitute for the paper's SLMs."""
    def __init__(self, dim=48):
        super().__init__()
        self.embed = nn.Embedding(259, dim)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.lm_head = nn.Linear(dim, 259)

    def forward(self, input_ids, attention_mask=None):
        hidden, _ = self.rnn(self.embed(input_ids))
        return SimpleNamespace(logits=self.lm_head(hidden))


def prepare_batch(tokenizer, prompts, responses, device, max_length=2048):
    sequences, masks = [], []
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    if bos is None or tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer must define EOS, PAD, and an initial token.')
    for prompt, response in zip(prompts, responses):
        prefix = [bos] + tokenizer.encode(prompt, add_special_tokens=False)
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


def load_model(model_name=None, dim=48, device='cpu', lora=False):
    if not model_name:
        return TinyLanguageModel(dim).to(device), ByteTokenizer()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    if lora:
        from peft import LoraConfig, get_peft_model
        config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type='CAUSAL_LM',
                            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
        model = get_peft_model(model, config)
    return model, tokenizer
