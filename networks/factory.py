import torch

from .tiny_lm import TinyLanguageModel
from .tokenizer import ByteTokenizer


def load_model(model_name=None, dim=48, device='cuda', lora=False):
    if torch.device(device).type != 'cuda':
        raise ValueError('Preference-model execution requires cuda or cuda:N.')
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
