from pathlib import Path
import re

import torch

from .pretrained import resolve_backbone


def cuda_device(device):
    result = torch.device(device)
    if result.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Language-model execution requires CUDA-enabled PyTorch and an accessible GPU.')
    return result


def load_model(model_name='qwen3-4b', dim=None, device='cuda', lora=True, *,
               local_dir=None, cache_dir='.cache/huggingface', offline=False,
               attention=None, gradient_checkpointing=False, dtype='bfloat16'):
    from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForConditionalGeneration
    device = cuda_device(device)
    spec = resolve_backbone(model_name)
    source = local_dir or spec.repo
    if local_dir and not (Path(local_dir) / 'config.json').is_file():
        raise FileNotFoundError(f'{local_dir}/config.json is required; preserve the full downloaded model directory.')
    common = dict(cache_dir=cache_dir, local_files_only=offline, trust_remote_code=spec.remote_code)
    if not local_dir and not Path(source).is_dir():
        common['revision'] = spec.revision
    precision = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[dtype]
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, **common)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_class = Gemma3ForConditionalGeneration if spec.multimodal else AutoModelForCausalLM
        model = model_class.from_pretrained(source, torch_dtype=precision, low_cpu_mem_usage=True,
                                             device_map={'': str(device)},
                                             attn_implementation=attention or spec.attention, **common)
    except OSError as error:
        raise RuntimeError(f'Cannot load {source}. Download https://huggingface.co/{spec.repo}/tree/{spec.revision} '
                           f'to checkpoints/{model_name} and use --local-model checkpoints/{model_name}; '
                           'gated models require accepted terms and hf auth login.') from error
    model.config.use_cache = False
    if lora:
        from peft import LoraConfig, get_peft_model
        leaves = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'out_proj',
                  'gate_proj', 'up_proj', 'down_proj', 'c_fc', 'c_proj'}
        targets = [name for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear) and name.rsplit('.', 1)[-1] in leaves
                   and 'vision_tower' not in name and 'multi_modal_projector' not in name]
        if spec.multimodal:
            targets.extend(name for name, module in model.named_modules()
                           if isinstance(module, torch.nn.Linear) and name.endswith('lm_head'))
        if not targets:
            raise ValueError('No attention/MLP projection was found for LoRA.')
        pattern = '(?:' + '|'.join(re.escape(name) for name in targets) + ')'
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                                                task_type='CAUSAL_LM', target_modules=pattern))
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    return model, tokenizer
