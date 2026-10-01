"""Pinned language backbones, CUDA loading and saved policy restoration."""
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import torch


@dataclass(frozen=True)
class Backbone:
    repo: str
    revision: str
    attention: str = 'sdpa'
    remote_code: bool = False
    multimodal: bool = False


BACKBONES = {
    'qwen3-1.7b': Backbone('Qwen/Qwen3-1.7B', '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'),
    'qwen3-4b': Backbone('Qwen/Qwen3-4B', '1cfa9a7208912126459214e8b04321603b3df60c'),
    'kanana-2.1b': Backbone('kakaocorp/kanana-nano-2.1b-base', '3d1ace1214dbebf52b9d2cd6f18b78857194d72b'),
    'exaone-2.4b': Backbone('LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct', 'ccce25bd39c141fe053e0bc75818a8f5fe962802', 'eager', True),
    'gemma3-4b': Backbone('google/gemma-3-4b-it', '093f9f388b31de276ce2de164bdc2081324b9767', multimodal=True),
    'teacher-strong': Backbone('Qwen/Qwen3-32B', '9216db5781bf21249d130ec9da846c4624c16137'),
    'teacher-medium': Backbone('Qwen/Qwen3-14B', '40c069824f4251a91eefaf281ebe4c544efd3e18'),
    'teacher-weak': Backbone('Qwen/Qwen3-8B-AWQ', '4da05a8edb55c6046cce958586c33b61da07bb79'),
}


def resolve_backbone(name='qwen3-4b'):
    name = name or 'qwen3-4b'
    if name in BACKBONES:
        return BACKBONES[name]
    for spec in BACKBONES.values():
        if spec.repo == name:
            return spec
    return Backbone(name, 'main')


def download(name, cache_dir='.cache/huggingface', destination=None, offline=False):
    from huggingface_hub import snapshot_download
    spec = resolve_backbone(name)
    try:
        return snapshot_download(spec.repo, revision=spec.revision, cache_dir=cache_dir,
                                 local_dir=destination, local_files_only=offline,
                                 allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt',
                                                 '*.jinja', '*.py', 'tokenizer.*'])
    except OSError as error:
        folder = destination or f'checkpoints/{name}'
        raise RuntimeError(
            f'Get the complete checkpoint from https://huggingface.co/{spec.repo}/tree/{spec.revision} '
            f'into {folder}, then pass --local-model {folder}. '
            'For gated repositories accept the model terms and run hf auth login first.'
        ) from error


def command_download(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='qwen3-4b')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--cache-dir', default='.cache/huggingface')
    parser.add_argument('--destination')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps({key: asdict(value) for key, value in BACKBONES.items()}, indent=2))
        return
    print(download(args.model, args.cache_dir, args.destination, args.offline))


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


def restore(path, device, trainable=False, local_dir=None, cache_dir=None, offline=False):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if checkpoint.get('format') != 2:
        raise ValueError('Use a pretrained-backbone checkpoint produced by the current training entrypoint.')
    options = dict(config['pretrained'])
    options.update(local_dir=local_dir or options.get('local_dir'), offline=offline,
                   cache_dir=cache_dir or options.get('cache_dir', '.cache/huggingface'))
    adapter_dir = Path(path).resolve().parent / checkpoint['pretrained']
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)
    if options['lora']:
        from peft import PeftModel
        base, _ = load_model(device=device, **{**options, 'lora': False, 'gradient_checkpointing': False})
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=trainable)
    else:
        model, _ = load_model(device=device, **{**options, 'local_dir': str(adapter_dir), 'lora': False})
    config['pretrained'].update({key: options[key] for key in ('local_dir', 'cache_dir', 'offline')})
    model.train(trainable)
    return model, tokenizer, config
