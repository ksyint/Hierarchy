import json
from pathlib import Path

import torch

from networks.factory import load_model


def save_experiment(strategy, args, history):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    strategy.model.save_pretrained(output / 'pretrained', safe_serialization=True)
    strategy.tokenizer.save_pretrained(output / 'pretrained')
    torch.save({'format': 2, 'config': strategy.config, 'pretrained': 'pretrained',
                'curriculum': strategy.curriculum.state_dict()}, output / 'last.pt')
    (output / 'metrics.json').write_text(json.dumps(history, indent=2) + '\n')


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
