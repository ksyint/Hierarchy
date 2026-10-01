import json
from pathlib import Path

import torch

from networks.factory import load_model


def save_experiment(strategy, args, history):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({'model': strategy.model.state_dict(), 'config': strategy.config, 'model_name': args.model,
                'lora': args.lora, 'curriculum': strategy.curriculum.state_dict()}, output / 'last.pt')
    if args.model:
        strategy.model.save_pretrained(output / 'pretrained')
        strategy.tokenizer.save_pretrained(output / 'pretrained')
    (output / 'metrics.json').write_text(json.dumps(history, indent=2) + '\n')


def restore(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model, tokenizer = load_model(checkpoint['model_name'], checkpoint['config']['model_dim'], device, checkpoint['lora'])
    model.load_state_dict(checkpoint['model'])
    return model.eval(), tokenizer, checkpoint['config']
