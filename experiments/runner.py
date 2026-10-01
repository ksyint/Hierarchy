import json
import random
from pathlib import Path

import torch
import yaml

from benchmarks import LevelBenchmark
from methods import HARMLearner, SFTLearner
from networks import load_model
from .checkpoint import save_experiment
from .catalog import recipes, resolve_recipe, validate_recipe


def run_experiment(args):
    if args.list_recipes:
        print('\n'.join(recipes()))
        return
    if args.recipe and args.config:
        raise ValueError('Select --recipe or --config, not both.')
    config_path = resolve_recipe(args.recipe) if args.recipe else (args.config or f'configs/{args.dataset}/harm.yaml')
    config = validate_recipe(yaml.safe_load(Path(config_path).read_text()))
    if args.dry_run:
        print(json.dumps({'config_path': str(config_path), 'strategy': args.stage, 'config': config,
                          'data': args.data, 'validation': args.validation, 'model': args.model,
                          'checkpoint': args.checkpoint, 'device': args.device}, indent=2))
        return
    if args.recipe and not all((args.data, args.validation, args.model)):
        raise ValueError('Catalog experiments require --data, --validation, and --model.')
    seed = args.seed if args.seed is not None else config['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(args.threads)
    epochs = args.epochs or config['epochs']
    benchmark = LevelBenchmark.from_paths(args.data, args.validation, epochs)
    model, tokenizer = load_model(args.model, config['model_dim'], args.device, args.lora)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True)['model'])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config['lr'], weight_decay=config['weight_decay'])
    method = SFTLearner if args.stage == 'sft' else HARMLearner
    strategy = method(model, tokenizer, optimizer, config, args.device, epochs)
    results = []
    for experience in benchmark.train_stream:
        strategy.train(experience)
        accuracy = strategy.eval(benchmark.test_stream)
        results.append(strategy.finish_experience(experience, accuracy))
        print(json.dumps(results[-1]))
    save_experiment(strategy, args, results)
