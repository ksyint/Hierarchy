import json
import math
import random
from pathlib import Path

import torch
import yaml

from benchmarks import LevelBenchmark
from methods import HARMLearner, SFTLearner
from networks import load_model
from .checkpoint import restore, save_experiment
from .catalog import recipes, resolve_recipe, validate_recipe


def run_experiment(args):
    if args.list_recipes:
        print('\n'.join(recipes()))
        return
    if args.recipe and args.config:
        raise ValueError('Select --recipe or --config, not both.')
    config_path = resolve_recipe(args.recipe) if args.recipe else (args.config or f'configs/{args.dataset}/harm.yaml')
    config = validate_recipe(yaml.safe_load(Path(config_path).read_text()))
    config['pretrained'] = dict(model_name=args.model, local_dir=args.local_model, cache_dir=args.cache_dir,
                                offline=args.offline, attention=args.attention,
                                gradient_checkpointing=args.gradient_checkpointing, lora=args.lora)
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.accumulation is not None:
        config['gradient_accumulation'] = args.accumulation
    if min(config['batch_size'], config.get('gradient_accumulation', 1)) < 1:
        raise ValueError('Batch size and accumulation must be positive.')
    if args.stage == 'sft':
        config['lr'] = config.get('sft_lr', 2e-5)
        config['epochs'] = config.get('sft_epochs', 50)
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.dry_run:
        print(json.dumps({'config_path': str(config_path), 'strategy': args.stage, 'config': config,
                          'data': args.data, 'validation': args.validation, 'model': args.model,
                          'checkpoint': args.checkpoint, 'device': args.device}, indent=2))
        return
    if not args.data or not args.validation:
        raise ValueError('Supply prompt-disjoint --data and --validation preference JSONL files.')
    seed = args.seed if args.seed is not None else config['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(args.threads)
    epochs = config['epochs']
    benchmark = LevelBenchmark.from_paths(args.data, args.validation, epochs)
    if config['steps_per_epoch'] is None:
        examples = sum(len(rows) for rows in benchmark.levels.values())
        config['steps_per_epoch'] = math.ceil(examples / (config['batch_size'] * config.get('gradient_accumulation', 1)))
    if args.checkpoint:
        model, tokenizer, saved_config = restore(args.checkpoint, args.device, trainable=True,
                                                 local_dir=args.local_model, cache_dir=args.cache_dir,
                                                 offline=args.offline)
        config['pretrained'] = saved_config['pretrained']
        config['pretrained'].update(gradient_checkpointing=args.gradient_checkpointing)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.enable_input_require_grads()
    else:
        model, tokenizer = load_model(device=args.device, **config['pretrained'])
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
