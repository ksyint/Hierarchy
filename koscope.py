"""Run preparation, teacher generation and Korean preference experiments."""
import argparse
import copy
import gc
from itertools import product
import json
import math
from pathlib import Path
import random

import torch
import yaml

from benchmarks.korean import LevelBenchmark
from benchmarks.korean import load_records
from methods.preference import HARMLearner, SFTLearner, restore, save_experiment
from methods.preference import cuda_device, load_model
from methods.preference import decision_logits, prompt_ids, resolve_backbone


ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / 'configs' / 'experiments'


def recipes():
    return {path.relative_to(CATALOG).with_suffix('').as_posix(): path
            for path in sorted(CATALOG.rglob('*.yaml'))}


def resolve_recipe(name):
    available = recipes()
    if name not in available:
        raise ValueError(f'Unknown recipe {name!r}. Use --list-recipes to inspect available settings.')
    return available[name]


def validate_recipe(config):
    required = {'seed', 'batch_size', 'epochs', 'steps_per_epoch', 'lr',
                'weight_decay', 'beta', 'lambda_ccr', 'max_length', 'decision_tokens', 'curriculum'}
    if not required <= config.keys():
        raise ValueError(f'Missing recipe keys: {sorted(required - config.keys())}')
    if any(config[key] <= 0 for key in ('batch_size', 'epochs', 'lr', 'beta', 'max_length')):
        raise ValueError('Model, batching, optimization, beta, and sequence length settings must be positive.')
    if config['steps_per_epoch'] is not None and config['steps_per_epoch'] < 1:
        raise ValueError('steps_per_epoch must be positive or null to cover the data pool.')
    if len(config['decision_tokens']) != 2 or not all(isinstance(token, str) and token for token in config['decision_tokens']):
        raise ValueError('Specify two nonempty decision verbalizers in [comply, refuse] order.')
    curriculum = config['curriculum']
    if len(curriculum['thresholds']) != 2 or min(curriculum['thresholds']) <= 0:
        raise ValueError('The hierarchy requires positive Level-1 and Level-2 loss thresholds.')
    if not 0 <= curriculum['probe_threshold'] <= 1 or not 0 <= curriculum['lower_replay'] < 1:
        raise ValueError('Probe threshold and replay probability are outside their valid ranges.')
    if len(curriculum['ramps']) != 3 or min(curriculum['ramps']) <= 0:
        raise ValueError('Specify a positive hard-negative ramp for each of three levels.')
    if curriculum['gamma0'] < 0 or curriculum['kappa'] < 0 or not 0 < curriculum['rho'] <= 1:
        raise ValueError('Invalid margin decay or competence EMA settings.')
    if config['lambda_ccr'] < 0 or config['weight_decay'] < 0 or curriculum['explicit_fade'] <= 0:
        raise ValueError('Regularizer weights must be nonnegative and the fade duration positive.')
    return config


GAMMA = {'g05': 0.5, 'g10': 1.0, 'g20': 2.0}
BETA = {'b005': 0.05, 'b010': 0.1, 'b020': 0.2}
GATES = {'conservative': ([0.25, 0.20], 0.90),
         'standard': ([0.35, 0.30], 0.85),
         'permissive': ([0.45, 0.40], 0.80)}
DECAY = {'k004': 0.04, 'k008': 0.08, 'k012': 0.12}
REPLAY = {'r020': 0.20, 'r030': 0.30, 'r040': 0.40}


def command_catalog(argv=None):
    parser = argparse.ArgumentParser(description='Regenerate the Korean preference recipe catalog.')
    parser.parse_args(argv)
    base = yaml.safe_load((ROOT / 'configs/korean/harm.yaml').read_text())
    count = 0
    for gamma, beta, gate, decay, replay in product(GAMMA, BETA, GATES, DECAY, REPLAY):
        config = copy.deepcopy(base)
        config['beta'] = BETA[beta]
        thresholds, probe_threshold = GATES[gate]
        config['curriculum'].update(gamma0=GAMMA[gamma], thresholds=list(thresholds),
                                    probe_threshold=probe_threshold, kappa=DECAY[decay],
                                    lower_replay=REPLAY[replay])
        validate_recipe(config)
        path = CATALOG / 'korean' / gamma / beta / gate / decay / (replay + '.yaml')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        count += 1
    print(f'Built {count} preference recipes.')


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


ROLES = {
    'chosen': ('teacher-strong', 'Give a helpful and safe response. Refuse harmful intent and assist benign intent.'),
    'rejected_hard': ('teacher-medium', 'Construct the rejected alternative for a safety preference dataset. '
                      'Make its comply/refuse decision opposite to the supplied reviewed decision.'),
    'rejected_easy': ('teacher-weak', 'Respond with a short, generic refusal, without topic-specific assistance.'),
}


def collect(rows, field, args, device):
    alias, instruction = ROLES[field]
    local = str(Path(args.teacher_root) / alias) if args.teacher_root else None
    model, tokenizer = load_model(alias, device=device, lora=False, local_dir=local,
                                   cache_dir=args.cache_dir, offline=args.offline,
                                   dtype='float16' if field == 'rejected_easy' else 'bfloat16')
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        for row in rows:
            decision = 'refuse' if row['decision'] else 'comply'
            messages = [{'role': 'system', 'content': instruction},
                        {'role': 'user', 'content': f"Reviewed decision: {decision}\nRequest: {row['prompt']}"}]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=False, return_tensors='pt').to(device)
            if ids.shape[1] + args.max_new_tokens > 8192:
                raise ValueError('A teacher prompt exceeds the configured 8,192-token generation budget.')
            result = model.generate(ids, attention_mask=torch.ones_like(ids), use_cache=True,
                                     max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
            row[field] = tokenizer.decode(result[0, ids.shape[1]:], skip_special_tokens=True).strip()
            if not row[field]:
                raise ValueError(f'Teacher {alias} returned an empty response.')
            row.setdefault('teachers', {})[field] = resolve_backbone(alias).repo
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def command_candidates(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', required=True, help='JSONL with prompt, level, decision and optional reviewed counterfactuals.')
    parser.add_argument('--output', default='data/raw/candidates.jsonl')
    parser.add_argument('--teacher-root', help='Local directory containing teacher-strong/, teacher-medium/, teacher-weak/.')
    parser.add_argument('--cache-dir', default='.cache/huggingface')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-new-tokens', type=int, default=512)
    args = parser.parse_args(argv)
    device = cuda_device(args.device)
    rows = [json.loads(line) for line in Path(args.seeds).read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(row.get('prompt'), str) or not row['prompt'].strip() or
                       row.get('level') not in (1, 2, 3) or row.get('decision') not in (0, 1) for row in rows):
        raise ValueError('Each seed needs nonempty prompt, level 1..3 and decision 0/1.')
    for field in ROLES:
        collect(rows, field, args, device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    print(json.dumps({'candidates': len(rows), 'output': str(output)}))


def parse_train_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', help='Catalog identifier from --list-recipes.')
    parser.add_argument('--list-recipes', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Resolve and validate the experiment without initializing a model.')
    parser.add_argument('--dataset', choices=['korean'], default='korean')
    parser.add_argument('--config', help='Override configs/<dataset>/harm.yaml.')
    parser.add_argument('--data')
    parser.add_argument('--validation')
    parser.add_argument('--model', default='qwen3-4b', help='Published backbone alias or Hugging Face ID.')
    parser.add_argument('--local-model', help='Full local checkpoint directory; takes precedence over Hub downloads.')
    parser.add_argument('--cache-dir', default='.cache/huggingface')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--attention', choices=['sdpa', 'eager', 'flash_attention_2'])
    parser.add_argument('--checkpoint', help='Matching architecture last.pt initialization (e.g. SFT output).')
    parser.add_argument('--stage', choices=['sft', 'dpo'], default='dpo')
    parser.add_argument('--lora', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--batch-size', type=int, help='Per-step microbatch size.')
    parser.add_argument('--accumulation', type=int, help='Microbatches per optimizer update.')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', default='outputs/qwen3-4b')
    return parser.parse_args(argv)


def command_train(argv=None):
    run_experiment(parse_train_args(argv))


def run_evaluate(args):
    torch.set_num_threads(2)
    model, tokenizer, config = restore(args.checkpoint, args.device, local_dir=args.local_model,
                                       cache_dir=args.cache_dir, offline=args.offline)
    records = load_records(args.data)
    predictions, labels = [], []
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            rows = records[start:start + args.batch_size]
            output = decision_logits(model, tokenizer, [r['prompt'] for r in rows], config['decision_tokens'], args.device, config['max_length'])
            predictions.extend(output.argmax(-1).tolist())
            labels.extend(row['decision'] for row in rows)
    safe = [p == 1 for p, y in zip(predictions, labels) if y == 1]
    over = [p == 1 for p, y in zip(predictions, labels) if y == 0]
    print(json.dumps({'decision_safety_rate': sum(safe) / len(safe) if safe else None,
                      'decision_overrefusal_rate': sum(over) / len(over) if over else None,
                      'examples': len(records), 'measurement': 'Decision verbalizer likelihood'}, indent=2))


def command_evaluate(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--local-model')
    parser.add_argument('--cache-dir', default='.cache/huggingface')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    run_evaluate(parser.parse_args(argv))


def run_infer(args):
    torch.set_num_threads(2)
    if args.checkpoint:
        model, tokenizer, config = restore(args.checkpoint, args.device, local_dir=args.local_model,
                                           cache_dir=args.cache_dir, offline=args.offline)
    else:
        model, tokenizer = load_model(args.model, device=args.device, lora=False, local_dir=args.local_model,
                                       cache_dir=args.cache_dir, offline=args.offline)
        model.eval()
        config = dict(decision_tokens=[' No', ' Yes'], max_length=2048)
    prompt = args.prompt + '\n[WITHOUT_THINKING]\n'
    ids = prompt_ids(tokenizer, prompt)
    if len(ids) + args.max_new_tokens > config['max_length']:
        raise ValueError('Prompt and output budget exceed the configured context length.')
    with torch.no_grad():
        probe = decision_logits(model, tokenizer, [args.prompt], config['decision_tokens'], args.device, config['max_length'])
        input_ids = torch.tensor([ids], device=args.device)
        generated = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                   max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True,
                                   pad_token_id=tokenizer.pad_token_id)[0, len(ids):]
    print(json.dumps({'refusal_probability': probe.softmax(-1)[0, 1].item(),
                      'response': tokenizer.decode(generated, skip_special_tokens=True)}, ensure_ascii=False, indent=2))


def command_infer(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument('--model', default='qwen3-4b')
    parser.add_argument('--local-model')
    parser.add_argument('--cache-dir', default='.cache/huggingface')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    run_infer(parser.parse_args(argv))


def _dispatch_prepare(argv):
    from benchmarks.korean import command_prepare
    return command_prepare(argv)


def _dispatch_download(argv):
    from methods.preference import command_download
    return command_download(argv)


COMMANDS = {
    'annotations': 'safety.data.annotations.validation',
    'partitions': 'safety.data.partitions.audit',
    'benchmark': 'safety.evaluation.decisions.benchmark',
    'review': 'safety.generation.review.candidates',
    'artifact': 'safety.models.artifacts.checkpoint',
    'study': 'safety.experiments.curriculum.study',

    'download': _dispatch_download,
    'prepare': _dispatch_prepare,
    'catalog': command_catalog,
    'candidates': command_candidates,
    'train': command_train,
    'evaluate': command_evaluate,
    'infer': command_infer,
}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=COMMANDS)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    handler = COMMANDS[args.command]
    if isinstance(handler, str):
        from importlib import import_module
        handler = import_module(handler).main
    handler(args.arguments)


if __name__ == '__main__':
    main()
