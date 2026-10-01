"""Epoch-boundary recovery of policy, reference, optimizer and sampling state."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import torch


def config_fingerprint(config):
    data = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(data).hexdigest()


def trainable_names(model):
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not names:
        raise ValueError('There are no trainable policy parameters to checkpoint.')
    return names


def selected_weights(model, names):
    parameters = dict(model.named_parameters())
    if not set(names) <= parameters.keys():
        raise ValueError('Saved trainable parameter names do not match the model.')
    return {name: parameters[name].detach().cpu().clone() for name in names}


def restore_weights(model, weights):
    parameters = dict(model.named_parameters())
    if not weights.keys() <= parameters.keys():
        raise ValueError('Checkpoint contains unknown model parameters.')
    for name, value in weights.items():
        if not isinstance(value, torch.Tensor) or value.shape != parameters[name].shape:
            raise ValueError(f'Checkpoint shape mismatch for {name}.')
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite checkpoint values in {name}.')
    with torch.no_grad():
        for name, value in weights.items():
            parameters[name].copy_(value.to(parameters[name].device, dtype=parameters[name].dtype))


def random_state():
    return {'python': random.getstate(), 'torch': torch.random.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all()}


def restore_random(state):
    def tuples(value):
        return tuple(tuples(item) for item in value) if isinstance(value, (list, tuple)) else value
    if len(state['cuda']) != torch.cuda.device_count():
        raise ValueError('CUDA device visibility changed across resume.')
    random.setstate(tuples(state['python']))
    torch.random.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def save_state(path, strategy, sampler, history, next_epoch):
    if next_epoch < 0 or len(history) != next_epoch:
        raise ValueError('History length must equal the next epoch index.')
    names = trainable_names(strategy.model)
    payload = {
        'format': 'koscope-training-v1',
        'stage': strategy.stage,
        'config': strategy.config,
        'config_sha256': config_fingerprint(strategy.config),
        'next_epoch': next_epoch,
        'history': history,
        'model': selected_weights(strategy.model, names),
        'reference': selected_weights(strategy.reference, names) if hasattr(strategy, 'reference') else None,
        'optimizer': strategy.optimizer.state_dict(),
        'curriculum': strategy.curriculum.state_dict(),
        'sampler': sampler.state_dict(),
        'random': random_state(),
        'total_steps': strategy.total_steps,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {'path': str(path), 'next_epoch': next_epoch, 'trainable_tensors': len(names)}


def load_state(path, strategy, sampler):
    state = torch.load(path, map_location='cpu', weights_only=True)
    validate_state_payload(state)
    if state.get('format') != 'koscope-training-v1':
        raise ValueError('Use a training state produced by the epoch checkpoint writer.')
    if state['stage'] != strategy.stage or state['total_steps'] != strategy.total_steps:
        raise ValueError('Training stage or total step count changed across resume.')
    if state['config_sha256'] != config_fingerprint(strategy.config):
        raise ValueError('Training configuration changed across resume.')
    names = set(trainable_names(strategy.model))
    if names != set(state['model']):
        raise ValueError('The trainable parameter set changed across resume.')
    has_reference = hasattr(strategy, 'reference')
    if has_reference != (state['reference'] is not None):
        raise ValueError('Reference-policy state does not match the current learner.')
    if has_reference and set(state['reference']) != names:
        raise ValueError('Reference-policy parameter coverage is incomplete.')
    if state['next_epoch'] != len(state['history']):
        raise ValueError('Checkpoint history and epoch do not agree.')
    if [row['epoch'] for row in state['history']] != list(range(state['next_epoch'])):
        raise ValueError('Checkpoint history is not a consecutive epoch sequence.')
    sampler.load_state_dict(state['sampler'])
    restore_weights(strategy.model, state['model'])
    if has_reference:
        restore_weights(strategy.reference, state['reference'])
    strategy.optimizer.load_state_dict(state['optimizer'])
    strategy.curriculum.load_state_dict(state['curriculum'])
    restore_random(state['random'])
    return state['next_epoch'], state['history']


def state_inventory(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    validate_state_payload(state)
    if state.get('format') != 'koscope-training-v1':
        raise ValueError('Unknown training-state format.')
    parameters = []
    for name, tensor in state['model'].items():
        parameters.append({'name': name, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                           'elements': tensor.numel(), 'bytes': tensor.numel() * tensor.element_size()})
    return {'format': state['format'], 'stage': state['stage'], 'next_epoch': state['next_epoch'],
            'config_sha256': state['config_sha256'], 'corpus': state['sampler']['corpus'],
            'history_entries': len(state['history']), 'parameters': parameters,
            'elements': sum(row['elements'] for row in parameters),
            'trainable_bytes': sum(row['bytes'] for row in parameters),
            'has_reference': state['reference'] is not None,
            'optimizer_groups': len(state['optimizer']['param_groups']),
            'cuda_rng_states': len(state['random']['cuda'])}


def validate_state_payload(state):
    required = {'format', 'stage', 'config', 'config_sha256', 'next_epoch', 'history',
                'model', 'reference', 'optimizer', 'curriculum', 'sampler', 'random', 'total_steps'}
    if not isinstance(state, dict) or not required <= state.keys():
        raise ValueError('Training-state fields are incomplete.')
    if state['format'] != 'koscope-training-v1' or state['stage'] not in ('sft', 'dpo'):
        raise ValueError('Unknown training-state format or learner stage.')
    if config_fingerprint(state['config']) != state['config_sha256']:
        raise ValueError('Training-state configuration digest is inconsistent.')
    if type(state['next_epoch']) is not int or state['next_epoch'] < 0:
        raise ValueError('next_epoch must be a nonnegative integer.')
    if state['next_epoch'] != len(state['history']):
        raise ValueError('History length differs from the resume epoch.')
    if state['total_steps'] < 1:
        raise ValueError('Training-state total_steps must be positive.')
    if not isinstance(state['model'], dict) or not state['model']:
        raise ValueError('Training state has no trainable weights.')
    for field in ('model', 'reference'):
        weights = state[field]
        if weights is None and field == 'reference' and state['stage'] == 'sft':
            continue
        if not isinstance(weights, dict):
            raise ValueError(f'{field} weights are missing.')
        for name, tensor in weights.items():
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise ValueError('Checkpoint weights require string keys and tensors.')
    rng = state['random']
    if not isinstance(rng, dict) or not {'python', 'torch', 'cuda'} <= rng.keys():
        raise ValueError('RNG state is incomplete.')
    for tensor in [rng['torch'], *rng['cuda']]:
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1 or tensor.dtype != torch.uint8:
            raise ValueError('RNG states must be one-dimensional byte tensors.')
    optimizer = state['optimizer']
    if not isinstance(optimizer, dict) or not {'state', 'param_groups'} <= optimizer.keys():
        raise ValueError('Optimizer state is incomplete.')
    group_parameters = [key for group in optimizer['param_groups'] for key in group['params']]
    if len(set(group_parameters)) != len(group_parameters):
        raise ValueError('Optimizer groups contain duplicate parameter identities.')
    if not set(optimizer['state']) <= set(group_parameters):
        raise ValueError('Optimizer state references a parameter outside its groups.')
    return state


def compare_states(first, second):
    left, right = state_inventory(first), state_inventory(second)
    fields = ('format', 'stage', 'config_sha256', 'corpus', 'has_reference', 'optimizer_groups')
    differences = {field: [left[field], right[field]] for field in fields if left[field] != right[field]}
    a = {row['name']: row for row in left['parameters']}
    b = {row['name']: row for row in right['parameters']}
    changed = [name for name in a.keys() & b.keys() if a[name]['shape'] != b[name]['shape']]
    return {'metadata_changes': differences, 'added_parameters': sorted(b.keys() - a.keys()),
            'removed_parameters': sorted(a.keys() - b.keys()), 'shape_changes': sorted(changed),
            'epoch_difference': right['next_epoch'] - left['next_epoch']}


def save_inference_checkpoint(destination, strategy, epoch, score):
    """Publish a validation-selected policy using the existing inference checkpoint format."""
    import math
    import uuid
    if epoch < 0 or not math.isfinite(score):
        raise ValueError('Checkpoint selection requires a valid epoch and finite validation score.')
    destination = Path(destination)
    relative = Path('selected') / f'epoch_{epoch:04d}_{uuid.uuid4().hex[:8]}' / 'pretrained'
    weights = destination / relative
    if weights.exists():
        raise FileExistsError(f'Policy export already exists at {weights}.')
    weights.mkdir(parents=True)
    strategy.model.save_pretrained(weights, safe_serialization=True)
    strategy.tokenizer.save_pretrained(weights)
    payload = {'format': 2, 'config': strategy.config, 'pretrained': relative.as_posix(),
               'curriculum': strategy.curriculum.state_dict(), 'epoch': epoch,
               'selection_metric': 'macro_level_validation_decision_accuracy', 'selection_score': score}
    temporary = destination / 'best.pt.partial'
    try:
        torch.save(payload, temporary)
        temporary.replace(destination / 'best.pt')
    finally:
        temporary.unlink(missing_ok=True)
    return {'checkpoint': str(destination / 'best.pt'), 'epoch': epoch, 'score': score}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', required=True)
    parser.add_argument('--compare')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    report = compare_states(args.state, args.compare) if args.compare else state_inventory(args.state)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(output)}))
