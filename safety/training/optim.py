"""Optimizer groups, finite-gradient checks and update-level learning rates."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import torch


def learning_rate_scale(step, total_steps, warmup_fraction=.05, minimum_ratio=0.0):
    if total_steps < 1 or not 0 <= step < total_steps:
        raise ValueError('Step must lie inside the configured training interval.')
    if not 0 <= warmup_fraction < 1 or not 0 <= minimum_ratio <= 1:
        raise ValueError('Invalid warmup or minimum learning-rate ratio.')
    warmup = max(1, int(warmup_fraction * total_steps))
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    cosine = .5 * (1 + math.cos(math.pi * progress))
    return minimum_ratio + (1 - minimum_ratio) * cosine


def parameter_groups(model, weight_decay, no_decay_bias_norm=False):
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError('weight_decay must be finite and nonnegative.')
    grouped = defaultdict(list)
    names = defaultdict(list)
    identities = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in identities:
            continue
        identities.add(id(parameter))
        excluded = no_decay_bias_norm and (parameter.ndim < 2 or name.endswith('.bias'))
        decay = 0.0 if excluded else weight_decay
        grouped[decay].append(parameter)
        names[decay].append(name)
    if not grouped:
        raise ValueError('No trainable parameters were found.')
    groups = [{'params': grouped[decay], 'weight_decay': decay} for decay in sorted(grouped)]
    inventory = [{'weight_decay': decay, 'names': names[decay],
                  'elements': sum(parameter.numel() for parameter in grouped[decay])} for decay in sorted(grouped)]
    return groups, inventory


def build_optimizer(model, config):
    rate = float(config['lr'])
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('Learning rate must be finite and positive.')
    groups, inventory = parameter_groups(model, config['weight_decay'], config.get('no_decay_bias_norm', False))
    betas = tuple(config.get('adam_betas', [.9, .999]))
    if len(betas) != 2 or any(not 0 <= value < 1 for value in betas):
        raise ValueError('Adam beta values must lie in [0,1).')
    optimizer = torch.optim.AdamW(groups, lr=rate, betas=betas, eps=config.get('adam_epsilon', 1e-8))
    return optimizer, inventory


def update_learning_rate(optimizer, step, total_steps, config):
    scale = learning_rate_scale(step, total_steps, config.get('warmup_fraction', .05),
                                config.get('minimum_lr_ratio', 0.0))
    for group in optimizer.param_groups:
        group['lr'] = config['lr'] * scale
    return config['lr'] * scale


def gradient_statistics(model):
    squared = []
    maximum = []
    missing, nonfinite = [], []
    tensors = elements = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        gradient = parameter.grad.detach()
        if gradient.device.type != 'cuda':
            raise ValueError('Training gradients must reside on CUDA.')
        if gradient.is_sparse:
            gradient = gradient.coalesce().values()
        if not torch.isfinite(gradient).all():
            nonfinite.append(name)
            continue
        value = gradient.float()
        squared.append(value.square().sum())
        maximum.append(value.abs().max() if value.numel() else value.new_zeros(()))
        tensors += 1
        elements += gradient.numel()
    norm = torch.stack(squared).sum().sqrt().item() if squared else 0.0
    largest = torch.stack(maximum).max().item() if maximum else 0.0
    return {'l2_norm': norm, 'maximum_absolute': largest, 'gradient_tensors': tensors,
            'gradient_elements': elements, 'missing': missing, 'nonfinite': nonfinite}


def optimizer_update(model, optimizer, maximum_norm=1.0, inspect=False):
    if not math.isfinite(maximum_norm) or maximum_norm <= 0:
        raise ValueError('Gradient clipping norm must be finite and positive.')
    statistics = gradient_statistics(model) if inspect else None
    if statistics and statistics['nonfinite']:
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(f'Nonfinite gradients in {statistics["nonfinite"][:8]}')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), maximum_norm, error_if_nonfinite=True)
    optimizer.step()
    return {'gradient_norm': float(norm.detach()), 'details': statistics}


def estimate_optimizer_memory(model, optimizer_bits=32):
    if optimizer_bits not in (8, 16, 32):
        raise ValueError('Supported optimizer memory estimates use 8, 16 or 32 bits.')
    trainable = frozen = model_bytes = gradients = 0
    for parameter in model.parameters():
        count = parameter.numel()
        size = count * parameter.element_size()
        model_bytes += size
        if parameter.requires_grad:
            trainable += count
            gradients += size
        else:
            frozen += count
    moments = trainable * 2 * optimizer_bits // 8
    return {'trainable_elements': trainable, 'frozen_elements': frozen, 'parameter_bytes': model_bytes,
            'gradient_bytes': gradients, 'adam_moment_bytes': moments,
            'parameter_gradient_moment_bytes': model_bytes + gradients + moments,
            'excludes_activations_and_reference': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--learning-rate', type=float, default=5e-6)
    parser.add_argument('--warmup-fraction', type=float, default=.05)
    parser.add_argument('--minimum-ratio', type=float, default=0.0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.steps < 1:
        parser.error('Steps and learning rate must be positive.')
    rates = [{'step': step, 'learning_rate': args.learning_rate * learning_rate_scale(
        step, args.steps, args.warmup_fraction, args.minimum_ratio)} for step in range(args.steps)]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({'steps': args.steps, 'rates': rates}, indent=2) + '\n')
    print(json.dumps({'output': str(destination), 'steps': len(rates)}))
