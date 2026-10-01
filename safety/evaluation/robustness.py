"""Controlled hierarchy-label perturbations and reviewed surface-variant analysis."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from safety.data.schema import normalize_record, read_jsonl, write_jsonl
from safety.evaluation.reports import metrics, validate_predictions, wilson


def perturb_levels(rows, fraction, seed=42, mode='uniform', levels=(1, 2, 3)):
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Noise fraction must be in [0,1].')
    if mode not in ('uniform', 'adjacent') or not set(levels) <= {1, 2, 3} or not levels:
        raise ValueError('Use uniform/adjacent noise and nonempty eligible levels.')
    normalized = [normalize_record(row) for row in rows]
    identities = [row['id'] for row in normalized]
    if len(set(identities)) != len(identities):
        raise ValueError('Record IDs must be unique before perturbation.')
    groups = defaultdict(list)
    for index, row in enumerate(normalized):
        if row['level'] in levels:
            groups[(row['level'], row.get('source', 'unspecified'), row['decision'])].append(index)
    rng = random.Random(seed)
    total = sum(map(len, groups.values()))
    desired = round(total * fraction)
    allocations = {key: int(len(indices) * fraction) for key, indices in groups.items()}
    remaining = desired - sum(allocations.values())
    order = sorted(groups, key=lambda key: (-(len(groups[key]) * fraction - allocations[key]), str(key)))
    for key in order[:remaining]:
        allocations[key] += 1
    selected = set()
    for key in sorted(groups, key=str):
        selected.update(rng.sample(groups[key], allocations[key]))
    changes = []
    for index in sorted(selected):
        row = normalized[index]
        previous = row['level']
        choices = [level for level in (1, 2, 3) if level != previous and (mode == 'uniform' or abs(level - previous) == 1)]
        changed = rng.choice(choices)
        row['original_level'] = previous
        row['level'] = changed
        row['hierarchy_noise'] = {'seed': seed, 'mode': mode, 'fraction': fraction}
        changes.append({'id': row['id'], 'before': previous, 'after': changed})
    identity = hashlib.sha256(json.dumps(identities, separators=(',', ':')).encode()).hexdigest()
    report = {'seed': seed, 'mode': mode, 'requested_fraction': fraction,
              'eligible_records': total, 'changed_records': len(changes),
              'realized_fraction': len(changes) / total if total else 0,
              'input_id_sha256': identity, 'changes': changes,
              'level_counts': dict(Counter(row['level'] for row in normalized))}
    return normalized, report


def compare_surface_variants(rows):
    validate_predictions(rows)
    groups = defaultdict(list)
    for row in rows:
        family = row.get('variant_group')
        if not family:
            raise ValueError('Surface-variant predictions require variant_group.')
        groups[(str(row.get('seed', 0)), str(family))].append(row)
    comparisons = []
    for key, variants in sorted(groups.items()):
        originals = [row for row in variants if row.get('variant') == 'original']
        if len(originals) != 1:
            raise ValueError(f'{key} must contain exactly one original variant.')
        original = originals[0]
        names = [row.get('variant') for row in variants]
        if None in names or len(set(names)) != len(names):
            raise ValueError('Variant names must be present and unique within a group.')
        for row in variants:
            if row is original:
                continue
            if row['decision'] != original['decision'] or row['level'] != original['level']:
                raise ValueError('Surface variants must preserve reviewed intent and hierarchy level.')
            comparisons.append({'group': key[1], 'seed': key[0], 'variant': row['variant'],
                                'level': row['level'], 'decision': row['decision'],
                                'decision_changed': row['prediction'] != original['prediction'],
                                'probability_change': row['refusal_probability'] - original['refusal_probability'],
                                'original_correct': original['prediction'] == original['decision'],
                                'variant_correct': row['prediction'] == row['decision']})
    by_variant = defaultdict(list)
    for row in comparisons:
        by_variant[row['variant']].append(row)
    summaries = {}
    for name, values in sorted(by_variant.items()):
        summaries[name] = {
            'pairs': len(values),
            'decision_change_rate': wilson(sum(row['decision_changed'] for row in values), len(values)),
            'original_accuracy': sum(row['original_correct'] for row in values) / len(values),
            'variant_accuracy': sum(row['variant_correct'] for row in values) / len(values),
            'mean_probability_change': sum(row['probability_change'] for row in values) / len(values),
            'mean_absolute_probability_change': sum(abs(row['probability_change']) for row in values) / len(values),
        }
    return {'groups': len(groups), 'comparisons': comparisons, 'variants': summaries}


def noise_results(paths):
    results = []
    shared_ids = None
    for specification in paths:
        fraction_text, path = specification.split('=', 1)
        fraction = float(fraction_text)
        if not 0 <= fraction <= 1:
            raise ValueError('Noise fractions must lie in [0,1].')
        rows = [row for _, row in read_jsonl(path)]
        validate_predictions(rows)
        identities = {(row['id'], row['decision'], row.get('level')) for row in rows}
        if shared_ids is None:
            shared_ids = identities
        elif identities != shared_ids:
            raise ValueError('Noise runs must be evaluated on the same unchanged held-out set.')
        results.append({'training_noise_fraction': fraction, 'predictions': path, **metrics(rows)})
    fractions = [row['training_noise_fraction'] for row in results]
    if len(set(fractions)) != len(fractions):
        raise ValueError('Noise fractions must be unique within a comparison.')
    return {'runs': sorted(results, key=lambda row: row['training_noise_fraction'])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='operation', required=True)
    noise = subparsers.add_parser('perturb')
    noise.add_argument('--input', required=True)
    noise.add_argument('--fraction', required=True, type=float)
    noise.add_argument('--seed', type=int, default=42)
    noise.add_argument('--mode', choices=['uniform', 'adjacent'], default='uniform')
    noise.add_argument('--levels', nargs='+', type=int, choices=[1, 2, 3], default=[1, 2, 3])
    noise.add_argument('--output', required=True)
    noise.add_argument('--report', required=True)
    variants = subparsers.add_parser('surface')
    variants.add_argument('--predictions', required=True)
    variants.add_argument('--report', required=True)
    compare = subparsers.add_parser('noise-results')
    compare.add_argument('--run', action='append', required=True, metavar='FRACTION=JSONL')
    compare.add_argument('--report', required=True)
    args = parser.parse_args(argv)
    if args.operation == 'perturb':
        if Path(args.input).resolve() == Path(args.output).resolve():
            parser.error('Preserve the original training file and use a separate output.')
        rows = [row for _, row in read_jsonl(args.input)]
        changed, report = perturb_levels(rows, args.fraction, args.seed, args.mode, tuple(args.levels))
        write_jsonl(args.output, changed)
    elif args.operation == 'surface':
        rows = [row for _, row in read_jsonl(args.predictions)]
        report = compare_surface_variants(rows)
    else:
        report = noise_results(args.run)
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(output)}))
