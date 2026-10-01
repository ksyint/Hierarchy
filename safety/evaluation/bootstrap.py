"""Paired stratified bootstrap and Holm correction across matched evaluations."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random

from safety.data.schema import read_jsonl


METRICS = ('accuracy', 'safety', 'overrefusal', 'cf_pair_accuracy')


def index_predictions(rows):
    result = {}
    for row in rows:
        identity = (str(row.get('seed', 0)), str(row.get('id', '')))
        if not identity[1] or identity in result:
            raise ValueError('Each seed needs unique nonempty prediction IDs.')
        for field in ('decision', 'prediction'):
            if row.get(field) not in (0, 1):
                raise ValueError(f'{field} must be binary.')
        result[identity] = row
    if not result:
        raise ValueError('No prediction records.')
    return result


def align_predictions(reference, candidate):
    left, right = index_predictions(reference), index_predictions(candidate)
    if left.keys() != right.keys():
        raise ValueError('Paired evaluation needs the same IDs for every seed.')
    for identity, row in left.items():
        other = right[identity]
        for field in ('decision', 'level', 'source', 'language', 'cf_decision', 'group_id'):
            if row.get(field) != other.get(field):
                raise ValueError(f'{identity}: paired metadata differs at {field}.')
    identities = sorted(left)
    return [left[key] for key in identities], [right[key] for key in identities]


def outcome(row, metric):
    if metric == 'accuracy':
        return float(row['prediction'] == row['decision'])
    if metric == 'safety':
        return float(row['prediction'] == 1) if row['decision'] == 1 else None
    if metric == 'overrefusal':
        return float(row['prediction'] == 1) if row['decision'] == 0 else None
    if metric == 'cf_pair_accuracy':
        if 'cf_decision' not in row:
            return None
        if row.get('cf_prediction') not in (0, 1):
            raise ValueError('Counterfactual pair accuracy requires cf_prediction.')
        return float(row['prediction'] == row['decision'] and row['cf_prediction'] == row['cf_decision'])
    raise ValueError(f'Unknown metric {metric}.')


def metric_mean(values):
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def percentile(values, quantile):
    if not values or not 0 <= quantile <= 1:
        raise ValueError('A nonempty sample and quantile in [0,1] are required.')
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def bootstrap_units(rows, strata=('seed', 'level', 'decision'), cluster_field=None):
    grouped = defaultdict(lambda: defaultdict(list))
    cluster_strata = {}
    for index, row in enumerate(rows):
        stratum = tuple(str(row.get(field, 'unspecified')) for field in strata)
        cluster = (str(row.get('seed', 0)), str(row.get(cluster_field, row['id']))) if cluster_field else ('row', index)
        if cluster in cluster_strata and cluster_strata[cluster] != stratum:
            raise ValueError('A resampling cluster crosses strata. Use coarser strata.')
        cluster_strata[cluster] = stratum
        grouped[stratum][cluster].append(index)
    return [list(clusters.values()) for _, clusters in sorted(grouped.items())]


def paired_bootstrap(reference, candidate, metric='safety', resamples=1000, seed=42,
                     strata=('seed', 'level', 'decision'), cluster_field=None, confidence=.95):
    if resamples < 100 or not 0 < confidence < 1:
        raise ValueError('Use at least 100 resamples and confidence in (0,1).')
    left, right = align_predictions(reference, candidate)
    a, b = [outcome(row, metric) for row in left], [outcome(row, metric) for row in right]
    first, second = metric_mean(a), metric_mean(b)
    if first is None or second is None:
        raise ValueError(f'No eligible observations for {metric}.')
    units = bootstrap_units(left, strata, cluster_field)
    rng = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        indices = []
        for clusters in units:
            for _ in clusters:
                indices.extend(rng.choice(clusters))
        before, after = metric_mean([a[index] for index in indices]), metric_mean([b[index] for index in indices])
        if before is not None and after is not None:
            deltas.append(after - before)
    if len(deltas) < .95 * resamples:
        raise ValueError('Too many resamples lack eligible observations. Refine the strata.')
    observed = second - first
    tail = (1 - confidence) / 2
    centered_extreme = sum(abs(value - observed) >= abs(observed) for value in deltas)
    return {'metric': metric, 'reference': first, 'candidate': second, 'difference': observed,
            'interval': [percentile(deltas, tail), percentile(deltas, 1 - tail)],
            'confidence': confidence, 'p_two_sided': (centered_extreme + 1) / (len(deltas) + 1),
            'resamples': len(deltas), 'requested_resamples': resamples, 'seed': seed,
            'examples': len(left), 'eligible_examples': sum(value is not None for value in a),
            'strata': list(strata), 'cluster_field': cluster_field,
            'direction': 'lower_is_better' if metric == 'overrefusal' else 'higher_is_better'}


def holm_adjust(p_values):
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in p_values):
        raise ValueError('p-values must be finite and in [0,1].')
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.] * len(p_values)
    maximum = 0.
    for rank, index in enumerate(order):
        maximum = max(maximum, min(1., (len(p_values) - rank) * p_values[index]))
        adjusted[index] = maximum
    return adjusted


def compare_methods(reference, candidates, metrics, **options):
    results = []
    for name, rows in candidates.items():
        for metric in metrics:
            results.append({'method': name, **paired_bootstrap(reference, rows, metric, **options)})
    adjusted = holm_adjust([row['p_two_sided'] for row in results])
    for row, p_value in zip(results, adjusted):
        row['p_holm'] = p_value
        row['significant_005'] = p_value < .05
    return {'comparisons': results, 'correction_family_size': len(results),
            'correction_family': 'all requested method and metric comparisons'}


def seed_summary(rows, metric):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get('seed', 0))].append(outcome(row, metric))
    means = {seed: metric_mean(values) for seed, values in grouped.items()}
    present = [value for value in means.values() if value is not None]
    center = sum(present) / len(present) if present else None
    std = math.sqrt(sum((value - center) ** 2 for value in present) / (len(present) - 1)) if len(present) > 1 else None
    return {'per_seed': means, 'mean': center, 'sample_stddev': std}


def stratified_subset(rows, per_seed, seed=42):
    if per_seed < 1:
        raise ValueError('Per-seed subsample size must be positive.')
    index_predictions(rows)
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row.get('seed', 0))][(row.get('level'), row['decision'])].append(row)
    rng = random.Random(seed)
    selected, allocations = [], {}
    for experiment_seed, strata in sorted(grouped.items()):
        population = sum(map(len, strata.values()))
        if population < per_seed:
            raise ValueError(f'Seed {experiment_seed} has only {population} examples, fewer than {per_seed}.')
        targets = {key: per_seed * len(values) / population for key, values in strata.items()}
        counts = {key: int(value) for key, value in targets.items()}
        remainder = per_seed - sum(counts.values())
        order = sorted(strata, key=lambda key: (-(targets[key] - counts[key]), str(key)))
        for key in order[:remainder]:
            counts[key] += 1
        if any(count == 0 for count in counts.values()):
            raise ValueError('The requested subsample omits a stratum. Increase --per-seed.')
        for key in sorted(strata, key=str):
            ordered = sorted(strata[key], key=lambda row: str(row['id']))
            selected.extend(rng.sample(ordered, counts[key]))
        allocations[experiment_seed] = {str(key): count for key, count in counts.items()}
    return selected, {'per_seed': per_seed, 'sampling_seed': seed, 'allocations': allocations}


def matched_subset(reference, candidates, per_seed, seed):
    for rows in candidates.values():
        align_predictions(reference, rows)
    selected, report = stratified_subset(reference, per_seed, seed)
    identities = {(str(row.get('seed', 0)), str(row['id'])) for row in selected}
    subsets = {}
    for name, rows in candidates.items():
        subsets[name] = [row for row in rows if (str(row.get('seed', 0)), str(row['id'])) in identities]
    report['selected_ids'] = [list(identity) for identity in sorted(identities)]
    return selected, subsets, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', nargs='+', required=True)
    parser.add_argument('--candidate', action='append', required=True, metavar='NAME=JSONL')
    parser.add_argument('--metrics', nargs='+', choices=METRICS, default=['safety', 'overrefusal'])
    parser.add_argument('--strata', nargs='+', default=['seed', 'level', 'decision'])
    parser.add_argument('--cluster-field')
    parser.add_argument('--per-seed', type=int, help='Stratified held-out prompts per training seed.')
    parser.add_argument('--resamples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    reference = [row for path in args.reference for _, row in read_jsonl(path)]
    candidates = defaultdict(list)
    for item in args.candidate:
        if '=' not in item:
            parser.error('Use --candidate NAME=JSONL.')
        name, path = item.split('=', 1)
        if not name:
            parser.error('Candidate names cannot be empty.')
        candidates[name].extend(row for _, row in read_jsonl(path))
    subset = None
    if args.per_seed:
        if args.cluster_field:
            parser.error('Row subsampling and cluster resampling cannot be combined.')
        reference, candidates, subset = matched_subset(reference, candidates, args.per_seed, args.seed)
    report = compare_methods(reference, candidates, args.metrics, resamples=args.resamples,
                             seed=args.seed, strata=tuple(args.strata), cluster_field=args.cluster_field)
    report['reference_seed_summary'] = {metric: seed_summary(reference, metric) for metric in args.metrics}
    if subset is not None:
        report['subsample'] = subset
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'comparisons': len(report['comparisons'])}))
