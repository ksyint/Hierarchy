"""Apply a reviewed category-to-level mapping and compare independent annotations."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import yaml

from safety.data.schema import read_jsonl, write_jsonl


def load_taxonomy(path):
    value = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict) or not value.get('version'):
        raise ValueError('A taxonomy requires a version and a categories mapping.')
    categories = value.get('categories')
    if not isinstance(categories, dict) or not categories:
        raise ValueError('categories must be a nonempty mapping.')
    aliases = {}
    for name, rule in categories.items():
        if not isinstance(rule, dict) or rule.get('level') not in (1, 2, 3):
            raise ValueError(f'{name} needs a level in 1..3.')
        if isinstance(rule['level'], bool):
            raise ValueError(f'{name} needs an integer level.')
        names = [name, *rule.get('aliases', [])]
        for alias in names:
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError('Category aliases must be nonempty strings.')
            key = alias.strip().casefold()
            if key in aliases:
                raise ValueError(f'Ambiguous category alias {alias}.')
            aliases[key] = name
    return value, aliases


def assign_levels(rows, taxonomy, aliases, allow_override=False):
    assigned, unknown, conflicts = [], [], []
    for number, source in enumerate(rows):
        row = dict(source)
        key = str(row.get('category', '')).strip().casefold()
        name = aliases.get(key)
        if name is None:
            unknown.append({'row': number, 'id': row.get('id'), 'category': row.get('category')})
            continue
        level = taxonomy['categories'][name]['level']
        if row.get('level') is not None and row['level'] != level:
            conflicts.append({'id': row.get('id'), 'before': row['level'], 'after': level})
            if not allow_override:
                continue
        row.update(category=name, level=level, taxonomy_version=taxonomy['version'])
        assigned.append(row)
    return assigned, {'assigned': len(assigned), 'unknown_categories': unknown, 'conflicts': conflicts}


def confusion_matrix(reference, predicted, labels):
    positions = {label: index for index, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for target, output in zip(reference, predicted):
        if target not in positions or output not in positions:
            raise ValueError('An annotation uses an unknown label.')
        matrix[positions[target]][positions[output]] += 1
    return matrix


def classification_scores(matrix, labels):
    support = [sum(row) for row in matrix]
    predicted = [sum(row[index] for row in matrix) for index in range(len(labels))]
    total = sum(support)
    details = {}
    for index, label in enumerate(labels):
        true_positive = matrix[index][index]
        precision = true_positive / predicted[index] if predicted[index] else 0.0
        recall = true_positive / support[index] if support[index] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        details[str(label)] = {'support': support[index], 'precision': precision, 'recall': recall, 'f1': f1}
    accuracy = sum(matrix[index][index] for index in range(len(labels))) / total if total else None
    expected = sum(a * b for a, b in zip(support, predicted)) / total ** 2 if total else None
    kappa = (accuracy - expected) / (1 - expected) if total and expected < 1 else None
    represented = [details[str(label)]['f1'] for index, label in enumerate(labels) if support[index]]
    return {'examples': total, 'accuracy': accuracy, 'cohen_kappa': kappa,
            'macro_f1': sum(represented) / len(represented) if represented else None,
            'per_label': details, 'confusion': matrix, 'labels': labels}


def compare_annotations(reference, predicted, field='level'):
    def index(rows):
        result = {}
        for row in rows:
            identity = str(row.get('id', ''))
            if not identity or identity in result:
                raise ValueError('Independent annotations require unique IDs.')
            result[identity] = row
        return result
    truth, guesses = index(reference), index(predicted)
    if truth.keys() != guesses.keys():
        raise ValueError('Annotation sets must contain exactly the same IDs.')
    if not truth:
        raise ValueError('No paired annotations.')
    labels = [1, 2, 3] if field == 'level' else [0, 1]
    matrix = confusion_matrix([row[field] for row in truth.values()],
                              [guesses[identity][field] for identity in truth], labels)
    disagreements = []
    by_source = defaultdict(list)
    for identity, row in truth.items():
        other = guesses[identity]
        if row[field] != other[field]:
            disagreements.append({'id': identity, 'reference': row[field], 'predicted': other[field]})
        by_source[str(row.get('source', 'unspecified'))].append(identity)
    result = classification_scores(matrix, labels)
    result['disagreements'] = disagreements
    result['sources'] = {}
    for source, identities in sorted(by_source.items()):
        matrix = confusion_matrix([truth[key][field] for key in identities],
                                  [guesses[key][field] for key in identities], labels)
        result['sources'][source] = classification_scores(matrix, labels)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='operation', required=True)
    assign = subparsers.add_parser('assign')
    assign.add_argument('--input', required=True)
    assign.add_argument('--taxonomy', required=True)
    assign.add_argument('--output', required=True)
    assign.add_argument('--allow-override', action='store_true')
    compare = subparsers.add_parser('compare')
    compare.add_argument('--reference', required=True)
    compare.add_argument('--predicted', required=True)
    compare.add_argument('--field', choices=['level', 'decision'], default='level')
    parser.add_argument('--report', required=True)
    args = parser.parse_args(argv)
    if args.operation == 'assign':
        if Path(args.input).resolve() == Path(args.output).resolve():
            parser.error('Use a separate output file.')
        taxonomy, aliases = load_taxonomy(args.taxonomy)
        rows = [row for _, row in read_jsonl(args.input)]
        assigned, report = assign_levels(rows, taxonomy, aliases, args.allow_override)
        report['level_counts'] = dict(Counter(row['level'] for row in assigned))
        valid = not report['unknown_categories'] and (args.allow_override or not report['conflicts'])
        if valid:
            write_jsonl(args.output, assigned)
    else:
        reference = [row for _, row in read_jsonl(args.reference)]
        predicted = [row for _, row in read_jsonl(args.predicted)]
        report = compare_annotations(reference, predicted, args.field)
        valid = True
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if not valid:
        raise ValueError(f'Review unmapped or conflicting categories in {destination}.')
    print(json.dumps({'report': str(destination)}))
