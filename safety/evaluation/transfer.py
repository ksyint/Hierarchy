"""Matched multilingual decision analysis on reviewed parallel prompt groups."""
import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path

from safety.data.schema import read_jsonl, write_jsonl
from safety.evaluation.bootstrap import holm_adjust, paired_bootstrap
from safety.evaluation.reports import metrics, validate_predictions, wilson


def parallel_index(rows):
    grouped = defaultdict(dict)
    identities = set()
    for row in rows:
        language = str(row.get('language', '')).strip()
        parallel_id = str(row.get('parallel_id', '')).strip()
        identity = (str(row.get('seed', 0)), str(row.get('id', '')), language)
        if not language or not parallel_id or not identity[1] or identity in identities:
            raise ValueError('Parallel records require language, parallel_id and unique per-language IDs.')
        if row.get('decision') not in (0, 1) or row.get('level') not in (1, 2, 3):
            raise ValueError('Parallel records require reviewed decisions and hierarchy levels.')
        key = (identity[0], parallel_id)
        if language in grouped[key]:
            raise ValueError(f'Duplicate translation for {key}, {language}.')
        grouped[key][language] = row
        identities.add(identity)
    for key, translations in grouped.items():
        intents = {(row['decision'], row['level']) for row in translations.values()}
        if len(intents) != 1:
            raise ValueError(f'Parallel group {key} changes reviewed intent or hierarchy level.')
    return grouped


def coverage_report(rows, languages):
    indexed = parallel_index(rows)
    counts = {language: 0 for language in languages}
    incomplete, complete = [], []
    for key, translations in sorted(indexed.items()):
        missing = sorted(set(languages) - translations.keys())
        for language in counts:
            counts[language] += language in translations
        if missing:
            incomplete.append({'seed': key[0], 'parallel_id': key[1], 'missing_languages': missing})
        else:
            complete.append(key)
    return {'parallel_groups': len(indexed), 'complete_groups': len(complete),
            'language_counts': counts, 'incomplete': incomplete}, indexed, complete


def matched_language_rows(indexed, complete, language):
    result = []
    for seed, parallel_id in complete:
        source = indexed[(seed, parallel_id)][language]
        row = dict(source, id=parallel_id, seed=seed, language='parallel', source='parallel')
        row.pop('group_id', None)
        result.append(row)
    validate_predictions([dict(row, id=f'{row["seed"]}:{row["id"]}') for row in result])
    return result


def translation_disagreements(indexed, complete, first, second):
    results = []
    for key in complete:
        a, b = indexed[key][first], indexed[key][second]
        for row in (a, b):
            if row.get('prediction') not in (0, 1):
                raise ValueError('Translation comparison requires binary predictions.')
            probability = float(row['refusal_probability'])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('Refusal probabilities must be finite and in [0,1].')
        results.append({'seed': key[0], 'parallel_id': key[1], 'level': a['level'],
                        'decision': a['decision'], 'first_prediction': a['prediction'],
                        'second_prediction': b['prediction'],
                        'probability_difference': b['refusal_probability'] - a['refusal_probability'],
                        'decision_changed': a['prediction'] != b['prediction']})
    changed = sum(row['decision_changed'] for row in results)
    return {'first': first, 'second': second, 'decision_change_rate': wilson(changed, len(results)),
            'mean_probability_difference': sum(row['probability_difference'] for row in results) / len(results),
            'disagreements': [row for row in results if row['decision_changed']]}


def compare_languages(rows, languages, reference='ko', resamples=1000, seed=42):
    if reference not in languages or len(set(languages)) != len(languages) or len(languages) < 2:
        raise ValueError('Use at least two unique languages including the reference language.')
    coverage, indexed, complete = coverage_report(rows, languages)
    if not complete:
        raise ValueError('No fully aligned parallel groups.')
    aligned = {language: matched_language_rows(indexed, complete, language) for language in languages}
    summaries = {language: metrics(records) for language, records in aligned.items()}
    comparisons = []
    for language in languages:
        if language == reference:
            continue
        for metric in ('accuracy', 'safety', 'overrefusal'):
            if metric == 'safety' and not any(row['decision'] == 1 for row in aligned[reference]):
                continue
            if metric == 'overrefusal' and not any(row['decision'] == 0 for row in aligned[reference]):
                continue
            report = paired_bootstrap(aligned[reference], aligned[language], metric, resamples, seed)
            comparisons.append({'reference_language': reference, 'candidate_language': language, **report})
    corrected = holm_adjust([row['p_two_sided'] for row in comparisons])
    for row, value in zip(comparisons, corrected):
        row['p_holm'] = value
    disagreements = [translation_disagreements(indexed, complete, first, second)
                     for first, second in itertools.combinations(languages, 2)]
    return {'coverage': coverage, 'matched_language_metrics': summaries,
            'paired_comparisons': comparisons, 'translation_decisions': disagreements,
            'correction_family_size': len(comparisons)}


def export_matched(rows, languages, destination):
    coverage, indexed, complete = coverage_report(rows, languages)
    if not complete:
        raise ValueError('No complete parallel groups to export.')
    destination = Path(destination)
    paths = {}
    for language in languages:
        if not language.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Language names must be safe filename components.')
        selected = [indexed[key][language] for key in complete]
        path = destination / f'{language}.jsonl'
        write_jsonl(path, selected)
        paths[language] = str(path)
    return {'coverage': coverage, 'outputs': paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['coverage', 'compare', 'export'])
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--languages', nargs='+', default=['ko', 'ja', 'zh', 'en'])
    parser.add_argument('--reference', default='ko')
    parser.add_argument('--resamples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', required=True)
    parser.add_argument('--destination')
    args = parser.parse_args(argv)
    if len(set(args.languages)) != len(args.languages):
        parser.error('Language names must be unique.')
    rows = [row for path in args.input for _, row in read_jsonl(path)]
    if args.operation == 'coverage':
        report, _, _ = coverage_report(rows, args.languages)
    elif args.operation == 'export':
        if not args.destination:
            parser.error('export requires --destination.')
        report = export_matched(rows, args.languages, args.destination)
    else:
        report = compare_languages(rows, args.languages, args.reference, args.resamples, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'records': len(rows)}))
