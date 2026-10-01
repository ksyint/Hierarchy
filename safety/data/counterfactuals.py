"""Join reviewed intent variants without breaking pair identity across partitions."""
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import json
from pathlib import Path

from safety.data.schema import binary_label, normalize_text, prompt_key, read_jsonl, write_jsonl


def index_rows(rows, field='id'):
    indexed = {}
    for row in rows:
        identity = str(row.get(field, '')).strip()
        if not identity or identity in indexed:
            raise ValueError(f'{field} must be present and unique.')
        indexed[identity] = row
    return indexed


def edit_summary(original, changed):
    original = normalize_text(original, 'prompt')
    changed = normalize_text(changed, 'cf_prompt')
    matcher = SequenceMatcher(None, original, changed, autojunk=False)
    edits = []
    touched = 0
    for operation, left_start, left_stop, right_start, right_stop in matcher.get_opcodes():
        if operation == 'equal':
            continue
        touched += max(left_stop - left_start, right_stop - right_start)
        edits.append({'operation': operation, 'original_span': [left_start, left_stop],
                      'changed_span': [right_start, right_stop],
                      'before': original[left_start:left_stop], 'after': changed[right_start:right_stop]})
    return {'similarity': matcher.ratio(), 'changed_fraction': touched / max(len(original), len(changed)),
            'edits': edits}


def attach_variants(records, reviews, require_all=False):
    originals, annotations = index_rows(records), index_rows(reviews, 'original_id')
    unknown = annotations.keys() - originals.keys()
    if unknown:
        raise ValueError(f'Reviews reference unknown records: {sorted(unknown)[:10]}')
    if require_all and originals.keys() != annotations.keys():
        raise ValueError('Some records have no reviewed counterfactual.')
    result, provenance = [], []
    for identity, source in originals.items():
        row = dict(source)
        if identity in annotations:
            annotation = annotations[identity]
            if annotation.get('reviewed') is not True:
                raise ValueError(f'{identity} needs an explicit reviewed=true annotation.')
            counterfactual = normalize_text(annotation.get('prompt'), 'cf_prompt')
            label = binary_label(annotation.get('decision'), 'cf_decision')
            original_label = binary_label(row.get('decision'), 'decision')
            if prompt_key(counterfactual) == prompt_key(row['prompt']):
                raise ValueError(f'{identity} has an unchanged counterfactual prompt.')
            relation = annotation.get('relation', 'intent_flip')
            if relation not in ('intent_flip', 'surface_variant'):
                raise ValueError('Use intent_flip or surface_variant as the reviewed relation.')
            if (relation == 'intent_flip') != (original_label != label):
                raise ValueError(f'{identity}: relation and reviewed labels disagree.')
            row.update(cf_prompt=counterfactual, cf_decision=label, cf_relation=relation)
            row['group_id'] = source.get('group_id', identity)
            row['cf_language'] = annotation.get('language', source.get('language', 'ko'))
            row['cf_review_id'] = annotation.get('review_id', identity)
            provenance.append({'id': identity, 'relation': relation,
                               **edit_summary(row['prompt'], counterfactual)})
        result.append(row)
    return result, provenance


def pair_inventory(rows):
    counts, groups = Counter(), defaultdict(list)
    primary, changed, collisions = {}, {}, []
    for row in rows:
        identity = str(row['id'])
        original = prompt_key(row['prompt'])
        primary.setdefault(original, []).append(identity)
        if 'cf_prompt' not in row:
            counts['unpaired'] += 1
            continue
        relation = 'intent_flip' if row['decision'] != row['cf_decision'] else 'surface_variant'
        counts[relation] += 1
        counts[f'level_{row["level"]}_{relation}'] += 1
        key = prompt_key(row['cf_prompt'])
        changed.setdefault(key, []).append(identity)
        groups[str(row.get('group_id', identity))].append(identity)
    for key in primary.keys() & changed.keys():
        collisions.append({'primary_ids': primary[key], 'variant_ids': changed[key]})
    return {'counts': dict(counts), 'pair_groups': dict(groups), 'shared_prompts': collisions}


def export_review(rows, maximum_per_stratum=20, seed=42):
    import random
    if maximum_per_stratum < 1:
        raise ValueError('Review sample size must be positive.')
    rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['level'], row['decision'], row.get('source', 'unspecified'))].append(row)
    output = []
    for key in sorted(groups, key=str):
        selected = rng.sample(groups[key], min(maximum_per_stratum, len(groups[key])))
        for row in selected:
            output.append({'original_id': row['id'], 'original_prompt': row['prompt'],
                           'original_decision': row['decision'], 'level': row['level'],
                           'prompt': '', 'decision': None, 'relation': 'intent_flip', 'reviewed': False})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['attach', 'inventory', 'review-sheet'])
    parser.add_argument('--input', required=True)
    parser.add_argument('--reviews')
    parser.add_argument('--output', required=True)
    parser.add_argument('--report')
    parser.add_argument('--require-all', action='store_true')
    parser.add_argument('--per-stratum', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    if Path(args.input).resolve() == Path(args.output).resolve():
        parser.error('Use a separate output path.')
    rows = [row for _, row in read_jsonl(args.input)]
    index_rows(rows)
    if args.operation == 'attach':
        if not args.reviews or not args.report:
            parser.error('attach requires --reviews and --report.')
        reviews = [row for _, row in read_jsonl(args.reviews)]
        output, edits = attach_variants(rows, reviews, args.require_all)
        write_jsonl(args.output, output)
        report = {**pair_inventory(output), 'edits': edits}
    elif args.operation == 'review-sheet':
        output = export_review(rows, args.per_stratum, args.seed)
        write_jsonl(args.output, output)
        report = {'review_records': len(output), 'seed': args.seed}
    else:
        report = pair_inventory(rows)
    destination = args.report or args.output
    if args.operation == 'inventory' or args.report:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key not in ('edits', 'pair_groups')}))
