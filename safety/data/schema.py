"""Record contracts for source conversion, teacher generation and training."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import unicodedata


TEXT_FIELDS = ('prompt', 'chosen', 'rejected_easy', 'rejected_hard')
OPTIONAL_TEXT = ('cf_prompt', 'category', 'source', 'language', 'group_id')
THOUGHT_FIELDS = tuple(field + '_thinking' for field in TEXT_FIELDS[1:])


def normalize_text(value, field):
    if not isinstance(value, str):
        raise ValueError(f'{field} must be text.')
    value = unicodedata.normalize('NFC', value).strip()
    if not value or '\x00' in value:
        raise ValueError(f'{field} must be nonempty text without null characters.')
    return value


def prompt_key(value):
    text = ' '.join(normalize_text(value, 'prompt').split())
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def binary_label(value, field):
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, int) or value not in (0, 1):
        raise ValueError(f'{field} must be the integer 0 or 1.')
    return value


def normalize_record(record, stage='preference'):
    if stage not in ('seed', 'preference', 'prediction'):
        raise ValueError(f'Unknown record stage {stage}.')
    if not isinstance(record, dict):
        raise ValueError('Each record must be an object.')
    result = dict(record)
    required = TEXT_FIELDS if stage == 'preference' else ('prompt',)
    for field in required:
        result[field] = normalize_text(record.get(field), field)
    for field in OPTIONAL_TEXT + THOUGHT_FIELDS:
        if field in record:
            result[field] = normalize_text(record[field], field)
    level = record.get('level')
    if isinstance(level, bool) or not isinstance(level, int) or level not in (1, 2, 3):
        raise ValueError('level must be an integer in 1..3.')
    result['decision'] = binary_label(record.get('decision'), 'decision')
    if ('cf_prompt' in record) != ('cf_decision' in record):
        raise ValueError('cf_prompt and cf_decision must occur together.')
    if 'cf_decision' in record:
        result['cf_decision'] = binary_label(record['cf_decision'], 'cf_decision')
        if prompt_key(result['cf_prompt']) == prompt_key(result['prompt']):
            raise ValueError('A counterfactual must have a different prompt.')
    identity = record.get('id')
    result['id'] = normalize_text(str(identity), 'id') if identity is not None else prompt_key(result['prompt'])
    result.setdefault('language', 'ko')
    if stage == 'preference':
        chosen = result['chosen']
        if chosen == result['rejected_easy'] or chosen == result['rejected_hard']:
            raise ValueError('Chosen and rejected responses must differ.')
    if stage == 'prediction':
        result['prediction'] = binary_label(record.get('prediction'), 'prediction')
        probability = float(record['refusal_probability'])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('refusal_probability must be finite and in [0,1].')
        result['refusal_probability'] = probability
    return result


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'{path}:{number}: {error.msg}') from error
            if not isinstance(value, dict):
                raise ValueError(f'{path}:{number}: expected an object.')
            yield number, value


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def inspect_records(path, stage='preference'):
    accepted, issues = [], []
    identities, prompts = set(), set()
    counts = {key: Counter() for key in ('level', 'source', 'category', 'language', 'decision')}
    duplicate_prompts = 0
    for number, source in read_jsonl(path):
        try:
            row = normalize_record(source, stage)
            if row['id'] in identities:
                raise ValueError(f'Duplicate id {row["id"]}.')
        except (ValueError, TypeError, KeyError) as error:
            issues.append({'line': number, 'error': str(error)})
            continue
        identities.add(row['id'])
        key = prompt_key(row['prompt'])
        duplicate_prompts += key in prompts
        prompts.add(key)
        accepted.append(row)
        for field, counter in counts.items():
            counter[str(row.get(field, 'unspecified'))] += 1
    report = {
        'stage': stage,
        'accepted': len(accepted),
        'rejected': len(issues),
        'duplicate_prompts': duplicate_prompts,
        'counterfactual_pairs': sum('cf_prompt' in row for row in accepted),
        'counts': {field: dict(sorted(counter.items())) for field, counter in counts.items()},
        'issues': issues,
    }
    return accepted, report


def require_disjoint(paths):
    owners, collisions = {}, []
    for name, path in paths.items():
        for number, row in read_jsonl(path):
            for field in ('prompt', 'cf_prompt'):
                if field not in row:
                    continue
                key = prompt_key(row[field])
                previous = owners.get(key)
                if previous and previous[0] != name:
                    collisions.append({'first': previous, 'second': [name, number, field]})
                else:
                    owners[key] = [name, number, field]
    if collisions:
        raise ValueError(f'Cross-partition prompt overlap: {collisions[:10]}')
    return {'partitions': list(paths), 'unique_prompts': len(owners)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--stage', choices=['seed', 'preference', 'prediction'], default='preference')
    parser.add_argument('--output')
    parser.add_argument('--report', required=True)
    args = parser.parse_args(argv)
    if args.output and Path(args.output).resolve() == Path(args.input).resolve():
        parser.error('Use a separate normalized output path.')
    rows, report = inspect_records(args.input, args.stage)
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if report['rejected'] or not rows:
        raise ValueError(f'Record validation failed. See {destination}.')
    if args.output:
        write_jsonl(args.output, rows)
    print(json.dumps({key: value for key, value in report.items() if key != 'issues'}))
