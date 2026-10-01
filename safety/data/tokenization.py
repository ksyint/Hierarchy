"""Inspect response masks and token budgets with the selected backbone tokenizer."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from safety.data.korean import format_pair
from safety.data.schema import normalize_record, read_jsonl, write_jsonl


def load_tokenizer(name, local=None, cache_dir=None, offline=False):
    from transformers import AutoTokenizer
    from safety.models.backbones import resolve_backbone
    spec = resolve_backbone(name)
    options = {'local_files_only': offline, 'cache_dir': cache_dir, 'trust_remote_code': spec.remote_code}
    if not local:
        options['revision'] = spec.revision
    tokenizer = AutoTokenizer.from_pretrained(local or spec.repo, **options)
    if tokenizer.eos_token_id is None:
        raise ValueError('The tokenizer needs an EOS token.')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prefix_tokens(tokenizer, prompt):
    if getattr(tokenizer, 'chat_template', None):
        return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                                             add_generation_prompt=True, enable_thinking=False)
    first = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    return [first] + tokenizer.encode(prompt, add_special_tokens=False)


def sequence_contract(tokenizer, prompt, response, max_length):
    if max_length < 2:
        raise ValueError('max_length must allow both context and a completion.')
    prefix = prefix_tokens(tokenizer, prompt)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    suffix = response_ids + [tokenizer.eos_token_id]
    if len(suffix) >= max_length:
        return {'valid': False, 'prompt_tokens': len(prefix), 'response_tokens': len(suffix),
                'error': 'response_exceeds_budget'}
    retained = min(len(prefix), max_length - len(suffix))
    return {'valid': True, 'prompt_tokens': len(prefix), 'response_tokens': len(suffix),
            'retained_prompt_tokens': retained, 'truncated_prompt_tokens': len(prefix) - retained,
            'sequence_tokens': retained + len(suffix), 'scored_tokens': len(suffix)}


def quantiles(values):
    ordered = sorted(values)
    if not ordered:
        return {'count': 0, 'min': None, 'median': None, 'p95': None, 'max': None}
    def percentile(q):
        position = (len(ordered) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])
    return {'count': len(values), 'min': ordered[0], 'median': percentile(.5),
            'p95': percentile(.95), 'max': ordered[-1], 'mean': sum(values) / len(values)}


def inspect_tokens(rows, tokenizer, max_length=2048):
    details, rejected = [], []
    totals = Counter()
    for source in rows:
        row = normalize_record(source)
        for explicit in (False, True):
            for hard in (False, True):
                prompt, chosen, rejected_response = format_pair(row, hard, explicit)
                for role, response in (('chosen', chosen), ('rejected', rejected_response)):
                    result = sequence_contract(tokenizer, prompt, response, max_length)
                    entry = {'id': row['id'], 'level': row['level'], 'explicit': explicit,
                             'negative': 'hard' if hard else 'easy', 'role': role, **result}
                    details.append(entry)
                    totals['sequences'] += 1
                    totals['invalid'] += not result['valid']
                    totals['truncated_prompts'] += result.get('truncated_prompt_tokens', 0) > 0
                    if not result['valid']:
                        rejected.append(entry)
    summaries = {}
    for field in ('prompt_tokens', 'response_tokens', 'sequence_tokens', 'truncated_prompt_tokens'):
        summaries[field] = quantiles([row[field] for row in details if field in row])
    return details, {'max_length': max_length, 'counts': dict(totals),
                     'token_lengths': summaries, 'invalid_sequences': rejected}


def cache_tokens(rows, tokenizer, destination, max_length=2048):
    tokenizer_name = str(getattr(tokenizer, 'name_or_path', 'unspecified'))
    records = []
    for source in rows:
        row = normalize_record(source)
        entries = []
        for explicit in (False, True):
            for hard in (False, True):
                prompt, chosen, rejected = format_pair(row, hard, explicit)
                prefix = prefix_tokens(tokenizer, prompt)
                for role, response in (('chosen', chosen), ('rejected', rejected)):
                    suffix = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
                    if len(suffix) >= max_length:
                        raise ValueError(f'{row["id"]}: {role} exceeds the response budget.')
                    context = prefix[-(max_length - len(suffix)):]
                    entries.append({'explicit': explicit, 'hard': hard, 'role': role,
                                    'input_ids': context + suffix,
                                    'response_mask': [0] * len(context) + [1] * len(suffix)})
        digest = hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        records.append({'id': row['id'], 'record_sha256': digest, 'tokenizer': tokenizer_name,
                        'max_length': max_length, 'variants': entries})
    write_jsonl(destination, records)
    return len(records)


def padding_estimate(lengths, batch_size):
    if batch_size < 1:
        raise ValueError('batch_size must be positive.')
    def count(values):
        return sum(max(values[start:start + batch_size]) * len(values[start:start + batch_size])
                   for start in range(0, len(values), batch_size))
    real = sum(lengths)
    original = count(lengths)
    bucketed = count(sorted(lengths))
    return {'real_tokens': real, 'original_padded_tokens': original, 'sorted_padded_tokens': bucketed,
            'original_padding_fraction': 1 - real / original if original else 0,
            'sorted_padding_fraction': 1 - real / bucketed if bucketed else 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--model', default='qwen3-4b')
    parser.add_argument('--local-model')
    parser.add_argument('--cache-dir', default='.cache/huggingface')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--report', required=True)
    parser.add_argument('--details')
    parser.add_argument('--token-cache')
    args = parser.parse_args(argv)
    tokenizer = load_tokenizer(args.model, args.local_model, args.cache_dir, args.offline)
    rows = [row for _, row in read_jsonl(args.input)]
    details, report = inspect_tokens(rows, tokenizer, args.max_length)
    lengths = [row['sequence_tokens'] for row in details if row['valid']]
    report['padding'] = padding_estimate(lengths, args.batch_size)
    report['model'] = args.model
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    if args.details:
        write_jsonl(args.details, details)
    if report['counts'].get('invalid'):
        raise ValueError(f'Some responses exceed the token budget. See {output}.')
    if args.token_cache:
        cache_tokens(rows, tokenizer, args.token_cache, args.max_length)
    print(json.dumps({'report': str(output), 'records': len(rows)}))
