"""Resumable teacher candidates and round-trip preference annotation packets."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

from methods.preference import cuda_device
from safety.data.annotations.validation import AnnotationCorpus


ROLES = ('chosen', 'rejected_easy', 'rejected_hard')


def read_jsonl(path):
    result = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not result:
        raise ValueError(f'Empty JSONL input: {path}')
    if any(not isinstance(row, dict) for row in result):
        raise ValueError('Each JSONL row must be an object.')
    return result


def save_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
    temporary.replace(path)


def response_id(identity, role, text):
    return hashlib.sha256((identity + '\n' + role + '\n' + text).encode('utf-8')).hexdigest()[:20]


@dataclass(frozen=True)
class ReviewChoice:
    identity: str
    status: str
    chosen: str = ''
    rejected_easy: str = ''
    rejected_hard: str = ''
    note: str = ''

    @classmethod
    def from_record(cls, row):
        if row.get('status') not in {'accepted', 'rejected', 'pending'}:
            raise ValueError('Review status must be accepted, rejected or pending.')
        return cls(str(row['id']), row['status'], str(row.get('chosen', '')),
                   str(row.get('rejected_easy', '')), str(row.get('rejected_hard', '')), str(row.get('note', '')))


class ReviewPacket:
    def __init__(self, records):
        self.records = {}
        for original in records:
            row = dict(original)
            row.setdefault('id', AnnotationCorpus.identity(row))
            if row['id'] in self.records:
                raise ValueError(f'Duplicate candidate ID: {row["id"]}')
            if any(not isinstance(row.get(role), str) or not row[role].strip() for role in ROLES):
                raise ValueError('Review candidates need all three nonempty response fields.')
            self.records[row['id']] = row

    def response_map(self, row):
        return {response_id(row['id'], role, row[role]): role for role in ROLES}

    def export(self, destination, seed=42):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)
        review = []
        for identity, row in self.records.items():
            responses = [{'id': response_id(identity, role, row[role]), 'text': row[role]} for role in ROLES]
            rng.shuffle(responses)
            item = {'id': identity, 'prompt': row['prompt'], 'level': row['level'],
                    'decision': row['decision'], 'responses': responses, 'status': 'pending',
                    'chosen': '', 'rejected_easy': '', 'rejected_hard': '', 'note': ''}
            for key in ('cf_prompt', 'cf_decision', 'category', 'language'):
                if key in row:
                    item[key] = row[key]
            review.append(item)
        save_jsonl(destination / 'candidates.jsonl', self.records.values())
        save_jsonl(destination / 'review.jsonl', review)
        manifest = {'format': 1, 'seed': seed, 'examples': len(review),
                    'candidate_sha256': hashlib.sha256((destination / 'candidates.jsonl').read_bytes()).hexdigest()}
        (destination / 'packet.json').write_text(json.dumps(manifest, indent=2) + '\n')
        return manifest

    def apply(self, reviews, require_complete=True):
        decisions = {}
        for row in reviews:
            choice = ReviewChoice.from_record(row)
            if choice.identity in decisions or choice.identity not in self.records:
                raise ValueError('Reviews must refer to unique IDs in the candidate packet.')
            decisions[choice.identity] = choice
        if require_complete and decisions.keys() != self.records.keys():
            raise ValueError('A review entry is required for every candidate example.')
        accepted, rejected, pending = [], [], []
        for identity, source in self.records.items():
            choice = decisions.get(identity, ReviewChoice(identity, 'pending'))
            if choice.status != 'accepted':
                (rejected if choice.status == 'rejected' else pending).append(dict(source, review_status=choice.status))
                continue
            selections = [getattr(choice, role) for role in ROLES]
            available = self.response_map(source)
            if len(set(selections)) != 3 or set(selections) != available.keys():
                raise ValueError('Accepted reviews must assign every response exactly once to the three roles.')
            record = dict(source, review_status='accepted')
            teachers = {}
            for role, selected in zip(ROLES, selections):
                original_role = available[selected]
                record[role] = source[original_role]
                for key in (role + '_thinking',):
                    record.pop(key, None)
                original_thinking = source.get(original_role + '_thinking')
                if original_thinking is not None:
                    record[role + '_thinking'] = original_thinking
                if source.get('teachers', {}).get(original_role):
                    teachers[role] = source['teachers'][original_role]
            record['teachers'] = teachers
            record['review_note'] = choice.note
            accepted.append(record)
        return accepted, rejected, pending


def generate_candidates(args):
    from koscope import collect
    device = cuda_device(args.device)
    seeds = read_jsonl(args.seeds)
    for row in seeds:
        if not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
            raise ValueError('Each generation seed needs prompt text.')
        if row.get('level') not in (1, 2, 3) or row.get('decision') not in (0, 1):
            raise ValueError('Each generation seed needs reviewed level and decision annotations.')
        row.setdefault('id', AnnotationCorpus.identity(row))
    if len({row['id'] for row in seeds}) != len(seeds):
        raise ValueError('Seed identities must be unique.')
    destination = Path(args.output)
    previous = {row['id']: row for row in read_jsonl(destination)} if args.resume and destination.exists() else {}
    for row in seeds:
        saved = previous.get(row['id'])
        if saved:
            for key in ('prompt', 'level', 'decision', 'cf_prompt', 'cf_decision'):
                if saved.get(key) != row.get(key):
                    raise ValueError(f'Generation seed changed after checkpointing: {row["id"]}')
            row.update(saved)
    try:
        for role in ROLES:
            pending = [row for row in seeds if not row.get(role)]
            if pending:
                collect(pending, role, args, device)
                save_jsonl(destination, seeds)
    finally:
        save_jsonl(destination, seeds)
    print(json.dumps({'examples': len(seeds), 'complete': sum(all(row.get(role) for role in ROLES) for row in seeds)}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest='operation', required=True)
    generate = operations.add_parser('generate')
    generate.add_argument('--seeds', required=True)
    generate.add_argument('--output', required=True)
    generate.add_argument('--resume', action='store_true')
    generate.add_argument('--teacher-root')
    generate.add_argument('--cache-dir', default='.cache/huggingface')
    generate.add_argument('--offline', action='store_true')
    generate.add_argument('--device', default='cuda')
    generate.add_argument('--max-new-tokens', type=int, default=512)
    export = operations.add_parser('export')
    export.add_argument('--candidates', required=True)
    export.add_argument('--output', required=True)
    export.add_argument('--seed', type=int, default=42)
    apply = operations.add_parser('apply')
    apply.add_argument('--candidates', required=True)
    apply.add_argument('--review', required=True)
    apply.add_argument('--output', required=True)
    apply.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args(argv)
    if args.operation == 'generate':
        generate_candidates(args)
        return
    packet = ReviewPacket(read_jsonl(args.candidates))
    if args.operation == 'export':
        result = packet.export(args.output, args.seed)
    else:
        accepted, rejected, pending = packet.apply(read_jsonl(args.review), not args.allow_partial)
        output = Path(args.output)
        for name, rows in [('accepted', accepted), ('rejected', rejected), ('pending', pending)]:
            save_jsonl(output / f'{name}.jsonl', rows)
        result = {'accepted': len(accepted), 'rejected': len(rejected), 'pending': len(pending)}
        (output / 'review-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
