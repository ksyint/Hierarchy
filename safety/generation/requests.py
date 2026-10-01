"""Create stable teacher requests and join their reviewed response provenance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from safety.data.schema import normalize_record, read_jsonl, write_jsonl


TEACHERS = {'chosen': 'teacher-strong', 'rejected_hard': 'teacher-medium', 'rejected_easy': 'teacher-weak'}
INSTRUCTIONS = {
    'chosen': 'Give a helpful and safe response. Refuse harmful intent and assist benign intent.',
    'rejected_hard': 'Construct the rejected alternative for a safety preference dataset. '
                     'Make its comply/refuse decision opposite to the supplied reviewed decision.',
    'rejected_easy': 'Respond with a short, generic refusal, without topic-specific assistance.',
}


def stable_digest(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def make_request(source, role, variant=0, instructions=None):
    row = normalize_record(source, 'seed')
    if role not in TEACHERS or variant < 0:
        raise ValueError('A teacher role and a nonnegative variant are required.')
    instruction = (instructions or INSTRUCTIONS)[role]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('Teacher instructions must be nonempty strings.')
    decision = 'refuse' if row['decision'] else 'comply'
    messages = [{'role': 'system', 'content': instruction},
                {'role': 'user', 'content': f'Reviewed decision: {decision}\nRequest: {row["prompt"]}'}]
    contract = {'record_id': row['id'], 'record_sha256': stable_digest(row),
                'role': role, 'variant': variant, 'teacher': TEACHERS[role], 'messages': messages}
    return {'request_id': stable_digest(contract), **contract,
            'level': row['level'], 'decision': row['decision'], 'language': row['language']}


def request_catalog(rows, roles=None, variants=1, instructions=None):
    roles = tuple(roles or TEACHERS)
    if variants < 1 or len(set(roles)) != len(roles):
        raise ValueError('Use a positive variant count and unique teacher roles.')
    seen, requests = set(), []
    for row in rows:
        normalized = normalize_record(row, 'seed')
        if normalized['id'] in seen:
            raise ValueError(f'Duplicate seed id {normalized["id"]}.')
        seen.add(normalized['id'])
        for role in roles:
            for variant in range(variants):
                requests.append(make_request(normalized, role, variant, instructions))
    if not requests:
        raise ValueError('No requests were constructed.')
    return requests


def validate_request(row):
    keys = ('record_id', 'record_sha256', 'role', 'variant', 'teacher', 'messages')
    if any(key not in row for key in keys):
        raise ValueError(f'A request needs {keys}.')
    if row['role'] not in TEACHERS or row['teacher'] != TEACHERS[row['role']]:
        raise ValueError('Teacher alias does not match the request role.')
    if stable_digest({key: row[key] for key in keys}) != row.get('request_id'):
        raise ValueError('Request contents changed after its identity was assigned.')
    messages = row['messages']
    if not isinstance(messages, list) or not messages:
        raise ValueError('A request needs chat messages.')
    for message in messages:
        if message.get('role') not in ('system', 'user', 'assistant'):
            raise ValueError('Unsupported chat role.')
        if not isinstance(message.get('content'), str) or not message['content'].strip():
            raise ValueError('Chat messages cannot be empty.')
    return row


def validate_response(row, request):
    if row.get('request_id') != request['request_id']:
        raise ValueError('Response identity does not match its request.')
    for field in ('record_id', 'role', 'teacher', 'record_sha256'):
        if row.get(field) != request[field]:
            raise ValueError(f'Response {field} does not match its request.')
    text = row.get('text')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Teacher response is empty.')
    if row.get('generated_tokens', 0) < 1:
        raise ValueError('Response needs a positive generated token count.')
    return row


def completed_requests(requests, response_paths):
    indexed = {row['request_id']: validate_request(row) for row in requests}
    if len(indexed) != len(requests):
        raise ValueError('Request IDs must be unique.')
    completed = {}
    for path in response_paths:
        for _, row in read_jsonl(path):
            identity = row.get('request_id')
            if identity not in indexed:
                raise ValueError(f'{path} contains an unknown response {identity}.')
            if identity in completed:
                raise ValueError(f'Duplicate response for {identity}.')
            completed[identity] = validate_response(row, indexed[identity])
    pending = [row for row in requests if row['request_id'] not in completed]
    return completed, pending


def repair_response_tail(path):
    path = Path(path)
    with path.open('r+b') as stream:
        stream.seek(0, 2)
        end = stream.tell()
        if not end:
            return 0
        stream.seek(end - 1)
        if stream.read(1) == b'\n':
            return 0
        position, tail = end, b''
        while position:
            size = min(position, 65536)
            position -= size
            stream.seek(position)
            tail = stream.read(size) + tail
            boundary = tail.rfind(b'\n')
            if boundary >= 0:
                position += boundary + 1
                tail = tail[boundary + 1:]
                break
        try:
            json.loads(tail.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            stream.truncate(position)
            return end - position
        stream.seek(end)
        stream.write(b'\n')
    return 0


def catalog_summary(requests):
    return {'requests': len(requests),
            'records': len({row['record_id'] for row in requests}),
            'roles': dict(Counter(row['role'] for row in requests)),
            'teachers': dict(Counter(row['teacher'] for row in requests)),
            'levels': dict(Counter(row['level'] for row in requests)),
            'catalog_sha256': stable_digest([row['request_id'] for row in requests])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='operation', required=True)
    build = subparsers.add_parser('build')
    build.add_argument('--seeds', required=True)
    build.add_argument('--roles', nargs='+', choices=list(TEACHERS), default=list(TEACHERS))
    build.add_argument('--variants', type=int, default=1)
    build.add_argument('--instructions', help='JSON mapping from role to reviewed system instruction.')
    pending = subparsers.add_parser('pending')
    pending.add_argument('--requests', required=True)
    pending.add_argument('--responses', nargs='+', required=True)
    for command in (build, pending):
        command.add_argument('--output', required=True)
        command.add_argument('--report', required=True)
    args = parser.parse_args(argv)
    if args.operation == 'build':
        rows = [row for _, row in read_jsonl(args.seeds)]
        instructions = json.loads(Path(args.instructions).read_text()) if args.instructions else None
        requests = request_catalog(rows, args.roles, args.variants, instructions)
        report = catalog_summary(requests)
    else:
        all_requests = [row for _, row in read_jsonl(args.requests)]
        completed, requests = completed_requests(all_requests, args.responses)
        report = {**catalog_summary(all_requests), 'completed': len(completed), 'pending': len(requests)}
    write_jsonl(args.output, requests)
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
