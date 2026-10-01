"""Assemble preferences from teacher responses and explicit candidate reviews."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from safety.data.schema import normalize_record, read_jsonl, write_jsonl
from safety.generation.requests import completed_requests, stable_digest


def parse_thinking(text):
    openings = ('<think>', '[THINKING]')
    closings = ('</think>', '[/THINKING]')
    for start, stop in zip(openings, closings):
        if text.lstrip().startswith(start):
            position = text.find(stop)
            if position < 0:
                raise ValueError('A response has an unterminated reasoning block.')
            begin = text.index(start) + len(start)
            thought = text[begin:position].strip()
            answer = text[position + len(stop):].strip()
            if not answer:
                raise ValueError('Reasoning-only responses cannot form a preference pair.')
            return answer, thought
    return text.strip(), ''


def review_index(rows):
    result = {}
    for row in rows:
        identity = row.get('request_id')
        if not identity or identity in result:
            raise ValueError('Candidate reviews require unique request IDs.')
        if row.get('accepted') not in (True, False):
            raise ValueError('Each review needs a boolean accepted value.')
        if row['accepted']:
            score = float(row.get('quality', 0))
            if not math.isfinite(score):
                raise ValueError('Review quality must be finite.')
            if row.get('decision') not in (0, 1):
                raise ValueError('Accepted candidates require a reviewed response decision.')
            row = dict(row, quality=score)
        result[identity] = row
    return result


def eligible_response(response, review, seed, allow_truncated=False):
    if not review['accepted']:
        return False, 'review_rejected'
    if response.get('truncated') and not allow_truncated:
        return False, 'length_truncated'
    role = response['role']
    if role == 'chosen' and review['decision'] != seed['decision']:
        return False, 'chosen_decision_mismatch'
    if role == 'rejected_hard' and review['decision'] == seed['decision']:
        return False, 'hard_negative_did_not_flip_decision'
    if role == 'rejected_easy' and review['decision'] != 1:
        return False, 'easy_negative_is_not_refusal'
    try:
        parse_thinking(response['text'])
    except ValueError as error:
        return False, str(error)
    return True, None


def assemble_preferences(seeds, requests, responses, reviews, allow_truncated=False):
    seeds = [normalize_record(row, 'seed') for row in seeds]
    seed_index = {row['id']: row for row in seeds}
    if len(seed_index) != len(seeds):
        raise ValueError('Seed IDs must be unique.')
    request_index = {row['request_id']: row for row in requests}
    reviewed = review_index(reviews)
    if not reviewed.keys() <= responses.keys():
        raise ValueError('Some reviews refer to absent teacher responses.')
    grouped = defaultdict(lambda: defaultdict(list))
    rejected = Counter()
    for identity, response in responses.items():
        request = request_index[identity]
        record_id = request['record_id']
        if record_id not in seed_index:
            raise ValueError(f'Unknown seed {record_id}.')
        seed = seed_index[record_id]
        if request['record_sha256'] != stable_digest(seed):
            raise ValueError(f'Seed {record_id} changed after teacher generation.')
        if identity not in reviewed:
            rejected['unreviewed'] += 1
            continue
        review = reviewed[identity]
        eligible, reason = eligible_response(response, review, seed, allow_truncated)
        if not eligible:
            rejected[reason] += 1
            continue
        grouped[record_id][response['role']].append((review['quality'], identity, response))
    output, incomplete = [], []
    for identity, seed in seed_index.items():
        roles = grouped[identity]
        missing = [role for role in ('chosen', 'rejected_hard', 'rejected_easy') if not roles[role]]
        if missing:
            incomplete.append({'id': identity, 'missing': missing})
            continue
        row = dict(seed)
        provenance = {}
        for role, candidates in roles.items():
            _, request_id, response = max(candidates, key=lambda item: (item[0], item[1]))
            answer, thought = parse_thinking(response['text'])
            row[role] = answer
            if thought:
                row[role + '_thinking'] = thought
            provenance[role] = {'request_id': request_id, 'teacher': response['teacher'],
                                'model_id': response.get('model_id'), 'revision': response.get('revision'),
                                'quality': reviewed[request_id]['quality']}
        try:
            row = normalize_record(row)
        except ValueError as error:
            incomplete.append({'id': identity, 'error': str(error)})
            continue
        row['teacher_selection'] = provenance
        output.append(row)
    return output, {'assembled': len(output), 'incomplete': incomplete, 'candidate_rejections': dict(rejected)}


def export_reviews(responses):
    return [{'request_id': row['request_id'], 'record_id': row['record_id'], 'role': row['role'],
             'text': row['text'], 'accepted': False, 'decision': None, 'quality': 0,
             'truncated': row.get('truncated', False)} for row in responses.values()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['review-sheet', 'assemble'])
    parser.add_argument('--requests', required=True)
    parser.add_argument('--responses', nargs='+', required=True)
    parser.add_argument('--seeds')
    parser.add_argument('--reviews')
    parser.add_argument('--output', required=True)
    parser.add_argument('--report')
    parser.add_argument('--allow-truncated', action='store_true')
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args(argv)
    requests = [row for _, row in read_jsonl(args.requests)]
    responses, pending = completed_requests(requests, args.responses)
    if args.operation == 'review-sheet':
        rows = export_reviews(responses)
        report = {'candidates': len(rows), 'pending_requests': len(pending)}
    else:
        if not args.seeds or not args.reviews or not args.report:
            parser.error('assemble requires --seeds, --reviews and --report.')
        seeds = [row for _, row in read_jsonl(args.seeds)]
        reviews = [row for _, row in read_jsonl(args.reviews)]
        rows, report = assemble_preferences(seeds, requests, responses, reviews, args.allow_truncated)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.operation == 'assemble' and (not rows or report['incomplete'] and not args.allow_incomplete):
        raise ValueError('Some seeds have no complete reviewed preference. Inspect the report.')
    write_jsonl(args.output, rows)
    print(json.dumps({'output': args.output, 'records': len(rows)}))
