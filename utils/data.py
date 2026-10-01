import json
from pathlib import Path


REQUIRED = {'prompt', 'chosen', 'rejected_easy', 'rejected_hard', 'level', 'decision'}


def load_records(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for row in records:
        if not REQUIRED <= row.keys() or row['level'] not in (1, 2, 3) or row['decision'] not in (0, 1):
            raise ValueError(f'Each row needs {sorted(REQUIRED)}, level 1..3 and decision 0/1.')
        if ('cf_prompt' in row) != ('cf_decision' in row):
            raise ValueError('cf_prompt and cf_decision must be provided together.')
        if 'cf_decision' in row and row['cf_decision'] not in (0, 1):
            raise ValueError('cf_decision must be 0 or 1.')
    if not records:
        raise ValueError('Empty dataset.')
    return records


def synthetic_records(size=48, offset=0):
    # Abstract benign access-control labels exercise the objective without generating
    # harmful responses or pretending to be a Korean safety benchmark.
    rows = []
    for i in range(size):
        decision = i % 2
        flag = 'DENY' if decision else 'ALLOW'
        opposite = 'ALLOW' if decision else 'DENY'
        rows.append({'prompt': f'Case {i + offset}: {flag}', 'chosen': 'No.' if decision else 'Yes.',
                     'rejected_easy': 'Maybe.', 'rejected_hard': 'Yes.' if decision else 'No.',
                     'level': i % 3 + 1, 'decision': decision,
                     'cf_prompt': f'Case {i + offset}: {opposite}', 'cf_decision': 1 - decision,
                     'chosen_thinking': 'Read the access flag.'})
    return rows


def format_pair(record, hard, explicit):
    rejected_key = 'rejected_hard' if hard else 'rejected_easy'
    prompt, chosen, rejected = record['prompt'], record['chosen'], record[rejected_key]
    if explicit:
        def add_thinking(answer, key):
            thought = record.get(key + '_thinking', '')
            return f'[THINKING]{thought}[/THINKING]\n{answer}' if thought else answer
        chosen = add_thinking(chosen, 'chosen')
        rejected = add_thinking(rejected, rejected_key)
    else:
        prompt += '\n[WITHOUT_THINKING]\n'
    return prompt, chosen, rejected
