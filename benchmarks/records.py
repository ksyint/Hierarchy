import json
from pathlib import Path
import unicodedata


REQUIRED = {'prompt', 'chosen', 'rejected_easy', 'rejected_hard', 'level', 'decision'}


def canonical_prompt(text):
    return ' '.join(unicodedata.normalize('NFC', text).split())


def record_prompts(row):
    return tuple(dict.fromkeys(canonical_prompt(row[key]) for key in ('prompt', 'cf_prompt') if key in row))


def load_records(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for row in records:
        if not REQUIRED <= row.keys() or row['level'] not in (1, 2, 3) or row['decision'] not in (0, 1):
            raise ValueError(f'Each row needs {sorted(REQUIRED)}, level 1..3 and decision 0/1.')
        if any(not isinstance(row[key], str) or not row[key].strip()
               for key in ('prompt', 'chosen', 'rejected_easy', 'rejected_hard')):
            raise ValueError('Prompt and response fields must be nonempty strings.')
        if ('cf_prompt' in row) != ('cf_decision' in row):
            raise ValueError('cf_prompt and cf_decision must be provided together.')
        if 'cf_decision' in row and row['cf_decision'] not in (0, 1):
            raise ValueError('cf_decision must be 0 or 1.')
        if 'cf_prompt' in row and (not isinstance(row['cf_prompt'], str) or not row['cf_prompt'].strip()):
            raise ValueError('cf_prompt must be a nonempty string.')
    if not records:
        raise ValueError('Empty dataset.')
    return records
