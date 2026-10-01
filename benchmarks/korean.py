"""Korean safety records, prompt-disjoint partitions and level experience streams."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
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


SPLIT_FRACTIONS = {'sft': .32, 'preferences': .48, 'validation': .10, 'test': .10}


def shingles(text):
    value = canonical_prompt(text)
    return {value[index:index + 5] for index in range(max(1, len(value) - 4))}


def group_related_records(records, threshold=.85, seed=42):
    """Connect records sharing an exact or near-duplicate prompt in either field."""
    from datasketch import MinHash, MinHashLSH
    index = MinHashLSH(threshold=threshold, num_perm=128)
    parent = list(range(len(records)))
    signatures, exact = {}, {}

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for number, row in enumerate(records):
        for prompt in record_prompts(row):
            if prompt in exact:
                parent[root(number)] = root(exact[prompt])
                continue
            tokens = shingles(prompt)
            signature = MinHash(num_perm=128, seed=seed)
            for token in sorted(tokens):
                signature.update(token.encode('utf-8'))
            for key in index.query(signature):
                other, previous = signatures[key]
                if len(tokens & previous) / len(tokens | previous) >= threshold:
                    parent[root(number)] = root(other)
            key = str(len(signatures))
            index.insert(key, signature)
            signatures[key] = (number, tokens)
            exact[prompt] = number
    groups = defaultdict(list)
    for number, row in enumerate(records):
        groups[root(number)].append(row)
    return list(groups.values())


def stratum(row):
    return row.get('source', 'annotated'), row.get('category', str(row['level'])), row['level']


def split_groups(groups, seed=42):
    """Allocate whole groups toward stratified targets and preserve level coverage."""
    rng = random.Random(seed)
    groups = list(groups)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    totals = Counter(stratum(row) for group in groups for row in group)
    remaining = Counter(level for group in groups for level in {row['level'] for row in group})
    if any(remaining[level] < len(SPLIT_FRACTIONS) for level in (1, 2, 3)):
        raise ValueError('Each level needs at least four independent prompt groups for four disjoint splits.')
    splits = {name: [] for name in SPLIT_FRACTIONS}
    counts = {name: Counter() for name in splits}
    coverage = {name: set() for name in splits}
    for group in groups:
        levels = {row['level'] for row in group}
        additions = Counter(stratum(row) for row in group)
        remaining.subtract(levels)
        scores = {}
        for name, fraction in SPLIT_FRACTIONS.items():
            missing = {level: sum(level not in (seen | levels if split == name else seen)
                                  for split, seen in coverage.items()) for level in (1, 2, 3)}
            if any(missing[level] > remaining[level] for level in missing):
                continue
            score = 0.0
            for key, size in additions.items():
                target = totals[key] * fraction
                previous = counts[name][key]
                score += ((previous + size - target) ** 2 - (previous - target) ** 2) / max(1, target)
            scores[name] = score
        if not scores:
            raise ValueError('Connected prompt groups cannot fill every split with all three levels. '
                             'Add independently annotated groups or choose another split seed.')
        name = min(scores, key=scores.get)
        splits[name].extend(group)
        counts[name].update(additions)
        coverage[name].update(levels)
    for rows in splits.values():
        rng.shuffle(rows)
    return splits


class LevelExperience:
    def __init__(self, epoch, records):
        self.current_experience = epoch
        self.records = records

    def sample(self, batch_size, curriculum, stage):
        probabilities = ({1: 1 / 3, 2: 1 / 3, 3: 1 / 3} if stage == 'sft'
                         else curriculum.level_probabilities(self.current_experience))
        levels = random.choices(list(probabilities), weights=list(probabilities.values()), k=batch_size)
        return [random.choice(self.records[level]) for level in levels]


class LevelBenchmark:
    """Disjoint safety-level train/evaluation streams for a preference experiment."""
    def __init__(self, train, validation, epochs):
        self.levels = {level: [row for row in train if row['level'] == level] for level in (1, 2, 3)}
        if any(not rows for rows in self.levels.values()):
            raise ValueError('Training data must contain all three levels.')
        if {row['level'] for row in validation} != {1, 2, 3}:
            raise ValueError('Validation data must contain all three levels for competence gates.')
        training_prompts = {prompt for row in train for prompt in record_prompts(row)}
        validation_prompts = {prompt for row in validation for prompt in record_prompts(row)}
        if training_prompts & validation_prompts:
            raise ValueError('Training and validation primary/counterfactual prompts must be disjoint.')
        self.epochs = epochs
        self.test_stream = validation

    @property
    def train_stream(self):
        return (LevelExperience(epoch, self.levels) for epoch in range(self.epochs))

    @classmethod
    def from_paths(cls, data, validation, epochs):
        if not data or not validation:
            raise ValueError('Training requires --data and a disjoint --validation JSONL.')
        train = load_records(data)
        test = load_records(validation)
        return cls(train, test, epochs)


def command_prepare(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Annotated UTF-8 JSONL with the training record schema.')
    parser.add_argument('--output', default='data/korean')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=.85)
    args = parser.parse_args(argv)
    from datasketch import MinHash, MinHashLSH
    index = MinHashLSH(threshold=args.threshold, num_perm=128)
    unique, token_sets = [], []
    for row in load_records(args.input):
        row = {key: unicodedata.normalize('NFC', value).strip() if isinstance(value, str) else value
               for key, value in row.items()}
        tokens = shingles(row['prompt'])
        signature = MinHash(num_perm=128, seed=args.seed)
        for token in sorted(tokens):
            signature.update(token.encode('utf-8'))
        duplicate = any(len(tokens & token_sets[int(hit)]) / len(tokens | token_sets[int(hit)]) >= args.threshold
                        for hit in index.query(signature))
        if duplicate:
            continue
        index.insert(str(len(unique)), signature)
        token_sets.append(tokens)
        unique.append(row)
    groups = group_related_records(unique, args.threshold, args.seed)
    splits = split_groups(groups, args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        (output / f'{name}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    metadata = dict(seed=args.seed, deduplicated_examples=len(unique), independent_prompt_groups=len(groups),
                    near_duplicate_threshold=args.threshold,
                    counts={key: len(value) for key, value in splits.items()})
    (output / 'split.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata))
