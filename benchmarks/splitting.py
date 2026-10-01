"""Keep linked primary and counterfactual prompts in the same data partition."""
from collections import Counter, defaultdict
import random

from .records import canonical_prompt, record_prompts


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
