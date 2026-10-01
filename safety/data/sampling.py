"""Resumable level sampling with optional within-level source balancing."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from safety.data.schema import normalize_record, read_jsonl


def corpus_identity(levels):
    digest = hashlib.sha256()
    for level in sorted(levels):
        for row in levels[level]:
            digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode())
            digest.update(b'\n')
    return digest.hexdigest()


def tuple_state(value):
    return tuple(tuple_state(item) for item in value) if isinstance(value, (tuple, list)) else value


class LevelSampler:
    def __init__(self, levels, seed=42, balance_field=None, temperature=1.0):
        self.levels = {int(level): list(rows) for level, rows in levels.items()}
        if set(self.levels) != {1, 2, 3} or any(not rows for rows in self.levels.values()):
            raise ValueError('Sampling requires nonempty data at each of the three levels.')
        if balance_field not in (None, 'source', 'category', 'language'):
            raise ValueError('balance_field must be source, category, language or null.')
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError('Sampling temperature must be finite and nonnegative.')
        self.balance_field = balance_field
        self.temperature = temperature
        self.rng = random.Random(seed)
        self.identity = corpus_identity(self.levels)
        self.groups = {}
        self.draws = Counter()
        self.level_draws = Counter()
        self.total = 0
        for level, rows in self.levels.items():
            groups = defaultdict(list)
            for index, row in enumerate(rows):
                if row['level'] != level:
                    raise ValueError('Row level does not match its sampling pool.')
                group = str(row.get(balance_field, 'unspecified')) if balance_field else 'all'
                groups[group].append(index)
            self.groups[level] = dict(sorted(groups.items()))

    def sample(self, batch_size, probabilities):
        if batch_size < 1 or not probabilities:
            raise ValueError('Sampling requires a positive batch and active levels.')
        if not set(probabilities) <= self.levels.keys():
            raise ValueError('A probability references an absent level.')
        weights = list(probabilities.values())
        if any(not math.isfinite(weight) or weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError('Level weights must be finite, nonnegative and nonzero.')
        selected = self.rng.choices(list(probabilities), weights=weights, k=batch_size)
        output = []
        for level in selected:
            pools = self.groups[level]
            group_weights = [len(indices) ** self.temperature for indices in pools.values()]
            group = self.rng.choices(list(pools), weights=group_weights, k=1)[0]
            index = self.rng.choice(pools[group])
            row = self.levels[level][index]
            output.append(row)
            self.draws[(level, group)] += 1
            self.level_draws[level] += 1
            self.total += 1
        return output

    def state_dict(self):
        return {'format': 1, 'corpus': self.identity, 'random': self.rng.getstate(),
                'balance_field': self.balance_field, 'temperature': self.temperature,
                'draws': [[level, group, count] for (level, group), count in sorted(self.draws.items())],
                'level_draws': dict(self.level_draws), 'total': self.total}

    def load_state_dict(self, state):
        if state.get('format') != 1 or state.get('corpus') != self.identity:
            raise ValueError('Sampler state belongs to another training corpus.')
        if state['balance_field'] != self.balance_field or state['temperature'] != self.temperature:
            raise ValueError('Sampler policy changed across resume.')
        draws = Counter()
        for level, group, count in state['draws']:
            if level not in self.groups or group not in self.groups[level] or count < 0:
                raise ValueError('Invalid sampling counters.')
            draws[(level, group)] = int(count)
        level_draws = Counter({int(level): int(count) for level, count in state['level_draws'].items()})
        if sum(draws.values()) != state['total'] or sum(level_draws.values()) != state['total']:
            raise ValueError('Sampling counters do not sum to the recorded total.')
        self.rng.setstate(tuple_state(state['random']))
        self.draws, self.level_draws = draws, level_draws
        self.total = int(state['total'])

    def report(self):
        groups = []
        for level, pools in self.groups.items():
            for group, indices in pools.items():
                count = self.draws[(level, group)]
                groups.append({'level': level, 'group': group, 'population': len(indices),
                               'draws': count, 'fraction': count / self.total if self.total else 0})
        return {'corpus': self.identity, 'total_draws': self.total,
                'level_draws': dict(self.level_draws), 'groups': groups}


class SampledExperience:
    def __init__(self, epoch, sampler):
        self.current_experience = epoch
        self.sampler = sampler
        self.records = sampler.levels

    def sample(self, batch_size, curriculum, stage):
        probabilities = ({1: 1 / 3, 2: 1 / 3, 3: 1 / 3} if stage == 'sft'
                         else curriculum.level_probabilities(self.current_experience))
        return self.sampler.sample(batch_size, probabilities)


def expected_draws(sampler, probabilities, draws):
    total = sum(probabilities.values())
    if total <= 0 or draws < 1:
        raise ValueError('Expected counts require positive draws and level weights.')
    result = []
    for level, probability in probabilities.items():
        pools = sampler.groups[level]
        normalizer = sum(len(indices) ** sampler.temperature for indices in pools.values())
        for group, indices in pools.items():
            weight = len(indices) ** sampler.temperature / normalizer
            result.append({'level': level, 'group': group,
                           'expected_draws': draws * probability / total * weight,
                           'expected_per_record': draws * probability / total * weight / len(indices)})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--balance', choices=['source', 'category', 'language'])
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--probabilities', type=float, nargs=3, default=[1, 1, 1])
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    rows = [normalize_record(row) for _, row in read_jsonl(args.input)]
    levels = {level: [row for row in rows if row['level'] == level] for level in (1, 2, 3)}
    sampler = LevelSampler(levels, args.seed, args.balance, args.temperature)
    weights = dict(zip((1, 2, 3), args.probabilities))
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        parser.error('Probabilities must be finite and nonnegative.')
    report = {'seed': args.seed, 'balance_field': args.balance, 'temperature': args.temperature,
              'expected': expected_draws(sampler, weights, args.draws), 'corpus': sampler.identity}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'records': len(rows)}))
