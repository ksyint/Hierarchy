"""Prompt-group split auditing and reproducible curriculum subsets."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

from safety.data.korean import canonical_prompt, group_related_records, load_records, record_prompts, split_groups
from safety.data.validation import AnnotationCorpus


@dataclass(frozen=True)
class SplitSpec:
    name: str
    path: Path

    @classmethod
    def parse(cls, value):
        name, separator, path = value.partition('=')
        if not separator or not name or not path:
            raise ValueError('Use --split NAME=MANIFEST for each partition.')
        return cls(name, Path(path).resolve())


class PartitionAudit:
    def __init__(self, specifications):
        self.specifications = specifications
        names = [spec.name for spec in specifications]
        if len(set(names)) != len(names):
            raise ValueError('Partition names must be unique.')
        self.records = {spec.name: load_records(spec.path) for spec in specifications}
        self.findings = []

    def exact_overlap(self):
        owners = defaultdict(set)
        ids = defaultdict(set)
        for name, records in self.records.items():
            for row in records:
                for prompt in record_prompts(row):
                    owners[prompt].add(name)
                if row.get('id'):
                    ids[str(row['id'])].add(name)
        for prompt, partitions in owners.items():
            if len(partitions) > 1:
                self.findings.append({'kind': 'prompt_overlap', 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                                      'partitions': sorted(partitions)})
        for identity, partitions in ids.items():
            if len(partitions) > 1:
                self.findings.append({'kind': 'example_overlap', 'id': identity, 'partitions': sorted(partitions)})
        return self

    def near_overlap(self, threshold=.85, seed=42):
        combined = []
        for name, records in self.records.items():
            combined.extend(dict(row, _partition=name) for row in records)
        for group in group_related_records(combined, threshold, seed):
            owners = {row['_partition'] for row in group}
            if len(owners) > 1:
                self.findings.append({'kind': 'related_prompt_overlap', 'partitions': sorted(owners),
                                      'examples': [row.get('id', AnnotationCorpus.identity(row)) for row in group]})
        return self

    def coverage(self):
        result = {}
        for name, rows in self.records.items():
            fields = {key: Counter(str(row.get(key, 'unspecified')) for row in rows)
                      for key in ('source', 'category', 'level', 'decision', 'language')}
            levels = {row['level'] for row in rows}
            if levels != {1, 2, 3}:
                self.findings.append({'kind': 'level_coverage', 'partition': name,
                                      'missing_levels': sorted({1, 2, 3} - levels)})
            counterfactuals = [row for row in rows if 'cf_prompt' in row]
            result[name] = {'examples': len(rows), 'strata': {key: dict(value) for key, value in fields.items()},
                            'counterfactual_pairs': len(counterfactuals),
                            'flipped_pairs': sum(row['decision'] != row['cf_decision'] for row in counterfactuals)}
        return result

    def report(self):
        coverage = self.coverage()
        return {'partitions': coverage, 'findings': self.findings,
                'inputs': {spec.name: {'path': str(spec.path), 'sha256': hashlib.sha256(spec.path.read_bytes()).hexdigest()}
                           for spec in self.specifications}}


def deterministic_subset(records, per_stratum, seed, group_threshold=.85):
    if per_stratum < 1:
        raise ValueError('Each requested stratum must have a positive budget.')
    groups = group_related_records(records, group_threshold, seed)
    random.Random(seed).shuffle(groups)
    counts = Counter()
    selected, omitted = [], []
    for group in groups:
        strata = Counter((row['level'], row['decision'], row.get('source', 'annotated')) for row in group)
        deficits = sum(max(0, per_stratum - counts[key]) for key in strata)
        if deficits:
            selected.extend(group)
            counts.update(strata)
        else:
            omitted.extend(group)
    if {row['level'] for row in selected} != {1, 2, 3}:
        raise ValueError('The selected pool does not cover all three levels.')
    return selected, omitted, counts


def write_records(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def create_splits(input_path, output, seed, threshold):
    rows = load_records(input_path)
    groups = group_related_records(rows, threshold, seed)
    result = split_groups(groups, seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    specifications = []
    for name, records in result.items():
        path = output / f'{name}.jsonl'
        write_records(path, records)
        specifications.append(SplitSpec(name, path))
    report = PartitionAudit(specifications).exact_overlap().report()
    report.update(seed=seed, threshold=threshold, independent_groups=len(groups))
    (output / 'partitions.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def export_language(records, language, output):
    rows = [row for row in records if row.get('language', 'ko') == language]
    if not rows:
        raise ValueError(f'No records match language {language}.')
    for row in rows:
        if not canonical_prompt(row['prompt']):
            raise ValueError('Language export contains an empty prompt.')
    write_records(output, rows)
    return {'language': language, 'examples': len(rows), 'output': str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest='operation', required=True)
    audit = operations.add_parser('audit')
    audit.add_argument('--split', action='append', required=True)
    audit.add_argument('--near', action='store_true')
    audit.add_argument('--threshold', type=float, default=.85)
    audit.add_argument('--seed', type=int, default=42)
    audit.add_argument('--output', required=True)
    build = operations.add_parser('build')
    build.add_argument('--input', required=True)
    build.add_argument('--output', required=True)
    build.add_argument('--threshold', type=float, default=.85)
    build.add_argument('--seed', type=int, default=42)
    subset = operations.add_parser('subset')
    subset.add_argument('--input', required=True)
    subset.add_argument('--output', required=True)
    subset.add_argument('--per-stratum', type=int, required=True)
    subset.add_argument('--seed', type=int, default=42)
    subset.add_argument('--threshold', type=float, default=.85)
    language = operations.add_parser('language')
    language.add_argument('--input', required=True)
    language.add_argument('--language', required=True)
    language.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.operation == 'audit':
        audit = PartitionAudit([SplitSpec.parse(value) for value in args.split]).exact_overlap()
        if args.near:
            audit.near_overlap(args.threshold, args.seed)
        result = audit.report()
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    elif args.operation == 'build':
        result = create_splits(args.input, args.output, args.seed, args.threshold)
    elif args.operation == 'subset':
        selected, omitted, counts = deterministic_subset(load_records(args.input), args.per_stratum, args.seed, args.threshold)
        output = Path(args.output)
        write_records(output / 'selected.jsonl', selected)
        write_records(output / 'omitted.jsonl', omitted)
        result = {'selected': len(selected), 'omitted': len(omitted),
                  'strata': [{'level': key[0], 'decision': key[1], 'source': key[2], 'examples': value}
                             for key, value in sorted(counts.items())]}
        (output / 'subset.json').write_text(json.dumps(result, indent=2) + '\n')
    else:
        result = export_language(load_records(args.input), args.language, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get('findings'):
        raise SystemExit('Partition audit found overlap or missing level coverage.')
