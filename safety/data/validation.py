"""Annotation contracts and corpus inventories before curriculum construction."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import unicodedata

from safety.data.korean import canonical_prompt, record_prompts


TEXT_FIELDS = ('prompt', 'chosen', 'rejected_easy', 'rejected_hard')
THINKING_FIELDS = ('chosen_thinking', 'rejected_easy_thinking', 'rejected_hard_thinking')
REQUIRED = set(TEXT_FIELDS) | {'level', 'decision'}


@dataclass(frozen=True)
class Finding:
    line: int
    field: str
    message: str
    severity: str = 'error'


class AnnotationCorpus:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.records = []
        self.findings = []
        self.line_numbers = []
        for number, line in enumerate(self.path.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                self.findings.append(Finding(number, 'record', str(error)))
                continue
            if not isinstance(row, dict):
                self.findings.append(Finding(number, 'record', 'Expected a JSON object.'))
                continue
            self.records.append(row)
            self.line_numbers.append(number)

    @staticmethod
    def identity(row):
        payload = [canonical_prompt(str(row.get('prompt', ''))), str(row.get('source', 'annotated'))]
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode('utf-8')).hexdigest()[:24]

    def add(self, line, field, message, severity='error'):
        self.findings.append(Finding(line, field, message, severity))

    def validate_record(self, row, line, require_counterfactual=False):
        for field in sorted(REQUIRED - row.keys()):
            self.add(line, field, 'Required field is absent.')
        for field in TEXT_FIELDS:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                self.add(line, field, 'Expected nonempty text.')
            elif '\x00' in value:
                self.add(line, field, 'NUL bytes are not valid annotation text.')
        if type(row.get('level')) is not int or row['level'] not in (1, 2, 3):
            self.add(line, 'level', 'Expected integer level 1, 2 or 3.')
        if type(row.get('decision')) is not int or row['decision'] not in (0, 1):
            self.add(line, 'decision', 'Expected integer decision 0 or 1.')
        supplied = ('cf_prompt' in row, 'cf_decision' in row)
        if supplied[0] != supplied[1] or (require_counterfactual and not all(supplied)):
            self.add(line, 'cf_prompt', 'Counterfactual prompt and decision must be supplied together.')
        if supplied[0]:
            text = row['cf_prompt']
            if not isinstance(text, str) or not text.strip():
                self.add(line, 'cf_prompt', 'Expected nonempty counterfactual text.')
            elif isinstance(row.get('prompt'), str) and canonical_prompt(text) == canonical_prompt(row['prompt']):
                self.add(line, 'cf_prompt', 'Counterfactual and primary prompts are identical.')
        if supplied[1] and (type(row['cf_decision']) is not int or row['cf_decision'] not in (0, 1)):
            self.add(line, 'cf_decision', 'Expected integer decision 0 or 1.')
        for field in THINKING_FIELDS:
            if field in row and not isinstance(row[field], str):
                self.add(line, field, 'Reasoning annotations must be strings.')
        if all(isinstance(row.get(field), str) for field in TEXT_FIELDS[1:]):
            responses = [canonical_prompt(row[field]) for field in TEXT_FIELDS[1:]]
            if len(set(responses)) < len(responses):
                self.add(line, 'responses', 'Preference alternatives contain identical text.')
        for field in ('source', 'category', 'language'):
            if field in row and (not isinstance(row[field], str) or not row[field].strip()):
                self.add(line, field, 'Corpus metadata must be nonempty strings.')
        if row.get('review_status') not in (None, 'accepted', 'pending', 'rejected'):
            self.add(line, 'review_status', 'Use accepted, pending or rejected.')

    def validate(self, require_counterfactual=False):
        identifiers = {}
        for line, row in zip(self.line_numbers, self.records):
            self.validate_record(row, line, require_counterfactual)
            identity = str(row.get('id') or self.identity(row))
            if identity in identifiers:
                self.add(line, 'id', f'Duplicate example identity first seen at line {identifiers[identity]}.')
            identifiers[identity] = line
        if not self.records:
            self.add(0, 'record', 'No annotation records were loaded.')
        return self

    def normalized(self):
        invalid = {item.line for item in self.findings if item.severity == 'error'}
        result = []
        for line, original in zip(self.line_numbers, self.records):
            if line in invalid:
                continue
            row = {key: unicodedata.normalize('NFC', value).strip() if isinstance(value, str) else value
                   for key, value in original.items()}
            row.setdefault('id', self.identity(row))
            row.setdefault('language', 'ko')
            result.append(row)
        return result

    def inventory(self):
        counters = {key: Counter() for key in ('level', 'decision', 'source', 'category', 'language', 'review_status')}
        lengths = defaultdict(list)
        counterfactual = Counter()
        prompt_owners = defaultdict(list)
        for line, row in zip(self.line_numbers, self.records):
            for key in counters:
                counters[key][str(row.get(key, 'unspecified'))] += 1
            for field in TEXT_FIELDS + THINKING_FIELDS:
                if isinstance(row.get(field), str):
                    lengths[field].append(len(row[field]))
            if 'cf_decision' in row and 'decision' in row:
                counterfactual['flipped' if row['cf_decision'] != row['decision'] else 'same'] += 1
            if isinstance(row.get('prompt'), str) and isinstance(row.get('cf_prompt', ''), str):
                for prompt in record_prompts(row):
                    prompt_owners[prompt].append(line)
        summaries = {}
        for field, values in lengths.items():
            ordered = sorted(values)
            summaries[field] = {'minimum': ordered[0], 'median': ordered[len(ordered) // 2],
                                'maximum': ordered[-1], 'mean': sum(values) / len(values)}
        return {'examples': len(self.records), 'fields': {key: dict(value) for key, value in counters.items()},
                'character_lengths': summaries, 'counterfactuals': dict(counterfactual),
                'shared_prompt_groups': [lines for lines in prompt_owners.values() if len(lines) > 1],
                'findings': [asdict(item) for item in self.findings]}

    def save(self, destination, accepted_only=False):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        rows = self.normalized()
        if accepted_only:
            rows = [row for row in rows if row.get('review_status') == 'accepted']
        (destination / 'records.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        report = self.inventory()
        report['exported_examples'] = len(rows)
        (destination / 'annotations.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        return report


def annotation_agreement(records, additional_paths):
    panels = [records]
    for path in additional_paths:
        corpus = AnnotationCorpus(path).validate()
        if any(item.severity == 'error' for item in corpus.findings):
            raise ValueError(f'Validate annotation panel before agreement analysis: {path}')
        panels.append(corpus.normalized())
    observations = defaultdict(list)
    prompts = {}
    for panel, rows in enumerate(panels):
        seen = set()
        for row in rows:
            identity = str(row.get('id') or AnnotationCorpus.identity(row))
            if identity in seen:
                raise ValueError('A panel may annotate an example only once.')
            seen.add(identity)
            prompt = canonical_prompt(row['prompt'])
            if identity in prompts and prompts[identity] != prompt:
                raise ValueError('The same annotation identity has different prompt text across panels.')
            prompts[identity] = prompt
            observations[identity].append((panel, row))
    report, disagreements = {}, []
    for field in ('level', 'decision', 'cf_decision'):
        agree = pairs = 0
        pooled = Counter()
        examined = 0
        for identity, values in observations.items():
            labels = [row[field] for _, row in values if field in row]
            if len(labels) < 2:
                continue
            examined += 1
            counts = Counter(labels)
            pooled.update(counts)
            agree += sum(count * (count - 1) for count in counts.values())
            pairs += len(labels) * (len(labels) - 1)
            if len(counts) > 1:
                disagreements.append({'id': identity, 'field': field, 'votes': dict(counts),
                                      'panels': [panel for panel, row in values if field in row]})
        observed = agree / pairs if pairs else None
        total = sum(pooled.values())
        expected = sum((count / total) ** 2 for count in pooled.values()) if total else None
        kappa = (observed - expected) / (1 - expected) if observed is not None and expected < 1 else None
        report[field] = {'examples': examined, 'ordered_pairs': pairs, 'pairwise_agreement': observed,
                         'chance_agreement': expected, 'kappa': kappa, 'labels': dict(pooled)}
    return {'panels': len(panels), 'fields': report, 'disagreements': disagreements}


def tokenization_inventory(records, model_name, local_model, cache, offline, max_length):
    from transformers import AutoTokenizer
    from safety.models.learner import prompt_ids
    from safety.models.backbones import resolve_backbone
    if max_length < 2:
        raise ValueError('Tokenization context must contain at least two tokens.')
    spec = resolve_backbone(model_name)
    options = {'cache_dir': cache, 'local_files_only': offline, 'trust_remote_code': spec.remote_code}
    if not local_model:
        options['revision'] = spec.revision
    tokenizer = AutoTokenizer.from_pretrained(local_model or spec.repo, **options)
    if tokenizer.eos_token_id is None:
        raise ValueError('The selected tokenizer must define an EOS token.')
    rows = []
    totals = Counter()
    for record in records:
        prompts = [('plain', record['prompt'] + '\n[WITHOUT_THINKING]\n'), ('explicit', record['prompt'])]
        for form, prompt in prompts:
            prefix = len(prompt_ids(tokenizer, prompt))
            for field in ('chosen', 'rejected_easy', 'rejected_hard'):
                response = record[field]
                thought = record.get(field + '_thinking')
                if form == 'explicit' and thought:
                    response = f'[THINKING]{thought}[/THINKING]\n{response}'
                suffix = len(tokenizer.encode(response, add_special_tokens=False)) + 1
                oversized = suffix >= max_length
                truncated = max(0, prefix + suffix - max_length)
                totals['sequences'] += 1
                totals['oversized_responses'] += oversized
                totals['prompt_truncations'] += truncated > 0 and not oversized
                rows.append({'id': record.get('id', AnnotationCorpus.identity(record)), 'form': form, 'field': field,
                             'prompt_tokens': prefix, 'response_tokens': suffix, 'total_tokens': prefix + suffix,
                             'prompt_tokens_removed': min(prefix, truncated), 'response_exceeds_context': oversized})
    return {'model': spec.repo, 'revision': spec.revision, 'max_length': max_length,
            'summary': dict(totals), 'sequences': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--require-counterfactual', action='store_true')
    parser.add_argument('--accepted-only', action='store_true')
    parser.add_argument('--schema')
    parser.add_argument('--agreement', action='append', default=[])
    parser.add_argument('--tokenizer')
    parser.add_argument('--local-model')
    parser.add_argument('--cache-dir', default='.cache/huggingface')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--context-length', type=int, default=2048)
    args = parser.parse_args(argv)
    corpus = AnnotationCorpus(args.input).validate(args.require_counterfactual)
    if args.schema:
        from jsonschema import Draft202012Validator
        validator = Draft202012Validator(json.loads(Path(args.schema).read_text()))
        for line, row in zip(corpus.line_numbers, corpus.records):
            for issue in validator.iter_errors(row):
                corpus.add(line, '.'.join(map(str, issue.path)) or 'record', issue.message)
    report = corpus.save(args.output, args.accepted_only)
    if args.agreement:
        agreement = annotation_agreement(corpus.normalized(), args.agreement)
        (Path(args.output) / 'agreement.json').write_text(json.dumps(agreement, ensure_ascii=False, indent=2) + '\n')
    if args.tokenizer:
        inventory = tokenization_inventory(corpus.normalized(), args.tokenizer, args.local_model,
                                            args.cache_dir, args.offline, args.context_length)
        (Path(args.output) / 'tokenization.json').write_text(json.dumps(inventory, indent=2) + '\n')
    print(json.dumps({'examples': report['examples'], 'exported': report['exported_examples'],
                      'findings': len(report['findings'])}))
    if any(item.severity == 'error' for item in corpus.findings):
        raise SystemExit('Annotation validation found records requiring correction.')
