"""Held-out decision likelihoods, counterfactual consistency and grouped measurements."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from safety.data.korean import load_records
from safety.models.learner import decision_logits, prompt_ids
from safety.models.backbones import restore
from safety.data.curation.validation import AnnotationCorpus

from safety.evaluation.reports import (
    validate_predictions, grouped_report, compare_predictions, decision_curve,
    choose_threshold, apply_threshold, language_comparisons,
)


@dataclass(frozen=True)
class DecisionOptions:
    batch_size: int = 1
    max_new_tokens: int = 256
    generate_responses: bool = False
    device: str = 'cuda'

    def validate(self):
        if self.batch_size < 1 or self.max_new_tokens < 1:
            raise ValueError('Batch size and generation budget must be positive.')
        if not self.device.startswith('cuda'):
            raise ValueError('Decision-model evaluation requires CUDA.')


class DecisionEvaluator:
    def __init__(self, checkpoint, options, local_model=None, cache_dir=None, offline=False):
        options.validate()
        self.options = options
        self.model, self.tokenizer, self.config = restore(checkpoint, options.device,
            local_dir=local_model, cache_dir=cache_dir, offline=offline)
        self.model.eval()

    def probabilities(self, prompts):
        logits = decision_logits(self.model, self.tokenizer, prompts,
            self.config['decision_tokens'], self.options.device, self.config['max_length'])
        return logits.float().softmax(-1)[:, 1].tolist()

    def response(self, prompt):
        prompt += '\n[WITHOUT_THINKING]\n'
        ids = prompt_ids(self.tokenizer, prompt)
        if len(ids) + self.options.max_new_tokens > self.config['max_length']:
            raise ValueError('Prompt plus generation budget exceeds the saved context length.')
        values = torch.tensor([ids], device=self.options.device)
        output = self.model.generate(input_ids=values, attention_mask=torch.ones_like(values),
            max_new_tokens=self.options.max_new_tokens, do_sample=False, use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.decode(output[0, len(ids):], skip_special_tokens=True)

    def evaluate(self, records):
        with torch.inference_mode():
            for start in range(0, len(records), self.options.batch_size):
                batch = records[start:start + self.options.batch_size]
                probabilities = self.probabilities([row['prompt'] for row in batch])
                changed = [(index, row) for index, row in enumerate(batch) if 'cf_prompt' in row]
                cf_probabilities = self.probabilities([row['cf_prompt'] for _, row in changed]) if changed else []
                counterfactual = {index: value for (index, _), value in zip(changed, cf_probabilities)}
                for index, (row, probability) in enumerate(zip(batch, probabilities)):
                    result = {key: row[key] for key in ('source', 'category', 'level', 'language', 'decision') if key in row}
                    result.update(id=row.get('id', AnnotationCorpus.identity(row)),
                                  refusal_probability=probability, prediction=int(probability >= .5))
                    if index in counterfactual:
                        value = counterfactual[index]
                        result.update(cf_decision=row['cf_decision'], cf_refusal_probability=value,
                                      cf_prediction=int(value >= .5))
                    if self.options.generate_responses:
                        result['response'] = self.response(row['prompt'])
                    yield result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    parser.add_argument('--data')
    parser.add_argument('--predictions')
    parser.add_argument('--compare')
    parser.add_argument('--output', required=True)
    parser.add_argument('--local-model')
    parser.add_argument('--cache-dir', default='.cache/huggingface')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--responses', action='store_true')
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--calibration')
    parser.add_argument('--target-safety', type=float)
    parser.add_argument('--maximum-overrefusal', type=float)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.predictions:
        rows = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
    else:
        if not args.checkpoint or not args.data:
            parser.error('Use --predictions or both --checkpoint and --data.')
        evaluator = DecisionEvaluator(args.checkpoint, DecisionOptions(args.batch_size, args.max_new_tokens, args.responses, args.device),
                                      args.local_model, args.cache_dir, args.offline)
        rows = []
        with (output / 'predictions.jsonl').open('w') as stream:
            for row in evaluator.evaluate(load_records(args.data)):
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
    if args.threshold is not None and args.calibration:
        parser.error('Select a fixed threshold or separate calibration predictions.')
    threshold = args.threshold if args.threshold is not None else .5
    selected = None
    if args.calibration:
        calibration_rows = [json.loads(line) for line in Path(args.calibration).read_text().splitlines() if line.strip()]
        if {row['id'] for row in rows} & {row['id'] for row in calibration_rows}:
            raise ValueError('Calibration and evaluation example IDs must be disjoint.')
        selected, curve = choose_threshold(calibration_rows, args.target_safety, args.maximum_overrefusal)
        threshold = selected['threshold']
        (output / 'calibration.json').write_text(json.dumps(curve, indent=2) + '\n')
    elif args.target_safety is not None or args.maximum_overrefusal is not None:
        parser.error('Decision-rate constraints require separate --calibration predictions.')
    rows = apply_threshold(validate_predictions(rows), threshold)
    result = grouped_report(rows)
    result.update(threshold=threshold, calibration_selection=selected,
                  discrimination=decision_curve(rows), language_comparisons=language_comparisons(result['groups']))
    (output / 'scored-predictions.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    if args.compare:
        other = [json.loads(line) for line in Path(args.compare).read_text().splitlines() if line.strip()]
        result['comparison'] = compare_predictions(rows, other)
    (output / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['overall'], ensure_ascii=False, indent=2))
