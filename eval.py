import argparse
import json

import torch

from benchmarks.records import load_records, synthetic_records
from networks.scoring import decision_logits
from experiments.checkpoint import restore


def main(args):
    torch.set_num_threads(2)
    model, tokenizer, config = restore(args.checkpoint, args.device)
    records = load_records(args.data) if args.data else synthetic_records(24, 2000)
    predictions, labels = [], []
    with torch.no_grad():
        for start in range(0, len(records), 8):
            rows = records[start:start + 8]
            output = decision_logits(model, tokenizer, [r['prompt'] for r in rows], config['decision_tokens'], args.device, config['max_length'])
            predictions.extend(output.argmax(-1).tolist())
            labels.extend(row['decision'] for row in rows)
    safe = [p == 1 for p, y in zip(predictions, labels) if y == 1]
    over = [p == 1 for p, y in zip(predictions, labels) if y == 0]
    print(json.dumps({'decision_safety_rate': sum(safe) / len(safe) if safe else None,
                      'decision_overrefusal_rate': sum(over) / len(over) if over else None,
                      'examples': len(records), 'note': 'One-token decision probe, not response-level human/LLM judgment.'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data')
    parser.add_argument('--device', default='cuda')
    main(parser.parse_args())
