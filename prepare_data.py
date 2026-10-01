"""Normalize annotated Korean records and create disjoint curriculum splits."""
import argparse
import json
from pathlib import Path
import unicodedata

from benchmarks.records import load_records
from benchmarks.splitting import group_related_records, shingles, split_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Annotated UTF-8 JSONL with the training record schema.')
    parser.add_argument('--output', default='data/korean')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold', type=float, default=.85)
    args = parser.parse_args()
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


if __name__ == '__main__':
    main()
