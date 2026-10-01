"""Build candidate preference responses with the three released Qwen teachers."""
import argparse
import gc
import json
from pathlib import Path

import torch

from networks.factory import cuda_device, load_model
from networks.pretrained import resolve_backbone


ROLES = {
    'chosen': ('teacher-strong', 'Give a helpful and safe response. Refuse harmful intent and assist benign intent.'),
    'rejected_hard': ('teacher-medium', 'Construct the rejected alternative for a safety preference dataset. '
                      'Make its comply/refuse decision opposite to the supplied reviewed decision.'),
    'rejected_easy': ('teacher-weak', 'Respond with a short, generic refusal, without topic-specific assistance.'),
}


def collect(rows, field, args, device):
    alias, instruction = ROLES[field]
    local = str(Path(args.teacher_root) / alias) if args.teacher_root else None
    model, tokenizer = load_model(alias, device=device, lora=False, local_dir=local,
                                   cache_dir=args.cache_dir, offline=args.offline,
                                   dtype='float16' if field == 'rejected_easy' else 'bfloat16')
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        for row in rows:
            decision = 'refuse' if row['decision'] else 'comply'
            messages = [{'role': 'system', 'content': instruction},
                        {'role': 'user', 'content': f"Reviewed decision: {decision}\nRequest: {row['prompt']}"}]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=False, return_tensors='pt').to(device)
            if ids.shape[1] + args.max_new_tokens > 8192:
                raise ValueError('A teacher prompt exceeds the configured 8,192-token generation budget.')
            result = model.generate(ids, attention_mask=torch.ones_like(ids), use_cache=True,
                                     max_new_tokens=args.max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id)
            row[field] = tokenizer.decode(result[0, ids.shape[1]:], skip_special_tokens=True).strip()
            if not row[field]:
                raise ValueError(f'Teacher {alias} returned an empty response.')
            row.setdefault('teachers', {})[field] = resolve_backbone(alias).repo
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', required=True, help='JSONL with prompt, level, decision and optional reviewed counterfactuals.')
    parser.add_argument('--output', default='data/raw/candidates.jsonl')
    parser.add_argument('--teacher-root', help='Local directory containing teacher-strong/, teacher-medium/, teacher-weak/.')
    parser.add_argument('--cache-dir', default='.cache/huggingface')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-new-tokens', type=int, default=512)
    args = parser.parse_args()
    device = cuda_device(args.device)
    rows = [json.loads(line) for line in Path(args.seeds).read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(row.get('prompt'), str) or not row['prompt'].strip() or
                       row.get('level') not in (1, 2, 3) or row.get('decision') not in (0, 1) for row in rows):
        raise ValueError('Each seed needs nonempty prompt, level 1..3 and decision 0/1.')
    for field in ROLES:
        collect(rows, field, args, device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    print(json.dumps({'candidates': len(rows), 'output': str(output)}))


if __name__ == '__main__':
    main()
