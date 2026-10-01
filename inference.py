import argparse
import json

import torch

from experiments.checkpoint import restore
from networks.batching import prompt_ids
from networks.factory import load_model
from networks.scoring import decision_logits


def main(args):
    torch.set_num_threads(2)
    if args.checkpoint:
        model, tokenizer, config = restore(args.checkpoint, args.device, local_dir=args.local_model,
                                           cache_dir=args.cache_dir, offline=args.offline)
    else:
        model, tokenizer = load_model(args.model, device=args.device, lora=False, local_dir=args.local_model,
                                       cache_dir=args.cache_dir, offline=args.offline)
        model.eval()
        config = dict(decision_tokens=[' No', ' Yes'], max_length=2048)
    prompt = args.prompt + '\n[WITHOUT_THINKING]\n'
    ids = prompt_ids(tokenizer, prompt)
    if len(ids) + args.max_new_tokens > config['max_length']:
        raise ValueError('Prompt and output budget exceed the configured context length.')
    with torch.no_grad():
        probe = decision_logits(model, tokenizer, [args.prompt], config['decision_tokens'], args.device, config['max_length'])
        input_ids = torch.tensor([ids], device=args.device)
        generated = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                   max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True,
                                   pad_token_id=tokenizer.pad_token_id)[0, len(ids):]
    print(json.dumps({'refusal_probability': probe.softmax(-1)[0, 1].item(),
                      'response': tokenizer.decode(generated, skip_special_tokens=True)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    parser.add_argument('--model', default='qwen3-4b')
    parser.add_argument('--local-model')
    parser.add_argument('--cache-dir', default='.cache/huggingface')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    main(parser.parse_args())
