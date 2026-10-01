import argparse
import json

import torch

from experiments.checkpoint import restore
from networks.scoring import decision_logits


def main(args):
    torch.set_num_threads(2)
    model, tokenizer, config = restore(args.checkpoint, args.device)
    prompt = args.prompt + '\n[WITHOUT_THINKING]\n'
    ids = [tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id]
    ids += tokenizer.encode(prompt, add_special_tokens=False)
    generated = []
    with torch.no_grad():
        probe = decision_logits(model, tokenizer, [args.prompt], config['decision_tokens'], args.device, config['max_length'])
        for _ in range(args.max_new_tokens):
            input_ids = torch.tensor([ids[-config['max_length']:]], device=args.device)
            token = model(input_ids=input_ids).logits[0, -1].argmax().item()
            if token == tokenizer.eos_token_id:
                break
            ids.append(token)
            generated.append(token)
    print(json.dumps({'refusal_probability': probe.softmax(-1)[0, 1].item(),
                      'response': tokenizer.decode(generated, skip_special_tokens=True)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_new_tokens', type=int, default=64)
    main(parser.parse_args())
