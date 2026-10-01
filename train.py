import argparse
import copy
import json
import math
import random
from pathlib import Path

import torch
import yaml

from utils.curriculum import HierarchicalCurriculum
from utils.data import format_pair, load_records, synthetic_records
from utils.losses import ccr_loss, harm_dpo_loss
from utils.models import decision_logits, load_model, response_log_probs


def probe(model, tokenizer, records, tokens, device, max_length):
    was_training = model.training
    model.eval()
    results = {}
    with torch.no_grad():
        for level in (1, 2, 3):
            rows = [r for r in records if r['level'] == level]
            if rows:
                correct = []
                for start in range(0, len(rows), 8):
                    chunk = rows[start:start + 8]
                    logits = decision_logits(model, tokenizer, [r['prompt'] for r in chunk], tokens, device, max_length)
                    target = torch.tensor([r['decision'] for r in chunk], device=device)
                    correct.extend((logits.argmax(-1) == target).float().tolist())
                results[level] = sum(correct) / len(correct)
    model.train(was_training)
    return results


def main(args):
    config = yaml.safe_load(Path(args.config).read_text())
    seed = args.seed if args.seed is not None else config['seed']
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(args.threads)
    model, tokenizer = load_model(args.model, config['model_dim'], args.device, args.lora)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=True)['model'])
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    train = load_records(args.data) if args.data else synthetic_records()
    if args.data and not args.validation:
        raise ValueError('Real-data training requires a disjoint --validation JSONL for competence gates.')
    validation = load_records(args.validation) if args.validation else synthetic_records(24, 1000)
    if {row['level'] for row in validation} != {1, 2, 3}:
        raise ValueError('Validation data must contain all three levels for competence gates.')
    if set(row['prompt'] for row in train) & set(row['prompt'] for row in validation):
        raise ValueError('Training and validation prompts must be disjoint.')
    levels = {level: [r for r in train if r['level'] == level] for level in (1, 2, 3)}
    if any(not rows for rows in levels.values()):
        raise ValueError('Training data must contain all three levels.')
    curriculum = HierarchicalCurriculum(**config['curriculum'])
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config['lr'], weight_decay=config['weight_decay'])
    epochs = args.epochs or config['epochs']
    steps_per_epoch = config['steps_per_epoch']
    total_steps = epochs * steps_per_epoch
    warmup = max(1, int(0.05 * total_steps))
    logs = []
    for epoch in range(epochs):
        model.train()
        by_level = {1: [], 2: [], 3: []}
        epoch_loss = []
        for iteration in range(steps_per_epoch):
            probabilities = ({1: 1 / 3, 2: 1 / 3, 3: 1 / 3} if args.stage == 'sft'
                             else curriculum.level_probabilities(epoch))
            available = list(probabilities)
            selected = random.choices(available, weights=list(probabilities.values()), k=config['batch_size'])
            rows = [random.choice(levels[level]) for level in selected]
            pairs = [format_pair(row, random.random() < curriculum.hard_probability(row['level'], epoch),
                                 random.random() < curriculum.explicit_probability(epoch)) for row in rows]
            prompts, chosen, rejected = map(list, zip(*pairs))
            chosen_logp = response_log_probs(model, tokenizer, prompts, chosen, args.device, config['max_length'])
            if args.stage == 'sft':
                loss = -chosen_logp.mean()
            else:
                rejected_logp = response_log_probs(model, tokenizer, prompts, rejected, args.device, config['max_length'])
                with torch.no_grad():
                    ref_chosen = response_log_probs(reference, tokenizer, prompts, chosen, args.device, config['max_length'])
                    ref_rejected = response_log_probs(reference, tokenizer, prompts, rejected, args.device, config['max_length'])
                margins = torch.tensor([curriculum.margin(level, epoch) for level in selected], device=args.device)
                losses = harm_dpo_loss(chosen_logp, rejected_logp, ref_chosen, ref_rejected,
                                       margins, config['beta'], reduction='none')
                plain_losses = harm_dpo_loss(chosen_logp, rejected_logp, ref_chosen, ref_rejected,
                                             0, config['beta'], reduction='none')
                for level, value in zip(selected, plain_losses.detach().tolist()):
                    by_level[level].append(value)
                loss = losses.mean()
                cf_rows = [row for row in rows if 'cf_prompt' in row]
                if cf_rows:
                    p = decision_logits(model, tokenizer, [r['prompt'] for r in cf_rows],
                                        config['decision_tokens'], args.device, config['max_length'])
                    q = decision_logits(model, tokenizer, [r['cf_prompt'] for r in cf_rows],
                                        config['decision_tokens'], args.device, config['max_length'])
                    y = torch.tensor([r['decision'] for r in cf_rows], device=args.device)
                    y_cf = torch.tensor([r['cf_decision'] for r in cf_rows], device=args.device)
                    loss = loss + config['lambda_ccr'] * ccr_loss(p, q, y, y_cf)
            step = epoch * steps_per_epoch + iteration
            factor = (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
            for group in optimizer.param_groups:
                group['lr'] = config['lr'] * factor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss.append(loss.item())
        accuracies = probe(model, tokenizer, validation, config['decision_tokens'], args.device, config['max_length'])
        if args.stage == 'dpo':
            curriculum.update(epoch, {k: sum(v) / len(v) for k, v in by_level.items() if v}, accuracies)
        log = {'epoch': epoch, 'loss': sum(epoch_loss) / len(epoch_loss), 'probe_accuracy': accuracies,
               'curriculum': curriculum.state_dict(), 'explicit_probability': curriculum.explicit_probability(epoch)}
        logs.append(log)
        print(json.dumps(log))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': config, 'model_name': args.model,
                'lora': args.lora, 'curriculum': curriculum.state_dict()}, output / 'last.pt')
    if args.model:
        model.save_pretrained(output / 'pretrained')
        tokenizer.save_pretrained(output / 'pretrained')
    (output / 'metrics.json').write_text(json.dumps(logs, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/smoke.yaml')
    parser.add_argument('--data')
    parser.add_argument('--validation')
    parser.add_argument('--model', help='Hugging Face model identifier or local full-model directory.')
    parser.add_argument('--checkpoint', help='Matching architecture last.pt initialization (e.g. SFT output).')
    parser.add_argument('--stage', choices=['sft', 'dpo'], default='dpo')
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', default='outputs/smoke')
    main(parser.parse_args())
