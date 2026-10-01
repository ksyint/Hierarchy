import torch

from networks.scoring import decision_logits


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
