import copy
import math
import random

import torch

from benchmarks.formatting import format_pair
from experiments.probe import probe
from methods.curriculum import HierarchicalCurriculum
from networks.scoring import decision_logits, response_log_probs
from .losses import ccr_loss, harm_dpo_loss


class PreferenceLearner:
    """Strategy interface used by the experiment stream: train, eval, then update."""
    stage = None

    def __init__(self, model, tokenizer, optimizer, config, device, total_experiences):
        self.model, self.tokenizer, self.optimizer = model, tokenizer, optimizer
        self.config, self.device = config, device
        self.curriculum = HierarchicalCurriculum(**config['curriculum'])
        self.total_steps = total_experiences * config['steps_per_epoch']
        self.warmup = max(1, int(0.05 * self.total_steps))
        self.losses_by_level = {}
        self.last_train_loss = None

    def score(self, prompts, responses, model=None):
        return response_log_probs(model or self.model, self.tokenizer, prompts, responses,
                                  self.device, self.config['max_length'])

    def prepare(self, rows, epoch):
        pairs = [format_pair(row, random.random() < self.curriculum.hard_probability(row['level'], epoch),
                             random.random() < self.curriculum.explicit_probability(epoch)) for row in rows]
        return tuple(map(list, zip(*pairs)))

    def train(self, experience):
        self.model.train()
        self.losses_by_level = {1: [], 2: [], 3: []}
        losses = []
        epoch = experience.current_experience
        for iteration in range(self.config['steps_per_epoch']):
            rows = experience.sample(self.config['batch_size'], self.curriculum, self.stage)
            loss, plain_dpo = self.objective(rows, epoch)
            if plain_dpo is not None:
                for row, value in zip(rows, plain_dpo.detach().tolist()):
                    self.losses_by_level[row['level']].append(value)
            step = epoch * self.config['steps_per_epoch'] + iteration
            if step < self.warmup:
                scale = (step + 1) / self.warmup
            else:
                progress = (step - self.warmup) / max(1, self.total_steps - self.warmup)
                scale = 0.5 * (1 + math.cos(math.pi * progress))
            for group in self.optimizer.param_groups:
                group['lr'] = self.config['lr'] * scale
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            losses.append(loss.item())
        self.last_train_loss = sum(losses) / len(losses)
        return self.last_train_loss

    def eval(self, records):
        return probe(self.model, self.tokenizer, records, self.config['decision_tokens'],
                     self.device, self.config['max_length'])

    def finish_experience(self, experience, accuracy):
        epoch = experience.current_experience
        if self.stage == 'dpo':
            level_losses = {level: sum(values) / len(values)
                            for level, values in self.losses_by_level.items() if values}
            self.curriculum.update(epoch, level_losses, accuracy)
        return {'epoch': epoch, 'loss': self.last_train_loss, 'probe_accuracy': accuracy,
                'curriculum': self.curriculum.state_dict(),
                'explicit_probability': self.curriculum.explicit_probability(epoch)}


class SFTLearner(PreferenceLearner):
    stage = 'sft'

    def objective(self, rows, epoch):
        prompts, chosen, _ = self.prepare(rows, epoch)
        return -self.score(prompts, chosen).mean(), None


class HARMLearner(PreferenceLearner):
    stage = 'dpo'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference = copy.deepcopy(self.model).eval().requires_grad_(False)

    def objective(self, rows, epoch):
        prompts, chosen, rejected = self.prepare(rows, epoch)
        chosen_logp, rejected_logp = self.score(prompts, chosen), self.score(prompts, rejected)
        with torch.no_grad():
            ref_chosen = self.score(prompts, chosen, self.reference)
            ref_rejected = self.score(prompts, rejected, self.reference)
        margins = torch.tensor([self.curriculum.margin(row['level'], epoch) for row in rows], device=self.device)
        beta = self.config['beta']
        loss = harm_dpo_loss(chosen_logp, rejected_logp, ref_chosen, ref_rejected, margins, beta)
        plain = harm_dpo_loss(chosen_logp, rejected_logp, ref_chosen, ref_rejected, 0, beta, reduction='none')
        counterfactuals = [row for row in rows if 'cf_prompt' in row]
        if counterfactuals:
            loss = loss + self.config['lambda_ccr'] * self.counterfactual_loss(counterfactuals)
        return loss, plain

    def counterfactual_loss(self, rows):
        kwargs = (self.config['decision_tokens'], self.device, self.config['max_length'])
        original = decision_logits(self.model, self.tokenizer, [row['prompt'] for row in rows], *kwargs)
        changed = decision_logits(self.model, self.tokenizer, [row['cf_prompt'] for row in rows], *kwargs)
        labels = torch.tensor([row['decision'] for row in rows], device=self.device)
        changed_labels = torch.tensor([row['cf_decision'] for row in rows], device=self.device)
        return ccr_loss(original, changed, labels, changed_labels)
