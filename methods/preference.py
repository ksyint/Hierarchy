"""Pretrained policy loading and hierarchy-aware preference learning."""
import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import re

import torch
import torch.nn.functional as F

from benchmarks.korean import format_pair


@dataclass(frozen=True)
class Backbone:
    repo: str
    revision: str
    attention: str = 'sdpa'
    remote_code: bool = False
    multimodal: bool = False


BACKBONES = {
    'qwen3-1.7b': Backbone('Qwen/Qwen3-1.7B', '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'),
    'qwen3-4b': Backbone('Qwen/Qwen3-4B', '1cfa9a7208912126459214e8b04321603b3df60c'),
    'kanana-2.1b': Backbone('kakaocorp/kanana-nano-2.1b-base', '3d1ace1214dbebf52b9d2cd6f18b78857194d72b'),
    'exaone-2.4b': Backbone('LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct', 'ccce25bd39c141fe053e0bc75818a8f5fe962802', 'eager', True),
    'gemma3-4b': Backbone('google/gemma-3-4b-it', '093f9f388b31de276ce2de164bdc2081324b9767', multimodal=True),
    'teacher-strong': Backbone('Qwen/Qwen3-32B', '9216db5781bf21249d130ec9da846c4624c16137'),
    'teacher-medium': Backbone('Qwen/Qwen3-14B', '40c069824f4251a91eefaf281ebe4c544efd3e18'),
    'teacher-weak': Backbone('Qwen/Qwen3-8B-AWQ', '4da05a8edb55c6046cce958586c33b61da07bb79'),
}


def resolve_backbone(name='qwen3-4b'):
    name = name or 'qwen3-4b'
    if name in BACKBONES:
        return BACKBONES[name]
    for spec in BACKBONES.values():
        if spec.repo == name:
            return spec
    return Backbone(name, 'main')


def download(name, cache_dir='.cache/huggingface', destination=None, offline=False):
    from huggingface_hub import snapshot_download
    spec = resolve_backbone(name)
    try:
        return snapshot_download(spec.repo, revision=spec.revision, cache_dir=cache_dir,
                                 local_dir=destination, local_files_only=offline,
                                 allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt',
                                                 '*.jinja', '*.py', 'tokenizer.*'])
    except OSError as error:
        folder = destination or f'checkpoints/{name}'
        raise RuntimeError(
            f'Get the complete checkpoint from https://huggingface.co/{spec.repo}/tree/{spec.revision} '
            f'into {folder}, then pass --local-model {folder}. '
            'For gated repositories accept the model terms and run hf auth login first.'
        ) from error


def command_download(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='qwen3-4b')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--cache-dir', default='.cache/huggingface')
    parser.add_argument('--destination')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps({key: asdict(value) for key, value in BACKBONES.items()}, indent=2))
        return
    print(download(args.model, args.cache_dir, args.destination, args.offline))


def cuda_device(device):
    result = torch.device(device)
    if result.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Language-model execution requires CUDA-enabled PyTorch and an accessible GPU.')
    return result


def load_model(model_name='qwen3-4b', dim=None, device='cuda', lora=True, *,
               local_dir=None, cache_dir='.cache/huggingface', offline=False,
               attention=None, gradient_checkpointing=False, dtype='bfloat16'):
    from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForConditionalGeneration
    device = cuda_device(device)
    spec = resolve_backbone(model_name)
    source = local_dir or spec.repo
    if local_dir and not (Path(local_dir) / 'config.json').is_file():
        raise FileNotFoundError(f'{local_dir}/config.json is required; preserve the full downloaded model directory.')
    common = dict(cache_dir=cache_dir, local_files_only=offline, trust_remote_code=spec.remote_code)
    if not local_dir and not Path(source).is_dir():
        common['revision'] = spec.revision
    precision = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}[dtype]
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, **common)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_class = Gemma3ForConditionalGeneration if spec.multimodal else AutoModelForCausalLM
        model = model_class.from_pretrained(source, torch_dtype=precision, low_cpu_mem_usage=True,
                                             device_map={'': str(device)},
                                             attn_implementation=attention or spec.attention, **common)
    except OSError as error:
        raise RuntimeError(f'Cannot load {source}. Download https://huggingface.co/{spec.repo}/tree/{spec.revision} '
                           f'to checkpoints/{model_name} and use --local-model checkpoints/{model_name}; '
                           'gated models require accepted terms and hf auth login.') from error
    model.config.use_cache = False
    if lora:
        from peft import LoraConfig, get_peft_model
        leaves = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'out_proj',
                  'gate_proj', 'up_proj', 'down_proj', 'c_fc', 'c_proj'}
        targets = [name for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear) and name.rsplit('.', 1)[-1] in leaves
                   and 'vision_tower' not in name and 'multi_modal_projector' not in name]
        if spec.multimodal:
            targets.extend(name for name, module in model.named_modules()
                           if isinstance(module, torch.nn.Linear) and name.endswith('lm_head'))
        if not targets:
            raise ValueError('No attention/MLP projection was found for LoRA.')
        pattern = '(?:' + '|'.join(re.escape(name) for name in targets) + ')'
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=.05,
                                                task_type='CAUSAL_LM', target_modules=pattern))
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    return model, tokenizer


def prompt_ids(tokenizer, prompt):
    if getattr(tokenizer, 'chat_template', None):
        return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                             tokenize=True, add_generation_prompt=True, enable_thinking=False)
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    return [bos] + tokenizer.encode(prompt, add_special_tokens=False)


def prepare_batch(tokenizer, prompts, responses, device, max_length=2048):
    sequences, masks = [], []
    bos = tokenizer.bos_token_id
    if bos is None:
        bos = tokenizer.eos_token_id
    if bos is None or tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer must define EOS, PAD, and an initial token.')
    for prompt, response in zip(prompts, responses):
        prefix = prompt_ids(tokenizer, prompt)
        suffix = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
        if len(suffix) >= max_length:
            raise ValueError('Response exceeds max_length; raise limit instead of silently truncating preference data.')
        prefix = prefix[-(max_length - len(suffix)):]
        sequences.append(prefix + suffix)
        masks.append([0] * len(prefix) + [1] * len(suffix))
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    attention = torch.zeros_like(ids)
    for i, (seq, response_mask) in enumerate(zip(sequences, masks)):
        ids[i, :len(seq)] = torch.tensor(seq, device=device)
        mask[i, :len(seq)] = torch.tensor(response_mask, device=device)
        attention[i, :len(seq)] = 1
    return ids, attention, mask


def sequence_log_probs(logits, input_ids, completion_mask):
    """Sum only response-token log likelihood, with the autoregressive shift."""
    if logits.shape[:2] != input_ids.shape or completion_mask.shape != input_ids.shape:
        raise ValueError('Logits, token IDs and completion masks have incompatible shapes.')
    scores = []
    for start in range(0, input_ids.shape[1] - 1, 128):
        stop = min(start + 128, input_ids.shape[1] - 1)
        logp = logits[:, start:stop].float().log_softmax(-1)
        selected = logp.gather(-1, input_ids[:, start + 1:stop + 1, None]).squeeze(-1)
        scores.append((selected * completion_mask[:, start + 1:stop + 1]).sum(-1))
    return torch.stack(scores).sum(0)


def response_log_probs(model, tokenizer, prompts, responses, device, max_length=2048):
    ids, attention, mask = prepare_batch(tokenizer, prompts, responses, device, max_length)
    output = model(input_ids=ids, attention_mask=attention)
    return sequence_log_probs(output.logits, ids, mask)


def decision_logits(model, tokenizer, prompts, decision_tokens, device, max_length=2048):
    probe_prompts = [prompt + '\nRefuse? ' for prompt in prompts]
    values = []
    for verbalizer in decision_tokens:
        ids, attention, mask = prepare_batch(tokenizer, probe_prompts, [verbalizer] * len(prompts), device, max_length)
        mask[torch.arange(len(prompts), device=device), attention.sum(-1) - 1] = 0
        values.append(sequence_log_probs(model(input_ids=ids, attention_mask=attention).logits, ids, mask))
    return torch.stack(values, dim=-1)


class HierarchicalCurriculum:
    def __init__(self, rho=0.1, thresholds=(0.35, 0.30), probe_threshold=0.85,
                 gamma0=2.0, kappa=0.08, ramps=(8, 8, 8), lower_replay=0.2,
                 explicit_start=48, explicit_fade=32):
        if not 0 < rho <= 1 or not 0 <= lower_replay < 1:
            raise ValueError('Invalid EMA rate or replay floor.')
        if len(ramps) != 3 or min(ramps) <= 0 or explicit_fade <= 0:
            raise ValueError('Three positive ramp lengths and a positive fade length are required.')
        self.rho, self.thresholds, self.probe_threshold = rho, thresholds, probe_threshold
        self.gamma0, self.kappa, self.ramps = gamma0, kappa, ramps
        self.lower_replay = lower_replay
        self.explicit_start, self.explicit_fade = explicit_start, explicit_fade
        self.unlock_epoch = {1: 0}
        self.ema = {}

    def update(self, epoch, level_losses, probe_accuracy):
        # Unlock at most one level after an epoch, starting it in the next epoch.
        for level, loss in level_losses.items():
            if level in self.unlock_epoch:
                previous = self.ema.get(level, float(loss))
                self.ema[level] = (1 - self.rho) * previous + self.rho * float(loss)
        highest = max(self.unlock_epoch)
        if (highest < 3 and self.ema.get(highest, math.inf) < self.thresholds[highest - 1]
                and probe_accuracy.get(highest, 0) > self.probe_threshold):
            self.unlock_epoch[highest + 1] = epoch + 1

    def margin(self, level, epoch):
        if level not in self.unlock_epoch:
            raise ValueError('Cannot train a locked level.')
        return self.gamma0 * math.exp(-self.kappa * max(0, epoch - self.unlock_epoch[level]))

    def hard_probability(self, level, epoch):
        if level not in self.unlock_epoch:
            return 0.0
        return min(1.0, max(0.0, (epoch - self.unlock_epoch[level]) / self.ramps[level - 1]))

    def level_probabilities(self, epoch):
        available = sorted(level for level, start in self.unlock_epoch.items() if start <= epoch)
        # Increase the newest level's logit during its hard-negative ramp.
        logits = torch.tensor([(level - 1) * self.hard_probability(level, epoch) for level in available])
        probabilities = logits.softmax(0)
        if len(available) > 1 and probabilities[:-1].sum() < self.lower_replay:
            probabilities[:-1] *= self.lower_replay / probabilities[:-1].sum()
            probabilities[-1] = 1 - self.lower_replay
        return dict(zip(available, probabilities.tolist()))

    def explicit_probability(self, epoch):
        if 3 not in self.unlock_epoch:
            return 1.0
        start = max(self.explicit_start, self.unlock_epoch[3] + self.ramps[2])
        return max(0.0, 1 - max(0, epoch - start) / self.explicit_fade)

    def state_dict(self):
        return {'unlock_epoch': self.unlock_epoch.copy(), 'ema': self.ema.copy()}

    def load_state_dict(self, state):
        self.unlock_epoch = {int(k): v for k, v in state['unlock_epoch'].items()}
        self.ema = {int(k): v for k, v in state['ema'].items()}


def harm_dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected,
                  margin, beta=0.1, reduction='mean'):
    """Eq. (5). gamma shifts beta * log-ratio gap (not the unscaled gap)."""
    gap = (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
    loss = -F.logsigmoid(beta * gap - margin)
    if reduction == 'none':
        return loss
    if reduction != 'mean':
        raise ValueError('reduction must be mean or none')
    return loss.mean()


def ccr_loss(logits, counterfactual_logits, labels, counterfactual_labels):
    """Eq. (7): paired decision NLL on flipped intent, symmetric KL otherwise.

    Decision distributions have class order [comply, refuse]. For same-label
    pairs return 1/2 symmetric-KL, giving lambda/2 scaling in the total loss.
    """
    log_p = logits.log_softmax(-1)
    log_q = counterfactual_logits.log_softmax(-1)
    labels, counterfactual_labels = labels.long(), counterfactual_labels.long()
    paired_nll = F.nll_loss(log_p, labels, reduction='none') + F.nll_loss(log_q, counterfactual_labels, reduction='none')
    symmetric_kl = 0.5 * ((log_p.exp() * (log_p - log_q)).sum(-1) +
                          (log_q.exp() * (log_q - log_p)).sum(-1))
    return torch.where(labels != counterfactual_labels, paired_nll, symmetric_kl).mean()


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
            step = epoch * self.config['steps_per_epoch'] + iteration
            if step < self.warmup:
                scale = (step + 1) / self.warmup
            else:
                progress = (step - self.warmup) / max(1, self.total_steps - self.warmup)
                scale = 0.5 * (1 + math.cos(math.pi * progress))
            for group in self.optimizer.param_groups:
                group['lr'] = self.config['lr'] * scale
            self.optimizer.zero_grad(set_to_none=True)
            accumulation = self.config.get('gradient_accumulation', 1)
            accumulated_loss = 0
            for _ in range(accumulation):
                rows = experience.sample(self.config['batch_size'], self.curriculum, self.stage)
                loss, plain_dpo = self.objective(rows, epoch)
                if plain_dpo is not None:
                    for row, value in zip(rows, plain_dpo.detach().tolist()):
                        self.losses_by_level[row['level']].append(value)
                (loss / accumulation).backward()
                accumulated_loss += loss.detach().item() / accumulation
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            losses.append(accumulated_loss)
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


def save_experiment(strategy, args, history):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    strategy.model.save_pretrained(output / 'pretrained', safe_serialization=True)
    strategy.tokenizer.save_pretrained(output / 'pretrained')
    torch.save({'format': 2, 'config': strategy.config, 'pretrained': 'pretrained',
                'curriculum': strategy.curriculum.state_dict()}, output / 'last.pt')
    (output / 'metrics.json').write_text(json.dumps(history, indent=2) + '\n')


def restore(path, device, trainable=False, local_dir=None, cache_dir=None, offline=False):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if checkpoint.get('format') != 2:
        raise ValueError('Use a pretrained-backbone checkpoint produced by the current training entrypoint.')
    options = dict(config['pretrained'])
    options.update(local_dir=local_dir or options.get('local_dir'), offline=offline,
                   cache_dir=cache_dir or options.get('cache_dir', '.cache/huggingface'))
    adapter_dir = Path(path).resolve().parent / checkpoint['pretrained']
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)
    if options['lora']:
        from peft import PeftModel
        base, _ = load_model(device=device, **{**options, 'lora': False, 'gradient_checkpointing': False})
        model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=trainable)
    else:
        model, _ = load_model(device=device, **{**options, 'local_dir': str(adapter_dir), 'lora': False})
    config['pretrained'].update({key: options[key] for key in ('local_dir', 'cache_dir', 'offline')})
    model.train(trainable)
    return model, tokenizer, config
