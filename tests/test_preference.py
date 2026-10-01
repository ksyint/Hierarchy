import math
import pytest
import torch
import torch.nn.functional as F
from methods.preference import HierarchicalCurriculum
from methods.preference import ccr_loss, harm_dpo_loss
from methods.preference import sequence_log_probs
from methods.preference import prepare_batch


class ByteTokenizer:
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2

    def encode(self, text, add_special_tokens=False):
        return [b + 3 for b in text.encode('utf-8')]

    def decode(self, tokens, skip_special_tokens=True):
        return bytes([i - 3 for i in tokens if i >= 3]).decode('utf-8', errors='replace')


def test_curriculum_requires_both_gates_and_sequential_unlock():
    hsc = HierarchicalCurriculum(rho=1)
    hsc.update(0, {1: 0.1}, {1: 0.5})
    assert list(hsc.unlock_epoch) == [1]
    hsc.update(1, {1: 0.5}, {1: 1.0})
    assert list(hsc.unlock_epoch) == [1]
    hsc.update(2, {1: 0.1}, {1: 1.0, 2: 1.0})
    assert hsc.unlock_epoch == {1: 0, 2: 3}
    assert 3 not in hsc.unlock_epoch
    assert hsc.margin(2, 3) == 2
    assert hsc.margin(2, 4) < 2

def test_replay_distribution_and_fading_constraint():
    hsc = HierarchicalCurriculum(explicit_start=0, ramps=(1, 1, 8))
    hsc.unlock_epoch = {1: 0, 2: 2, 3: 100}
    p = hsc.level_probabilities(120)
    assert sum(p.values()) == pytest.approx(1)
    assert p[1] + p[2] >= 0.2
    assert hsc.explicit_probability(107) == 1
    assert hsc.explicit_probability(109) < 1


def test_margin_is_outside_beta_and_gradients_favor_chosen():
    chosen = torch.tensor([0.0], requires_grad=True)
    rejected = torch.tensor([0.0], requires_grad=True)
    loss = harm_dpo_loss(chosen, rejected, torch.zeros(1), torch.zeros(1), 2.0, beta=0.1)
    assert loss.item() == pytest.approx(F.softplus(torch.tensor(2.0)).item())
    loss.backward()
    assert chosen.grad < 0 and rejected.grad > 0

def test_reference_cancellation_recovers_log_two():
    p, q = torch.randn(8), torch.randn(8)
    assert harm_dpo_loss(p, q, p, q, 0).item() == pytest.approx(math.log(2))

def test_ccr_rewards_intent_flip_not_invariance():
    y, yf = torch.tensor([0]), torch.tensor([1])
    p = torch.tensor([[10.0, -10.0]])
    flipped = -p
    assert ccr_loss(p, flipped, y, yf) < 1e-5
    assert ccr_loss(p, p, y, yf) > 19
    assert ccr_loss(p, p, y, y).item() == pytest.approx(0)


def test_response_mask_excludes_prompt_and_padding():
    tokenizer = ByteTokenizer()
    ids, attention, mask = prepare_batch(tokenizer, ['long prompt', 'x'], ['A', 'BC'], 'cpu')
    logits = torch.zeros(*ids.shape, 259)
    values = sequence_log_probs(logits, ids, mask)
    assert torch.allclose(values, -math.log(259) * torch.tensor([2.0, 3.0]))
    assert not torch.any(mask[attention == 0])
