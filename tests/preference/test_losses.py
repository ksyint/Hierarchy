import math

import pytest
import torch
import torch.nn.functional as F

from methods.curriculum import HierarchicalCurriculum
from methods.preference.losses import ccr_loss, harm_dpo_loss
from networks.scoring import sequence_log_probs
from networks.tokenizer import ByteTokenizer
from networks.batching import prepare_batch


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
