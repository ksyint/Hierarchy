import math

import pytest
import torch
import torch.nn.functional as F

from methods.curriculum import HierarchicalCurriculum
from methods.preference.losses import ccr_loss, harm_dpo_loss
from networks.scoring import sequence_log_probs
from networks.tokenizer import ByteTokenizer
from networks.batching import prepare_batch


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
