import math

import pytest
import torch
import torch.nn.functional as F

from methods.curriculum import HierarchicalCurriculum
from methods.preference.losses import ccr_loss, harm_dpo_loss
from networks.scoring import sequence_log_probs
from networks.tokenizer import ByteTokenizer
from networks.batching import prepare_batch


def test_response_mask_excludes_prompt_and_padding():
    tokenizer = ByteTokenizer()
    ids, attention, mask = prepare_batch(tokenizer, ['long prompt', 'x'], ['A', 'BC'], 'cpu')
    logits = torch.zeros(*ids.shape, 259)
    values = sequence_log_probs(logits, ids, mask)
    assert torch.allclose(values, -math.log(259) * torch.tensor([2.0, 3.0]))
    assert not torch.any(mask[attention == 0])
