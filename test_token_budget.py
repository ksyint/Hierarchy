import sys
from types import SimpleNamespace

import pytest

from safety.data.korean import format_pair
from safety.data.schema import read_jsonl
from safety.data.tokenization import cache_tokens, prefix_tokens, sequence_contract
from safety.data.validation import tokenization_inventory
from safety.models.learner import prepare_batch
from test_preference import ByteTokenizer


class ChatTokenizer(ByteTokenizer):
    chat_template = 'stub'

    def apply_chat_template(self, messages, **kwargs):
        return [self.bos_token_id, 4] + self.encode(messages[0]['content']) + [5, 6]


@pytest.mark.parametrize('tokenizer', [ByteTokenizer(), ChatTokenizer()])
def test_preference_alternatives_keep_identical_complete_context(tokenizer):
    prompt = '0123456789'
    prefix = prefix_tokens(tokenizer, prompt)
    budget = len(prefix) + 2
    ids, attention, mask = prepare_batch(tokenizer, [prompt], ['A'], 'cpu', budget)
    assert ids[0].tolist() == prefix + tokenizer.encode('A') + [tokenizer.eos_token_id]
    assert attention[0].tolist() == [1] * budget
    assert mask[0].tolist() == [0] * len(prefix) + [1, 1]
    accepted = sequence_contract(tokenizer, prompt, 'A', budget)
    assert accepted['valid'] and accepted['retained_prompt_tokens'] == len(prefix)
    assert accepted['truncated_prompt_tokens'] == 0
    for response in ('AB', 'ABCD'):
        with pytest.raises(ValueError, match='max_length'):
            prepare_batch(tokenizer, [prompt], [response], 'cpu', budget)
        rejected = sequence_contract(tokenizer, prompt, response, budget)
        assert not rejected['valid'] and rejected['error'] == 'sequence_exceeds_budget'


def test_response_only_overflow_remains_invalid():
    with pytest.raises(ValueError, match='max_length'):
        prepare_batch(ByteTokenizer(), ['p'], ['AB'], 'cpu', 3)
    result = sequence_contract(ByteTokenizer(), 'p', 'AB', 3)
    assert not result['valid'] and result['error'] == 'response_exceeds_budget'


def record():
    return dict(id='one', prompt='p', chosen='A', rejected_easy='BC',
                rejected_hard='BCD', level=1, decision=0)


def test_token_cache_preserves_training_masks_and_rejects_overflow(tmp_path):
    tokenizer, row = ByteTokenizer(), record()
    destination = tmp_path / 'valid.jsonl'
    assert cache_tokens([row], tokenizer, destination, 64) == 1
    _, saved = next(read_jsonl(destination))
    for variant in saved['variants']:
        prompt, chosen, rejected = format_pair(row, variant['hard'], variant['explicit'])
        response = chosen if variant['role'] == 'chosen' else rejected
        ids, _, mask = prepare_batch(tokenizer, [prompt], [response], 'cpu', 64)
        assert variant['input_ids'] == ids[0].tolist()
        assert variant['response_mask'] == mask[0].tolist()
    overflow = tmp_path / 'overflow.jsonl'
    with pytest.raises(ValueError):
        cache_tokens([row], tokenizer, overflow, 4)
    assert not overflow.exists()


def test_legacy_audit_marks_overflow_without_truncating(monkeypatch):
    tokenizer_loader = SimpleNamespace(from_pretrained=lambda *args, **kwargs: ByteTokenizer())
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoTokenizer=tokenizer_loader))
    report = tokenization_inventory([record()], 'qwen3-4b', None, None, True, 4)
    assert report['summary']['invalid_sequences'] == 5
    assert report['summary']['prompt_truncations'] == 0
    assert [row['valid'] for row in report['sequences']] == [False, False, False, True, False, False]
    assert all(row['prompt_tokens_removed'] == 0 for row in report['sequences'])
