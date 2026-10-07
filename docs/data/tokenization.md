# Tokenizer budgets

Tokenization inspection uses the selected published tokenizer and its chat template. It counts the complete prompt, answer and EOS for plain and explicit-reasoning forms. Sequences exceeding the configured context are invalid and are rejected by training and token caching. Prompts are never shortened according to answer length: HARM-DPO must compare chosen and rejected responses conditioned on the same prompt (paper Section 3.3, Eq. (5)). Increase the context budget or curate invalid records before training. Inspection does not construct a language model.

```bash
python koscope.py annotations --input data/raw/annotations.jsonl --tokenizer qwen3-4b --context-length 2048 --output outputs/token_audit
```

The registry pins tokenizer revisions. Use `--local-model checkpoints/qwen3-4b --offline` for a complete local tokenizer snapshot. `--cache-dir` chooses the shared Hugging Face cache. Inspect `tokenization.json` before selecting the training context length or revising long examples.
