# Tokenizer budgets

Tokenization inspection uses the selected published tokenizer and its chat template. It counts answer tokens plus EOS for plain and explicit-reasoning forms. The report records which prompts would be shortened by the configured context and which responses exceed that context. It does not construct a language model.

```bash
python koscope.py annotations --input data/raw/annotations.jsonl --tokenizer qwen3-4b --context-length 2048 --output outputs/token_audit
```

The registry pins tokenizer revisions. Use `--local-model checkpoints/qwen3-4b --offline` for a complete local tokenizer snapshot. `--cache-dir` chooses the shared Hugging Face cache. Inspect `tokenization.json` before selecting the training context length or revising long examples.
