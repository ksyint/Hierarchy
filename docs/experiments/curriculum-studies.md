# Curriculum studies

A study expands recipe/seed pairs into SFT, DPO and optional evaluation stages. DPO depends on the SFT artifact and evaluation depends on DPO. Plans record input hashes and save complete command lines. Existing SFT checkpoints can be supplied to share a reviewed initialization. Their checkpoint, adapter and tokenizer files are fingerprinted during planning. Completion records fingerprint generated prerequisite artifacts as well, so changed upstream weights cannot reuse a previous stage completion.

```bash
python koscope.py study plan --config configs/korean/harm.yaml --sft-data data/korean/sft.jsonl --preferences data/korean/preferences.jsonl --validation data/korean/validation.jsonl --test data/korean/test.jsonl --output outputs/study
python koscope.py study run --plan outputs/study/study.json --skip-complete
```

`study history --history outputs/dpo/metrics.json --output outputs/trajectory` exports per-level loss EMA, probe accuracy, unlock epochs and the explicit-reasoning probability. The parser checks that recorded unlock events remain consistent through the trajectory. Stage logs and hashes support comparing observed runs.
