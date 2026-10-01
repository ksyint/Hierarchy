# Language and paired comparisons

Preserve `language` when preparing multilingual annotations. The benchmark reports every language separately and pairs observed rate differences between languages. These comparisons retain sample counts and do not assume that two corpora contain matching prompts.

```bash
python koscope.py partitions language --input data/multilingual.jsonl --language ja --output data/japanese.jsonl
python koscope.py benchmark --predictions outputs/first/predictions.jsonl --compare outputs/second/predictions.jsonl --output outputs/comparison
```

Checkpoint comparisons require exactly matching example IDs and decisions. They list improved and regressed examples, accuracy change and the continuity-corrected McNemar statistic. Use the same prepared evaluation split for both checkpoints. Calibrate each model only on a separate validation partition.
