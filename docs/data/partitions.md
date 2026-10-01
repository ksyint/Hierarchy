# Prompt-disjoint partitions

Partition auditing checks primary and counterfactual prompts across every supplied split. Exact matching normalizes Unicode and whitespace. Optional near matching uses the same MinHash grouping as data preparation. Coverage reports retain the source/category/level composition and flipped-intent pair counts.

```bash
python koscope.py partitions audit --split sft=data/korean/sft.jsonl --split preferences=data/korean/preferences.jsonl --split validation=data/korean/validation.jsonl --split test=data/korean/test.jsonl --near --output outputs/partitions.json
```

`partitions build` creates the four standard partitions. `partitions subset --per-stratum N` selects complete connected prompt groups and exports selected and omitted records. `partitions language` exports a language slice without rewriting prompts or applying a translation model.
