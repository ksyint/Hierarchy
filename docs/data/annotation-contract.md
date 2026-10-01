# Annotation contracts

Annotation validation checks the three preference responses, reviewed level and decision, paired counterfactual fields and optional reasoning text. Normalization uses NFC and stable example identities. The inventory includes source, category, language, response lengths and duplicated prompt groups. Accepted-only export keeps records explicitly marked accepted.

```bash
python koscope.py annotations --input data/raw/annotations.jsonl --schema schemas/annotation.schema.json --output outputs/annotation_audit
```

The JSONL files under `examples/` illustrate field formats. Replace these annotations with the reviewed corpus for experiments. Use `--agreement PANEL.jsonl` to compare independent annotations for the same IDs. Agreement reports label counts, pairwise agreement and disagreement IDs without changing labels.
