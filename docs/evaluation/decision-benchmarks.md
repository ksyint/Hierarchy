# Decision benchmarks

Evaluation scores the saved comply/refuse verbalizers and records refusal probabilities. Counterfactual examples receive a second prediction. Reports separate safety, over-refusal, accuracy, paired accuracy, changed-intent consistency and same-intent consistency. Grouped results retain level, category, corpus and language.

```bash
python koscope.py benchmark --checkpoint outputs/dpo/last.pt --data data/korean/test.jsonl --output outputs/benchmark --device cuda
```

Use `--responses` to generate answer-only text with the same checkpoint. `--predictions` summarizes previously saved scores. Optional calibration uses a disjoint validation prediction file, while `--threshold` applies a fixed threshold. The output includes calibration error, Brier score and the decision ROC curve.
