# Source layout

`methods/preference.py` keeps model loading, likelihoods and learner state together. `benchmarks/korean.py` owns the training stream. Extended workflow modules form separate annotation, partition, evaluation, teacher-review, artifact and study branches under `safety/`.

```bash
python koscope.py annotations --help
python koscope.py review --help
python koscope.py artifact --help
```

The command registry imports an extended module only when its command is selected. Annotation and partition tools exchange JSONL records. Evaluation exchanges scored JSONL. Artifact and study tools exchange versioned JSON manifests. The corresponding contracts live in `schemas/`, alongside corpus examples under `examples/`.
