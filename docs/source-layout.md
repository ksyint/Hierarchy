# Source layout

`safety/models` pairs pinned backbone loading and checkpoint restoration with preference likelihoods and learner state. `safety/data` keeps the Korean training stream and teacher review beside `safety/data/curation`, which contains annotation validation and partition auditing. `safety/evaluation` separates model scoring from decision reports. `safety/study.py` and `safety/checkpoint.py` sit beside those packages and connect their experiment and artifact workflows.

```bash
python koscope.py annotations --help
python koscope.py review --help
python koscope.py artifact --help
```

The command registry imports an extended module only when its command is selected. Annotation and partition tools exchange JSONL records. Evaluation exchanges scored JSONL. Artifact and study tools exchange versioned JSON manifests. The corresponding contracts live in `schemas/`, alongside corpus examples under `examples/`.
