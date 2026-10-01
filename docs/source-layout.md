# Source layout

`safety/models` pairs pinned backbone loading and checkpoint restoration with preference likelihoods and learner state. `safety/data` groups the Korean training stream, validation, partition auditing and teacher review in four modules. `safety/evaluation` separates model scoring from decision reports. `safety/study.py` and `safety/checkpoint.py` sit beside those packages and connect their experiment and artifact workflows.

```bash
python koscope.py annotations --help
python koscope.py review --help
python koscope.py artifact --help
```

The command registry imports an extended module only when its command is selected. Annotation and partition tools exchange JSONL records. Evaluation exchanges scored JSONL. Artifact and study tools exchange versioned JSON manifests. The corresponding contracts live in `schemas/`, alongside corpus examples under `examples/`.
