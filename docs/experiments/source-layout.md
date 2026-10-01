# Source layout

`safety/models/preference` pairs pinned backbone loading and checkpoint restoration with preference likelihoods and learner state. `safety/data/annotations` groups the Korean training stream, validation, partition auditing and teacher review in four modules. `safety/evaluation/decisions` separates model scoring from decision reports. `safety/experiments/curriculum` keeps study execution beside checkpoint packaging.

```bash
python koscope.py annotations --help
python koscope.py review --help
python koscope.py artifact --help
```

The command registry imports an extended module only when its command is selected. Annotation and partition tools exchange JSONL records. Evaluation exchanges scored JSONL. Artifact and study tools exchange versioned JSON manifests. The corresponding contracts live in `schemas/`, alongside corpus examples under `examples/`.
