# Teacher candidate review

The review workflow resumes completed response fields for each seed and loads each of the three teacher models sequentially. Seed identity, level, decision and counterfactual annotations must remain unchanged during resume. Candidate generation uses CUDA and the same pinned model registry as the existing teacher command.

```bash
python koscope.py review generate --seeds data/raw/seeds.jsonl --output data/raw/candidates.jsonl --resume --device cuda
python koscope.py review export --candidates data/raw/candidates.jsonl --output data/review
python koscope.py review apply --candidates data/review/candidates.jsonl --review data/review/review.jsonl --output data/reviewed
```

The packet shuffles response order and assigns response IDs. Fill the chosen, rejected_easy and rejected_hard IDs, then set status. Accepted records must assign every response once. Applying decisions preserves teacher provenance and produces accepted, rejected and pending files.
