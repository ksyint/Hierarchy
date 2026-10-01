# Hierarchy Aware Preference Optimization

KoSCoPe trains released Korean language models with supervised adaptation, a three-level safety curriculum, HARM-DPO, counterfactual consistency and reasoning internalization. The default is **Qwen3-4B with rank-16 LoRA**, loaded from Hugging Face when training or inference starts.

## CUDA environment

Use Python 3.10+ and a CUDA-enabled PyTorch installation, then install the model dependencies:

```bash
pip install -r requirements.txt
```

The loader places model weights on the selected `cuda`/`cuda:N` device in BF16. Gradient checkpointing is enabled for training. The default microbatch is one and 32 accumulated microbatches form an optimizer update. `--batch-size` and `--accumulation` adjust this allocation. `--attention flash_attention_2` selects a separately installed FlashAttention kernel. EXAONE defaults to eager attention.

## Automatic model downloads

| Alias | Released checkpoint | Local directory |
|---|---|---|
| `qwen3-1.7b` | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | `checkpoints/qwen3-1.7b/` |
| `qwen3-4b` | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | `checkpoints/qwen3-4b/` |
| `kanana-2.1b` | [kakaocorp/kanana-nano-2.1b-base](https://huggingface.co/kakaocorp/kanana-nano-2.1b-base) | `checkpoints/kanana-2.1b/` |
| `exaone-2.4b` | [LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct](https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct) | `checkpoints/exaone-2.4b/` |
| `gemma3-4b` | [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | `checkpoints/gemma3-4b/` |

`safety/models/backbones.py` pins each published revision. Tokenizers, model configs and sharded weights download into `.cache/huggingface/`. Repeat runs reuse this cache. `--cache-dir /path/to/cache` selects a shared cache. No separate download command is needed for normal execution:

```bash
python koscope.py infer --model qwen3-4b --prompt '안전한 비밀번호 관리 방법을 알려 주세요.' --device cuda
```

To download on another connected machine, export a complete local directory:

```bash
python koscope.py download --model qwen3-4b --destination checkpoints/qwen3-4b
python koscope.py infer --model qwen3-4b --local-model checkpoints/qwen3-4b --offline \
  --prompt '안전한 비밀번호 관리 방법을 알려 주세요.' --device cuda
```

For a browser download, open the model's linked **Files and versions** page at the revision printed by `python koscope.py download --list`. Put `config.json`, tokenizer files, the safetensors index and **every** safetensors shard in the table's local directory. Preserve filenames. Include EXAONE's `configuration_exaone.py` and `modeling_exaone.py`. Pass the same `--model` alias together with `--local-model DIRECTORY`. Gemma requires accepting its Hugging Face model terms and running `hf auth login` on the downloading machine. Authentication can also use the standard `HF_TOKEN` environment variable.

The loader targets all language attention and MLP linear projections, resolves EXAONE projection names, and includes Gemma's language head. Gemma's visual tower remains frozen for this text task. Official chat templates are applied inside token batching. Input JSONL therefore contains ordinary prompt text.

## Data preparation

Start from permitted copies of the safety corpora used by the experiment: WildGuardMix, SQuARe, KorNAT, KOLD, KoSBi and BeaverTails-ko, or an annotated Korean corpus with the same fields. Export one UTF-8 JSON object per line to `data/raw/annotated.jsonl`:

```json
{"source":"corpus_name","category":"privacy","prompt":"사용자 질문","chosen":"선호 답변","rejected_easy":"단순 비선호 답변","rejected_hard":"어려운 비선호 답변","level":3,"decision":0,"cf_prompt":"의도가 바뀐 질문","cf_decision":1,"chosen_thinking":"검토 근거","rejected_easy_thinking":"검토 근거","rejected_hard_thinking":"검토 근거"}
```

Obtain SQuARe and KoSBi from the `data/` directories in [Korean Safety Benchmarks](https://github.com/naver-ai/korean-safety-benchmarks), KOLD from its [published JSON](https://github.com/boychaboy/KOLD/tree/main/data), and KorNAT from its [dataset repository](https://github.com/jiyounglee-0523/KorNAT). [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix) requires accepting its access terms. Preserve original example identifiers and official split assignments while converting records. Translate English source records into reviewed Korean text before annotation. An approved Korean BeaverTails export can enter the same schema.

`level` is an integer: 1 for explicit harms, 2 for contextual social harms, and 3 for nuanced intent-sensitive domains. `decision` is 0 for comply and 1 for refuse. Assign these from reviewed annotations. Preserve `source` and fine-grained `category` for stratification. Supply enough independent prompt groups in each stratum to populate the held-out splits. Each level needs at least four independent groups across the corpus. Both counterfactual fields are supplied together. Their decision labels should reflect the changed intent. Reasoning fields are optional. Keep prompts and completions separate and do not pre-render chat templates or add special tokenizer tokens.

```bash
python koscope.py prepare --input data/raw/annotated.jsonl --output data/korean --seed 42
```

Preparation normalizes Unicode to NFC, strips surrounding whitespace and applies five-character MinHash near-duplicate filtering at Jaccard 0.85 before splitting. Records connected through matching or near-duplicate primary/counterfactual prompts remain in the same partition. The grouped allocator balances source/category/level strata toward 10% validation, 10% test, 32% SFT and 48% preference training. The two training targets correspond to a 40/60 division of the training pool. Actual counts follow whole prompt groups and all four splits contain all three levels. The generated layout is:

```text
data/korean/
  sft.jsonl
  preferences.jsonl
  validation.jsonl
  test.jsonl
  split.json
```

Use official held-out sets unchanged when they already exist. Prepare training pools separately and export the same schema. The training loader verifies level coverage and rejects overlapping training/validation prompts in either primary or counterfactual fields, after Unicode and whitespace normalization. Long completions are rejected with a context-length error. Raise `max_length` or curate the affected records before training. Tokenization uses the downloaded tokenizer, masks prompt/padding tokens and includes completion EOS.

### Teacher-generated candidate responses

`koscope.py candidates` loads the three teachers sequentially and writes candidate response fields for review. Its input JSONL supplies `prompt`, reviewed `level` and `decision`, plus optional reviewed counterfactual fields. The strong teacher is [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B), the medium teacher [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B), and the weak teacher [Qwen3-8B-AWQ](https://huggingface.co/Qwen/Qwen3-8B-AWQ). Create a separate teacher environment with the pinned PyTorch 2.6.0, Transformers 4.51.3 and AutoAWQ 0.2.9 stack. AutoAWQ uses its Triton backend for CUDA inference:

```bash
python3.10 -m venv .venv-teachers
.venv-teachers/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv-teachers/bin/python -m pip install -r requirements-teachers.txt
.venv-teachers/bin/python koscope.py candidates --seeds data/raw/prompts.jsonl --output data/raw/candidates.jsonl --device cuda
```

Review the preference ordering and intent annotations, then provide the curated file to `koscope.py prepare`. The 32B teacher is placed on the selected GPU. Select a device with enough memory for its BF16 weights. Automatic downloads use the same pinned registry and cache as training. For an offline machine, download aliases `teacher-strong`, `teacher-medium` and `teacher-weak` with `python koscope.py download --model ALIAS --destination checkpoints/teachers/ALIAS`. Copy the entire parent folder and pass `--teacher-root checkpoints/teachers --offline`.

## Supervised and preference training

```bash
python koscope.py train --model qwen3-4b --stage sft --data data/korean/sft.jsonl \
  --validation data/korean/validation.jsonl --output outputs/sft --device cuda
python koscope.py train --checkpoint outputs/sft/last.pt --stage dpo \
  --data data/korean/preferences.jsonl --validation data/korean/validation.jsonl \
  --output outputs/dpo --device cuda
python koscope.py evaluate --checkpoint outputs/dpo/last.pt --data data/korean/test.jsonl --device cuda
python koscope.py infer --checkpoint outputs/dpo/last.pt \
  --prompt '안전한 비밀번호 관리 방법을 알려 주세요.' --device cuda
```

SFT uses 50 epochs and peak LR `2e-5`. DPO uses 80 epochs and `5e-6`. AdamW uses weight decay 0.01, cosine decay, 5% warm-up and gradient clipping at 1.0. An epoch samples a data-pool-sized number of records. Set `--epochs` to change the duration. `pretrained/` contains the portable adapter and tokenizer. `last.pt` records the experiment and curriculum. Restoring downloads the pinned base automatically or accepts `--local-model` and `--offline`.

## Hierarchy and objectives

`HARMLearner` scores completions against the frozen SFT reference. Its objective is:

```text
L_HARM = -log sigmoid(beta * [(log pi_w - log pi_l) - (log ref_w - log ref_l)] - gamma_level)
gamma_level = gamma0 * exp(-kappa * (epoch - unlock_epoch))
L = L_HARM + lambda_ccr * L_CCR
```

Opposite-label counterfactual pairs use paired decision likelihood. Same-label pairs use symmetric KL. Decision verbalizers may span multiple tokens. Competence gates combine per-level loss EMA and held-out decision accuracy. The curriculum retains lower-level replay, increases hard-negative probability and fades explicit reasoning supervision.

The 243 recipes under `configs/` vary initial margin, DPO temperature, competence gates, decay and replay. Every recipe drives the same pretrained-model learner. `--config` accepts YAML or a Python file containing one literal `CONFIG` dictionary. `--recipe` keeps the same logical names regardless of file format or placement. The catalog builder restores both formats in their assigned locations:

```bash
python koscope.py train --list-recipes
python koscope.py train --recipe korean/g20/b010/standard/k008-r020 --dry-run
python koscope.py train --recipe korean/g20/b010/standard/k008-r020 --checkpoint outputs/sft/last.pt \
  --data data/korean/preferences.jsonl --validation data/korean/validation.jsonl --output outputs/recipe
```

`safety/data/korean.py` handles prompt grouping, record formatting and experience streams. `safety/models/learner.py` contains token scoring, curriculum updates and preference objectives. The `safety/training/` modules manage optimizer updates, state recovery and training events. `koscope.py` manages preparation, teacher candidates, training and evaluation commands. Use `python koscope.py COMMAND --help` to inspect one command. Evaluation reports decision safety and over-refusal rates, while inference generates answer-only text.

## Workflow modules

The nested `safety/` modules connect annotation review, token budgets, prompt-disjoint splits, counterfactual benchmarks, adapter artifacts and SFT-to-DPO studies. Existing training and inference commands retain their arguments.

- [Source layout](docs/source-layout.md)
- [Adapter artifacts](docs/adapter-packages.md)
- [Annotation contracts](docs/data/annotation-contract.md)
- [Prompt-disjoint partitions](docs/data/partitions.md)
- [Tokenizer budgets](docs/data/tokenization.md)
- [Language and paired comparisons](docs/evaluation/cross-language.md)
- [Decision benchmarks](docs/evaluation/decision-benchmarks.md)
- [Curriculum studies](docs/curriculum-studies.md)
- [Teacher candidate review](docs/data/teacher-review.md)

Each extended command exposes its options through `python koscope.py COMMAND --help`. JSON schemas are in `schemas/` and replaceable input examples are in `examples/`.

## Source conversion and reviewed preferences

`convert-source` accepts downloaded JSONL, JSON, CSV and TSV. Supply a YAML mapping with `source`, `stage: seed`, `fields` mapping normalized names to source column names, and `defaults` for reviewed metadata. A nested source field uses dot notation. `values.decision` and `values.level` map source labels explicitly. No category or intent label is inferred from a dataset name.

```bash
python koscope.py convert-source --input data/raw/source.csv --mapping data/source-mapping.yaml --output data/raw/seeds.jsonl --report reports/conversion.json
python koscope.py records --input data/raw/seeds.jsonl --stage seed --report reports/seeds.json
python koscope.py teacher-requests build --seeds data/raw/seeds.jsonl --output data/raw/requests.jsonl --report reports/requests.json
python koscope.py teacher-batches --requests data/raw/requests.jsonl --output data/raw/responses.jsonl --device cuda --batch-size 2
python koscope.py select-preferences review-sheet --requests data/raw/requests.jsonl --responses data/raw/responses.jsonl --output data/raw/reviews.jsonl
```

Teacher batches load the same pinned strong, medium and AWQ weak models sequentially. `--resume` validates completed request identities before continuing. Each review specifies `accepted`, the response `decision` and a numeric `quality`. After reviewing these fields, assemble complete triples:

```bash
python koscope.py select-preferences assemble --seeds data/raw/seeds.jsonl --requests data/raw/requests.jsonl --responses data/raw/responses.jsonl --reviews data/raw/reviews.jsonl --output data/raw/preferences.jsonl --report reports/selection.json
python koscope.py token-budget --input data/raw/preferences.jsonl --model qwen3-4b --report reports/tokens.json
```

`counterfactuals` joins reviewed variants by original ID and records edit spans. `taxonomy` applies a versioned, reviewed category mapping or compares independent annotations. `sampling` reports the expected exposure of each level and source before a training run.

## Training state and component studies

Training now writes `training-state.pt` after each completed epoch. It contains trainable policy and reference weights, optimizer moments, curriculum state, random states and the sampler position. Resume with the same training arguments plus `--resume-state outputs/dpo/training-state.pt`. The input file digests and training configuration must match. The final portable policy remains at `last.pt`. Validation improvements export `best.pt`, selected by macro decision accuracy across the three levels.

`--ablation` selects `full`, `no_curriculum`, `fixed_schedule`, `no_margin_decay`, `no_ccr`, `explicit_only` or `implicit_only`. The full setting retains the competence-gated curriculum. Source balancing is optional through `--balance-source source --sampling-temperature 0`. The default sampling temperature of 1 preserves source proportions within each level.

```bash
python koscope.py training-events --events outputs/dpo/updates.jsonl --output reports/training.json --updates-csv reports/updates.csv
python koscope.py training-state --state outputs/dpo/training-state.pt --output reports/state.json
python koscope.py curriculum-replay --config harm.yaml --history outputs/dpo/metrics.json --output reports/curriculum.json
```

## Response evaluation and paired comparisons

`answers generate` writes held-out responses using either `--mode implicit` or `--mode explicit`. `answers judge` loads a selected CUDA language model and records structured judgments. `answers judge-agreement` compares those judgments with an independent reference review. `answers score` reports response safety, over-refusal and helpfulness from the reviewed answers. Decision-verbalizer evaluations remain available through `evaluate` and `benchmark`.

```bash
python koscope.py answers generate --checkpoint outputs/dpo/best.pt --input data/korean/test.jsonl --output outputs/answers.jsonl --device cuda
python koscope.py answers judge --predictions outputs/answers.jsonl --model teacher-strong --output outputs/judgments.jsonl --device cuda
python koscope.py answers score --predictions outputs/answers.jsonl --reviews outputs/judgments.jsonl --output reports/responses.json
python koscope.py paired-bootstrap --reference outputs/reference.jsonl --candidate full=outputs/full.jsonl --resamples 1000 --output reports/paired.json
python koscope.py latency --checkpoint outputs/dpo/best.pt --input data/korean/test.jsonl --limit 100 --output reports/latency.json --details reports/latency-batches.jsonl --device cuda
```

Paired bootstrap aligns exact IDs and training seeds, stratifies by seed, level and intent, and applies Holm correction across the requested comparisons. `--per-seed 2500` selects the same stratified held-out subset for every compared method. Latency records the GPU, generation backend, context and output budgets, synchronized batch times and memory use. Both reasoning modes use identical prompts and alternate measurement order.

`robustness perturb` creates a separate training copy with a specified fraction of hierarchy-label changes. Evaluation uses the unchanged held-out set. `robustness surface` compares reviewed intent-preserving variants. `transfer` joins translated prompts by `parallel_id`, checks intent and level agreement, and compares only complete language groups.
