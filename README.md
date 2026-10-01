# Hierarchy Aware Preference Optimization

**Hierarchy Aware Preference Optimization for the Safety of Korean Small Language Models**
Soo Yong Kim, Junyoung Koh, Kyeonghun Kim, Seunghyeok Hong.

KoSCoPe combines a three-level competence curriculum, HARM-DPO, counterfactual decision consistency, and stochastic reasoning internalization. This repository organizes training as a sequence of level experiences consumed by preference-learning strategies.

## Preference learning

`HARMLearner` scores preferred and rejected completions against a frozen reference. Its objective is:

```text
L_HARM = -log sigmoid(beta * [(log pi_w - log pi_l) - (log ref_w - log ref_l)] - gamma_level)
gamma_level = gamma0 * exp(-kappa * (epoch - unlock_epoch))
L_CCR = -log p(correct decision | x) - log p(correct flipped decision | x')
L = L_HARM + lambda_ccr * L_CCR
```

Completion likelihood includes EOS and excludes prompt/padding tokens. Counterfactual pairs with opposite labels use paired decision NLL; same-label pairs use half-symmetric KL. The probe reads two next-token verbalizers in `[comply, refuse]` order.

The curriculum opens levels sequentially when both the EMA loss and held-out behavioral accuracy satisfy their gates. Sampling retains a configured lower-level replay share, ramps toward hard negatives, and fades `[THINKING]` supervision after Level 3 reaches hard-negative saturation. `SFTLearner` supplies the preceding supervised phase and samples all three levels uniformly.

## Dataset and backbone

Use CUDA-enabled PyTorch with the Hugging Face/PEFT dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-models.txt
```

Prepare UTF-8 JSONL records with explicit prompt, response and level annotations:

```json
{"prompt":"사용자 질문", "chosen":"선호 답변", "rejected_easy":"단순 비선호 답변", "rejected_hard":"어려운 비선호 답변", "level":3, "decision":0, "cf_prompt":"의도가 바뀐 질문", "cf_decision":1, "chosen_thinking":"검토 근거", "rejected_easy_thinking":"검토 근거", "rejected_hard_thinking":"검토 근거"}
```

`decision` is 0 for comply and 1 for refuse. Supply counterfactual fields together. Optional reasoning fields are formatted as `[THINKING]...[/THINKING]`; faded examples receive `[WITHOUT_THINKING]`. Training and validation must each contain all three levels and use disjoint prompts. Keep the SFT/preference pools separate when following the 40%/60% partition.

Choose a causal-language-model checkpoint and set two single-token `decision_tokens` for its tokenizer. Export prompts using that backbone's chat template. The LoRA adapter targets attention projections and MLP gate/up/down projections; configure module names for the selected backbone.

## Run an experiment

Model execution requires `cuda` or `cuda:N`. The reference model is a frozen copy of the initialized policy.

```bash
python train.py --dataset korean --model /path/to/base-model \
  --data data/sft.jsonl --validation data/validation.jsonl --stage sft --epochs 50 \
  --lora --device cuda --output outputs/sft
python train.py --dataset korean --model /path/to/base-model \
  --checkpoint outputs/sft/last.pt --data data/preferences.jsonl \
  --validation data/validation.jsonl --lora --device cuda --output outputs/dpo
python eval.py --checkpoint outputs/dpo/last.pt --data data/test.jsonl --device cuda
python inference.py --checkpoint outputs/dpo/last.pt \
  --prompt '안전한 비밀번호 관리 방법을 알려 주세요.' --device cuda
```

Checkpoints retain model weights, configuration, curriculum state, and the base-model identifier. Keep that base checkpoint accessible when restoring a LoRA experiment. `steps_per_epoch` controls the number of sampled preference minibatches per experience.

## Strategy recipe catalog

The **243 recipes** under `configs/experiments/korean/` vary five implemented controls:

| Control | Settings |
| --- | --- |
| Initial margin `gamma0` | 0.5, 1.0, 2.0 |
| DPO temperature `beta` | 0.05, 0.10, 0.20 |
| Competence gates | conservative `(0.25,0.20; 0.90)`, standard `(0.35,0.30; 0.85)`, permissive `(0.45,0.40; 0.80)` |
| Margin decay `kappa` | 0.04, 0.08, 0.12 |
| Lower-level replay | 0.20, 0.30, 0.40 |

Gate tuples contain the Level-1/Level-2 loss thresholds followed by required probe accuracy. Each file includes the complete optimizer, level-ramp, reasoning-fade, and CCR settings used by the learner.

```bash
python train.py --list-recipes
python train.py --recipe korean/g20/b010/standard/k008/r020 --dry-run
python train.py --recipe korean/g20/b010/standard/k008/r020 \
  --model /path/to/base-model --checkpoint outputs/sft/last.pt \
  --data data/preferences.jsonl --validation data/validation.jsonl \
  --lora --device cuda --output outputs/harm_standard
python -m experiments.build_catalog
```

`--recipe` selects a validated catalog entry; `--config` accepts an explicit YAML path. `--dry-run` prints the resolved strategy and inputs without creating a model. The builder regenerates recipe files from `configs/korean/harm.yaml`.

## Code organization and evaluation

- `benchmarks/`: annotated records, reasoning formatting, and level-experience streams.
- `methods/preference/`: supervised/HARM learner strategies and counterfactual objectives.
- `methods/curriculum/`: competence gates, margins, replay and reasoning schedules.
- `networks/`: token batching, completion scoring, decision probes and model loading.
- `experiments/`: recipe discovery, experiment execution, held-out probes and checkpoints.

The evaluator reports decision-probe safety and over-refusal rates. The inference command generates answer-only text and reports the prompt's refusal probability. Training logs include per-level held-out accuracy, EMA loss, unlock epochs, and reasoning probability.
