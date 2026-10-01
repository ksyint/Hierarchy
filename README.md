# Hierarchy Aware Preference Optimization

**Hierarchy Aware Preference Optimization for the Safety of Korean Small Language Models**
Soo Yong Kim, Junyoung Koh, Kyeonghun Kim, Seunghyeok Hong.

Independent implementation of **KoSCoPe** from the supplied manuscript: competence-gated safety curriculum, HARM-DPO, counterfactual consistency, and stochastic reasoning internalization. No official authorship, trained weights, or reproduced results are claimed.

## Method

`utils/curriculum.py` implements the two-gate EMA-loss/held-out-behavior unlock, sequential levels, normalized level mixture with 20% aggregate lower-level replay, easy-to-hard negative sampling, decaying per-level margins, and constrained reasoning fade. `utils/losses.py` computes:

```text
L_HARM = -log sigmoid(beta * [(log pi_w - log pi_l) - (log ref_w - log ref_l)] - gamma_level)
gamma_level = gamma0 * exp(-kappa * (epoch - unlock_epoch))
L_CCR = -log p(correct decision | x) - log p(correct flipped decision | x')
L = L_HARM + lambda * L_CCR
```

Reference log probabilities are detached. Only completion tokens contribute to sequence likelihood, including the EOS token. The decision probe restricts next-token logits to two one-token verbalizers in `[comply, refuse]` order.

Opposite-decision counterfactual pairs use paired supervised NLL; same-decision pairs use half-symmetric KL. The configurable level mixture favors higher levels as their ramp progresses and normalizes over available levels. Reasoning fading starts only after Level 3 has saturated its hard-negative ramp.

## Installation and CPU check

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --config configs/smoke.yaml --output outputs/smoke
python eval.py --checkpoint outputs/smoke/last.pt
python inference.py --checkpoint outputs/smoke/last.pt --prompt 'Case 7: ALLOW'
python -m pytest -q
```

The smoke fixture trains an autoregressive byte-level GRU on synthetic access-control labels. It exercises actual DPO/CCR gradients and curriculum transitions; its text quality and probe scores are not Korean SLM safety results. Smoke thresholds and optimization settings deliberately differ from the paper.

## Real preference data

Use UTF-8 JSONL with one record per prompt:

```json
{"prompt":"사용자 질문", "chosen":"선호 답변", "rejected_easy":"단순 비선호 답변", "rejected_hard":"어려운 비선호 답변", "level":3, "decision":0, "cf_prompt":"의도가 바뀐 질문", "cf_decision":1, "chosen_thinking":"검토 근거", "rejected_easy_thinking":"검토 근거", "rejected_hard_thinking":"검토 근거"}
```

`decision`: 0 = comply, 1 = refuse. Counterfactual fields are optional but must appear together. Reasoning fields are optional; when present they are enclosed in `[THINKING]...[/THINKING]`. Faded samples get `[WITHOUT_THINKING]`. Provide a disjoint validation file with all three levels for the behavioral gate. Exact duplicate train/validation prompts are rejected; semantic deduplication and source/license checks are data-preparation responsibilities.

```bash
pip install -r requirements-models.txt
python train.py --config configs/paper.yaml --model /path/to/base-model \
  --data data/sft.jsonl --validation data/validation.jsonl --stage sft --epochs 50 \
  --lora --device cuda --output outputs/sft
python train.py --config configs/paper.yaml --model /path/to/base-model \
  --checkpoint outputs/sft/last.pt --data data/preferences.jsonl \
  --validation data/validation.jsonl --lora --device cuda --output outputs/dpo
python eval.py --checkpoint outputs/dpo/last.pt --data data/test.jsonl --device cuda
```

Set `decision_tokens` to two strings that each tokenize into **exactly one token** for your backbone; the loader rejects multi-token verbalizers. Prompts are explicit text, so include the backbone's chat template in your exported data when needed. The optional adapter uses Hugging Face causal models and PEFT attention/MLP LoRA; backbone-specific projection names may require changing the target module list. Checkpoints keep the base model identifier for reload and the original model must remain accessible. Reference model duplication needs sufficient memory.

The SFT and DPO input files should follow the paper's disjoint 40%/60% split. SFT samples all three levels uniformly; DPO uses competence-gated sampling.

## Scope and evaluation

The repository supplies original losses, scheduling, real sequence-likelihood optimization, a Hugging Face/LoRA adapter, greedy answer-only inference, and decision-probe evaluation. The three-teacher generation pipeline, Korean dependency parser, published category mapping, judge prompts, licensed source datasets and author-produced counterfactuals are not included. The paper's 4×A100, bf16/distributed training, batch-32 gradient accumulation and complete 50/80-epoch protocol have not been reproduced. `steps_per_epoch` is an explicit sampling budget, not one full dataset pass. The default real-data config uses paper loss/schedule hyperparameters but a smaller local batch and implementation-specific ramps.

`eval.py` reports one-token decision safety/over-refusal rates. These are diagnostic proxy metrics, not the paper's generated-response judge evaluation. No benchmark numbers are copied into an implementation results table.

## Citation

Use the published bibliographic record of the manuscript when available. This repository does not infer a venue, year, or public identifier that was not provided.
