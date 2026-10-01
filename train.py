import argparse

from experiments.runner import run_experiment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', help='Catalog identifier from --list-recipes.')
    parser.add_argument('--list-recipes', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Resolve and validate the experiment without initializing a model.')
    parser.add_argument('--dataset', choices=['korean'], default='korean')
    parser.add_argument('--config', help='Override configs/<dataset>/harm.yaml.')
    parser.add_argument('--data')
    parser.add_argument('--validation')
    parser.add_argument('--model', default='qwen3-4b', help='Published backbone alias or Hugging Face ID.')
    parser.add_argument('--local-model', help='Full local checkpoint directory; takes precedence over Hub downloads.')
    parser.add_argument('--cache-dir', default='.cache/huggingface')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--attention', choices=['sdpa', 'eager', 'flash_attention_2'])
    parser.add_argument('--checkpoint', help='Matching architecture last.pt initialization (e.g. SFT output).')
    parser.add_argument('--stage', choices=['sft', 'dpo'], default='dpo')
    parser.add_argument('--lora', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--batch-size', type=int, help='Per-step microbatch size.')
    parser.add_argument('--accumulation', type=int, help='Microbatches per optimizer update.')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', default='outputs/qwen3-4b')
    return parser.parse_args()


if __name__ == '__main__':
    run_experiment(parse_args())
