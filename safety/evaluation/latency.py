"""CUDA-synchronized generation timings for explicit and internalized reasoning."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from safety.data.schema import read_jsonl, write_jsonl
from safety.evaluation.generation import answer_batches
from safety.generation.batched import generation_settings, padded_inputs, trim_generated
from safety.models.backbones import cuda_device, load_model, restore


def timing_summary(values):
    if not values:
        return {'count': 0}
    ordered = sorted(values)
    def percentile(q):
        position = (len(ordered) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])
    return {'count': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values),
            'minimum': ordered[0], 'maximum': ordered[-1], 'p90': percentile(.9),
            'p95': percentile(.95), 'population_stddev': statistics.pstdev(values)}


def measure_batch(model, tokenizer, batch, device, max_new_tokens):
    inputs = padded_inputs(batch, tokenizer.pad_token_id, device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                    use_cache=True, pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    width = inputs['input_ids'].shape[1]
    lengths = [len(trim_generated(row[width:].tolist(), tokenizer.eos_token_id, tokenizer.pad_token_id))
               for row in generated]
    return {'batch_seconds': seconds, 'amortized_seconds_per_sample': seconds / len(batch),
            'generated_tokens': sum(lengths), 'generated_lengths': lengths,
            'generated_tokens_per_second': sum(lengths) / seconds,
            'peak_allocated_bytes': peak_allocated, 'peak_reserved_bytes': peak_reserved,
            'batch_size': len(batch), 'padded_prompt_tokens': width,
            'prompt_tokens': sum(len(tokens) for _, tokens in batch)}


def benchmark(model, tokenizer, rows, device, modes=('explicit', 'implicit'), batch_size=1,
              max_new_tokens=512, context_length=8192, warmup=3, repeats=3):
    device = cuda_device(device)
    if not rows or warmup < 0 or repeats < 1:
        raise ValueError('Use nonempty prompts, nonnegative warmup and positive repeats.')
    if len(set(modes)) != len(modes):
        raise ValueError('Benchmark modes must be unique.')
    batches = {mode: list(answer_batches(rows, tokenizer, batch_size, mode, context_length, max_new_tokens))
               for mode in modes}
    records = []
    with generation_settings(model, tokenizer):
        for mode in modes:
            for index in range(warmup):
                measure_batch(model, tokenizer, batches[mode][index % len(batches[mode])], device, max_new_tokens)
        for repeat in range(repeats):
            order = modes if repeat % 2 == 0 else tuple(reversed(modes))
            for mode in order:
                for index, batch in enumerate(batches[mode]):
                    result = measure_batch(model, tokenizer, batch, device, max_new_tokens)
                    records.append({'mode': mode, 'repeat': repeat, 'batch': index,
                                    'ids': [row['id'] for row, _ in batch], **result})
    return records


def benchmark_report(records, device, settings):
    modes = {}
    for mode in sorted({row['mode'] for row in records}):
        rows = [row for row in records if row['mode'] == mode]
        elapsed = sum(row['batch_seconds'] for row in rows)
        examples = sum(row['batch_size'] for row in rows)
        tokens = sum(row['generated_tokens'] for row in rows)
        modes[mode] = {'batches': len(rows), 'examples_including_repeats': examples,
                       'batch_seconds': timing_summary([row['batch_seconds'] for row in rows]),
                       'seconds_per_sample': elapsed / examples,
                       'generated_tokens_per_sample': tokens / examples,
                       'generated_tokens_per_second': tokens / elapsed,
                       'peak_allocated_bytes': max(row['peak_allocated_bytes'] for row in rows)}
    properties = torch.cuda.get_device_properties(device)
    result = {'settings': settings, 'modes': modes, 'torch_version': str(torch.__version__),
              'cuda_version': torch.version.cuda, 'gpu': properties.name,
              'gpu_memory_bytes': properties.total_memory,
              'backend': 'transformers.generate',
              'timing_scope': 'generation with pretokenized CUDA input and CUDA synchronization'}
    if 'explicit' in modes and 'implicit' in modes:
        result['explicit_over_implicit_seconds_ratio'] = modes['explicit']['seconds_per_sample'] / modes['implicit']['seconds_per_sample']
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--model', default='qwen3-4b')
    parser.add_argument('--checkpoint')
    parser.add_argument('--local-model')
    parser.add_argument('--cache-dir', default='.cache/huggingface')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--modes', nargs='+', choices=['explicit', 'implicit'], default=['explicit', 'implicit'])
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--context-length', type=int, default=8192)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', required=True)
    parser.add_argument('--details', required=True)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive.')
    device = cuda_device(args.device)
    rows = [row for _, row in read_jsonl(args.input)]
    if args.limit:
        rows = rows[:args.limit]
    if args.checkpoint:
        model, tokenizer, _ = restore(args.checkpoint, device, local_dir=args.local_model,
                                      cache_dir=args.cache_dir, offline=args.offline)
    else:
        model, tokenizer = load_model(args.model, device=device, lora=False, local_dir=args.local_model,
                                      cache_dir=args.cache_dir, offline=args.offline)
    records = benchmark(model, tokenizer, rows, device, tuple(args.modes), args.batch_size,
                        args.max_new_tokens, args.context_length, args.warmup, args.repeats)
    report = benchmark_report(records, device, vars(args))
    write_jsonl(args.details, records)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'details': args.details}))
