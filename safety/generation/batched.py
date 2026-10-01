"""CUDA teacher generation from resumable, identity-checked request manifests."""
import argparse
from contextlib import contextmanager
import gc
import json
import math
from pathlib import Path
import time

import torch

from safety.data.schema import read_jsonl
from safety.generation.requests import completed_requests, repair_response_tail, stable_digest, validate_request
from safety.models.backbones import cuda_device, load_model, resolve_backbone


@contextmanager
def generation_settings(model, tokenizer):
    training = model.training
    padding_side = tokenizer.padding_side
    use_cache = getattr(model.config, 'use_cache', None)
    try:
        model.eval()
        tokenizer.padding_side = 'left'
        model.config.use_cache = True
        yield
    finally:
        model.train(training)
        tokenizer.padding_side = padding_side
        if use_cache is not None:
            model.config.use_cache = use_cache


def encode_request(tokenizer, request, thinking=False):
    validate_request(request)
    return tokenizer.apply_chat_template(request['messages'], tokenize=True,
                                         add_generation_prompt=True, enable_thinking=thinking)


def token_batches(requests, tokenizer, batch_size, token_budget, context_length, max_new_tokens):
    if min(batch_size, token_budget, context_length, max_new_tokens) < 1:
        raise ValueError('Batch and token budgets must be positive.')
    encoded = []
    for row in requests:
        ids = encode_request(tokenizer, row)
        if not ids or len(ids) + max_new_tokens > context_length:
            raise ValueError(f'{row["request_id"]}: request exceeds the context budget.')
        if len(ids) + max_new_tokens > token_budget:
            raise ValueError('token_budget cannot hold even one request.')
        encoded.append((row, ids))
    encoded.sort(key=lambda item: (len(item[1]), item[0]['request_id']))
    batch = []
    for entry in encoded:
        width = len(entry[1]) + max_new_tokens
        if batch and (len(batch) >= batch_size or width * (len(batch) + 1) > token_budget):
            yield batch
            batch = []
        batch.append(entry)
    if batch:
        yield batch


def padded_inputs(batch, pad_token, device):
    width = max(len(ids) for _, ids in batch)
    ids = torch.full((len(batch), width), pad_token, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    for index, (_, tokens) in enumerate(batch):
        ids[index, -len(tokens):] = torch.tensor(tokens, dtype=torch.long, device=device)
        attention[index, -len(tokens):] = 1
    return {'input_ids': ids, 'attention_mask': attention}


def trim_generated(tokens, eos_token, pad_token):
    result = []
    eos = set(eos_token if isinstance(eos_token, (tuple, list)) else [eos_token])
    for token in tokens:
        result.append(token)
        if token in eos:
            break
    while result and result[-1] == pad_token and result[-1] not in eos:
        result.pop()
    return result


def generate_batch(model, tokenizer, batch, device, max_new_tokens, temperature=0.0):
    if temperature < 0:
        raise ValueError('temperature cannot be negative.')
    inputs = padded_inputs(batch, tokenizer.pad_token_id, device)
    options = {'max_new_tokens': max_new_tokens, 'do_sample': temperature > 0,
               'pad_token_id': tokenizer.pad_token_id, 'use_cache': True}
    if temperature > 0:
        options['temperature'] = temperature
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **options)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    width = inputs['input_ids'].shape[1]
    responses = []
    for index, (request, prompt_ids) in enumerate(batch):
        generated = trim_generated(output[index, width:].tolist(), tokenizer.eos_token_id, tokenizer.pad_token_id)
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not text:
            raise ValueError(f'Teacher returned an empty response for {request["request_id"]}.')
        spec = resolve_backbone(request['teacher'])
        responses.append({key: request[key] for key in ('request_id', 'record_id', 'record_sha256', 'role', 'teacher')})
        responses[-1].update(text=text, generated_tokens=len(generated), prompt_tokens=len(prompt_ids),
                             model_id=spec.repo, revision=spec.revision, batch_seconds=elapsed,
                             batch_size=len(batch), truncated=len(generated) >= max_new_tokens)
    return responses


def generation_contract(args):
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError('temperature must be finite and nonnegative.')
    if min(args.batch_size, args.token_budget, args.context_length, args.max_new_tokens) < 1:
        raise ValueError('Batch and token budgets must be positive.')
    if args.max_new_tokens >= args.context_length:
        raise ValueError('The context must leave room for prompt tokens.')
    teachers = {}
    for teacher in ('teacher-strong', 'teacher-medium', 'teacher-weak'):
        spec = resolve_backbone(teacher)
        local = Path(args.teacher_root) / teacher if args.teacher_root else None
        teachers[teacher] = {'model_id': spec.repo, 'revision': spec.revision,
                             'local_dir': str(local.resolve()) if local else None}
    return {'format': 1, 'teachers': teachers, 'temperature': args.temperature,
            'max_new_tokens': args.max_new_tokens, 'context_length': args.context_length,
            'batch_size': args.batch_size, 'token_budget': args.token_budget,
            'seed': args.seed, 'thinking': False}


def run_requests(args):
    contract = generation_contract(args)
    contract_id = stable_digest(contract)
    device = cuda_device(args.device)
    torch.manual_seed(args.seed)
    requests = [validate_request(row) for _, row in read_jsonl(args.requests)]
    if not requests:
        raise ValueError('The request manifest is empty.')
    destination = Path(args.output)
    if destination.resolve() == Path(args.requests).resolve():
        raise ValueError('Requests and responses require separate files.')
    if destination.exists() and not args.resume:
        raise FileExistsError('Response output exists. Use --resume to continue it.')
    repaired_bytes = repair_response_tail(destination) if destination.exists() else 0
    completed, pending = completed_requests(requests, [destination] if destination.exists() else [])
    for response in completed.values():
        if response.get('generation_sha256') != contract_id or response.get('generation') != contract:
            raise ValueError('Resume requires the same teacher generation options. Choose a new output file.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with destination.open('a', encoding='utf-8') as stream:
        for teacher in sorted({request['teacher'] for request in pending}):
            selected = [request for request in pending if request['teacher'] == teacher]
            local = str(Path(args.teacher_root) / teacher) if args.teacher_root else None
            model, tokenizer = load_model(teacher, device=device, lora=False, local_dir=local,
                                          cache_dir=args.cache_dir, offline=args.offline,
                                          dtype='float16' if teacher == 'teacher-weak' else 'bfloat16')
            try:
                with generation_settings(model, tokenizer):
                    batches = token_batches(selected, tokenizer, args.batch_size, args.token_budget,
                                            args.context_length, args.max_new_tokens)
                    for batch in batches:
                        responses = generate_batch(model, tokenizer, batch, device, args.max_new_tokens, args.temperature)
                        for response in responses:
                            response.update(generation=contract, generation_sha256=contract_id)
                            stream.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + '\n')
                        stream.flush()
                        written += len(responses)
                        print(json.dumps({'teacher': teacher, 'completed': len(completed) + written,
                                          'total': len(requests)}))
            finally:
                del model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()
    return {'previous': len(completed), 'written': written, 'total': len(requests),
            'discarded_incomplete_bytes': repaired_bytes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--teacher-root')
    parser.add_argument('--cache-dir', default='.cache/huggingface')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--token-budget', type=int, default=16384)
    parser.add_argument('--context-length', type=int, default=8192)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    print(json.dumps(run_requests(args)))
