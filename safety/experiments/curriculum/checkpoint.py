"""Portable adapter packages, integrity manifests and CUDA merged-model export."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil

import torch

from safety.models.preference.backbones import resolve_backbone, restore


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    size: int
    sha256: str

    @classmethod
    def inspect(cls, root, path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1 << 20), b''):
                digest.update(block)
        return cls(path.relative_to(root).as_posix(), path.stat().st_size, digest.hexdigest())


def read_checkpoint(path):
    path = Path(path).resolve()
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format') != 2 or not {'config', 'pretrained', 'curriculum'} <= state.keys():
        raise ValueError('Expected a format-2 hierarchy-aware preference checkpoint.')
    adapter = (path.parent / state['pretrained']).resolve()
    if path.parent not in adapter.parents or not adapter.is_dir():
        raise ValueError('The saved pretrained folder must be inside the checkpoint directory.')
    return state, adapter


def architecture_metadata(state, adapter):
    options = state['config']['pretrained']
    spec = resolve_backbone(options.get('model_name'))
    adapter_config = adapter / 'adapter_config.json'
    metadata = json.loads(adapter_config.read_text()) if adapter_config.exists() else {}
    return {'model_name': options.get('model_name'), 'base_repository': spec.repo,
            'base_revision': spec.revision, 'lora': bool(options.get('lora')),
            'rank': metadata.get('r'), 'alpha': metadata.get('lora_alpha'),
            'target_modules': metadata.get('target_modules'),
            'decision_tokens': state['config']['decision_tokens'],
            'max_length': state['config']['max_length']}


def tokenizer_inventory(adapter):
    configs = {}
    for name in ('tokenizer_config.json', 'special_tokens_map.json', 'generation_config.json'):
        path = adapter / name
        if path.exists():
            value = json.loads(path.read_text())
            configs[name] = {key: value[key] for key in ('bos_token', 'eos_token', 'pad_token', 'model_max_length', 'tokenizer_class') if key in value}
    if not (adapter / 'tokenizer_config.json').is_file():
        raise ValueError('The adapter package must contain its saved tokenizer configuration.')
    return configs


def weight_inventory(adapter):
    files = sorted(adapter.glob('*.safetensors')) + sorted(adapter.glob('*.bin'))
    if not files:
        raise ValueError('No serialized model or adapter weights were found.')
    entries = []
    for path in files:
        entry = {'file': path.name, 'bytes': path.stat().st_size}
        if path.suffix == '.safetensors':
            from safetensors import safe_open
            with safe_open(path, framework='pt', device='cpu') as weights:
                shapes = {name: list(weights.get_slice(name).get_shape()) for name in weights.keys()}
                entry['tensors'] = shapes
        entries.append(entry)
    return entries


def inspect(path):
    state, adapter = read_checkpoint(path)
    return {'checkpoint': str(Path(path).resolve()), 'architecture': architecture_metadata(state, adapter),
            'tokenizer': tokenizer_inventory(adapter), 'weights': weight_inventory(adapter),
            'curriculum': state['curriculum'], 'configuration': state['config']}


def package(checkpoint, destination):
    checkpoint = Path(checkpoint).resolve()
    state, adapter = read_checkpoint(checkpoint)
    tokenizer_inventory(adapter)
    weight_inventory(adapter)
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Use an empty package destination.')
    destination.mkdir(parents=True, exist_ok=True)
    packaged_adapter = destination / 'pretrained'
    shutil.copytree(adapter, packaged_adapter)
    state = dict(state, pretrained='pretrained')
    state['config'] = dict(state['config'])
    options = dict(state['config']['pretrained'])
    options['local_dir'] = None
    state['config']['pretrained'] = options
    torch.save(state, destination / 'last.pt')
    for name in ('metrics.json',):
        source = checkpoint.parent / name
        if source.exists():
            shutil.copy2(source, destination / name)
    files = [ArtifactFile.inspect(destination, path).__dict__ for path in sorted(destination.rglob('*')) if path.is_file()]
    manifest = {'format': 1, 'architecture': architecture_metadata(state, packaged_adapter), 'files': files}
    (destination / 'artifact.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def verify(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'artifact.json').read_text())
    if manifest.get('format') != 1:
        raise ValueError('Unsupported adapter artifact format.')
    files = set()
    for item in manifest['files']:
        path = (directory / item['path']).resolve()
        if directory not in path.parents or item['path'] in files:
            raise ValueError('Artifact manifest contains duplicate or outside paths.')
        files.add(item['path'])
        current = ArtifactFile.inspect(directory, path)
        if current.size != item['size'] or current.sha256 != item['sha256']:
            raise ValueError(f'Artifact integrity mismatch: {item["path"]}')
    state, adapter = read_checkpoint(directory / 'last.pt')
    if architecture_metadata(state, adapter) != manifest['architecture']:
        raise ValueError('Architecture metadata differs from the integrity manifest.')
    return {'verified_files': len(files), 'architecture': manifest['architecture']}


def compare(first, second):
    left_state, left_adapter = read_checkpoint(first)
    right_state, right_adapter = read_checkpoint(second)
    left = architecture_metadata(left_state, left_adapter)
    right = architecture_metadata(right_state, right_adapter)
    changed = {key: {'first': left.get(key), 'second': right.get(key)} for key in left.keys() | right.keys()
               if left.get(key) != right.get(key)}
    settings = {}
    for key in left_state['config'].keys() | right_state['config'].keys():
        a, b = left_state['config'].get(key), right_state['config'].get(key)
        if a != b:
            settings[key] = {'first': a, 'second': b}
    return {'architecture_changes': changed, 'setting_changes': settings,
            'first_curriculum': left_state['curriculum'], 'second_curriculum': right_state['curriculum']}


def merge_model(args):
    destination = Path(args.destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Merged-model destination must be empty.')
    model, tokenizer, config = restore(args.checkpoint, args.device, local_dir=args.local_model,
                                       cache_dir=args.cache_dir, offline=args.offline)
    if config['pretrained'].get('lora'):
        model = model.merge_and_unload(safe_merge=True)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination, safe_serialization=True, max_shard_size=args.shard_size)
    tokenizer.save_pretrained(destination)
    state, adapter = read_checkpoint(args.checkpoint)
    metadata = architecture_metadata(state, adapter)
    metadata.update(merged=True, checkpoint_sha256=ArtifactFile.inspect(Path(args.checkpoint).resolve().parent, Path(args.checkpoint).resolve()).sha256)
    (destination / 'task.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest='operation', required=True)
    inspect_parser = operations.add_parser('inspect')
    inspect_parser.add_argument('--checkpoint', required=True)
    bundle = operations.add_parser('package')
    bundle.add_argument('--checkpoint', required=True)
    bundle.add_argument('--destination', required=True)
    validate = operations.add_parser('verify')
    validate.add_argument('--directory', required=True)
    comparison = operations.add_parser('compare')
    comparison.add_argument('--first', required=True)
    comparison.add_argument('--second', required=True)
    merge = operations.add_parser('merge')
    merge.add_argument('--checkpoint', required=True)
    merge.add_argument('--destination', required=True)
    merge.add_argument('--local-model')
    merge.add_argument('--cache-dir', default='.cache/huggingface')
    merge.add_argument('--offline', action='store_true')
    merge.add_argument('--device', default='cuda')
    merge.add_argument('--shard-size', default='5GB')
    args = parser.parse_args(argv)
    if args.operation == 'inspect':
        result = inspect(args.checkpoint)
    elif args.operation == 'package':
        result = package(args.checkpoint, args.destination)
    elif args.operation == 'verify':
        result = verify(args.directory)
    elif args.operation == 'compare':
        result = compare(args.first, args.second)
    else:
        result = merge_model(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
