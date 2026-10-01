"""Typed SFT-to-DPO study plans and curriculum trajectory summaries."""
import argparse
import ast
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from pprint import pformat
import shlex
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]


def catalog_layout(paths):
    layout = {Path(path): Path(path) for path in paths}
    directories = sorted({parent for path in layout for parent in path.parents if parent != Path('.')},
                         key=lambda path: (-len(path.parts), str(path)))
    for directory in directories:
        direct = {path for path in layout.values() if path.parent == directory}
        candidates = sorted(key for key, path in layout.items()
                            if directory in path.parents and path.parent != directory)
        for key in candidates:
            if len(direct) >= 2:
                break
            current = layout[key]
            if sum(path.parent == current.parent for path in layout.values()) <= 2:
                continue
            destination = directory / '--'.join(current.relative_to(directory).parts)
            if destination in layout.values():
                raise ValueError(f'Conflicting recipe destination: {destination}')
            layout[key] = destination
            direct.add(destination)
    return layout


def load_recipe(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix in ('.yaml', '.yml'):
        config = yaml.safe_load(path.read_text())
    elif path.suffix == '.py':
        body = ast.parse(path.read_text(), filename=str(path)).body
        if len(body) != 1 or not isinstance(body[0], ast.Assign):
            raise ValueError('A Python recipe contains only CONFIG = {...}.')
        declaration = body[0]
        if (len(declaration.targets) != 1 or not isinstance(declaration.targets[0], ast.Name)
                or declaration.targets[0].id != 'CONFIG'):
            raise ValueError('Name the recipe dictionary CONFIG.')
        config = ast.literal_eval(declaration.value)
    else:
        raise ValueError(f'Expected a Python or YAML recipe: {path}')
    if not isinstance(config, dict):
        raise ValueError(f'Recipe must be a dictionary: {path}')
    return config


def save_recipe(path: Path, config: dict):
    text = ('CONFIG = ' + pformat(config, width=88, indent=4, sort_dicts=False) + '\n'
            if path.suffix == '.py' else yaml.safe_dump(config, sort_keys=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.with_suffix('.yaml' if path.suffix == '.py' else '.py').unlink(missing_ok=True)


def recipe_digest(path: Path) -> str:
    if path.suffix != '.py':
        return file_digest(path)
    serialized = yaml.safe_dump(load_recipe(path), sort_keys=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


@dataclass(frozen=True)
class Stage:
    identity: str
    phase: str
    output: str
    command: list
    prerequisites: list
    inputs: dict

    def as_record(self):
        return asdict(self)

    def validate_inputs(self):
        for filename, expected in self.inputs.items():
            path = Path(filename)
            if not path.is_file() or file_digest(path) != expected:
                raise ValueError(f'Study input changed: {filename}')
        for filename in self.prerequisites:
            if not Path(filename).is_file():
                raise FileNotFoundError(f'Complete the preceding stage first: {filename}')


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_inputs(paths):
    result = {}
    for path in paths:
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(path)] = file_digest(path)
    return result


def checkpoint_inputs(paths):
    from safety.checkpoint import read_checkpoint

    files = []
    for filename in paths:
        path = Path(filename).resolve()
        _, adapter = read_checkpoint(path)
        files.append(path)
        files.extend(sorted(value for value in adapter.rglob('*') if value.is_file()))
    return digest_inputs(files)


def native_command(operation, *arguments):
    return [sys.executable, str(ROOT / 'koscope.py'), operation, *map(str, arguments)]


def build_study(args):
    from koscope import validate_recipe
    root = Path(args.output).resolve()
    selected = [Path(path).resolve() for path in args.config]
    for path in selected:
        validate_recipe(load_recipe(path))
    stages = []
    for seed in args.seeds:
        if args.sft_checkpoint:
            initial = Path(args.sft_checkpoint).resolve()
            if not initial.is_file():
                raise FileNotFoundError(initial)
            supplied_checkpoint = checkpoint_inputs([initial])
        else:
            supplied_checkpoint = {}
            destination = root / args.model.replace('/', '--') / f'seed{seed}' / 'sft'
            command = native_command('train', '--stage', 'sft', '--model', args.model,
                '--config', selected[0], '--data', Path(args.sft_data).resolve(),
                '--validation', Path(args.validation).resolve(), '--seed', seed,
                '--device', args.device, '--output', destination)
            if args.local_model:
                command.extend(['--local-model', str(Path(args.local_model).resolve())])
            if args.offline:
                command.append('--offline')
            stages.append(Stage(f'seed{seed}-sft', 'sft', str(destination), command, [],
                                digest_inputs([selected[0], args.sft_data, args.validation])))
            initial = destination / 'last.pt'
        for path in selected:
            key = recipe_digest(path)[:12]
            destination = root / args.model.replace('/', '--') / f'seed{seed}' / key / 'dpo'
            command = native_command('train', '--stage', 'dpo', '--checkpoint', initial,
                '--config', path, '--data', Path(args.preferences).resolve(),
                '--validation', Path(args.validation).resolve(), '--seed', seed,
                '--device', args.device, '--output', destination)
            if args.local_model:
                command.extend(['--local-model', str(Path(args.local_model).resolve())])
            if args.offline:
                command.append('--offline')
            inputs = digest_inputs([path, args.preferences, args.validation])
            inputs.update(supplied_checkpoint)
            stages.append(Stage(f'seed{seed}-{key}-dpo', 'dpo', str(destination), command,
                                [str(initial)], inputs))
            if args.test:
                evaluation = destination.parent / 'evaluation'
                command = native_command('benchmark', '--checkpoint', destination / 'last.pt',
                    '--data', Path(args.test).resolve(), '--output', evaluation, '--device', args.device)
                if args.local_model:
                    command.extend(['--local-model', str(Path(args.local_model).resolve())])
                if args.offline:
                    command.append('--offline')
                stages.append(Stage(f'seed{seed}-{key}-eval', 'evaluation', str(evaluation), command,
                                    [str(destination / 'last.pt')], digest_inputs([args.test])))
    names = [stage.identity for stage in stages]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate recipe contents or seeds generate overlapping run identities.')
    return {'format': 1, 'repository': str(ROOT), 'stages': [stage.as_record() for stage in stages]}


def completed(stage):
    output = Path(stage.output)
    marker = output / 'study-stage.json'
    if not marker.exists():
        return False
    record = json.loads(marker.read_text())
    artifact = output / ('metrics.json' if stage.phase == 'evaluation' else 'last.pt')
    return (record.get('returncode') == 0 and record.get('stage') == stage.as_record()
            and artifact.is_file()
            and record.get('prerequisites') == checkpoint_inputs(stage.prerequisites))


def execute(plan, skip_complete=False):
    results = []
    for record in plan['stages']:
        stage = Stage(**record)
        stage.validate_inputs()
        if skip_complete and completed(stage):
            results.append({'id': stage.identity, 'status': 'skipped'})
            continue
        output = Path(stage.output)
        if (output / 'last.pt').exists() or (output / 'study-stage.json').exists():
            raise FileExistsError(f'Stage output exists: {output}')
        output.mkdir(parents=True, exist_ok=True)
        prerequisites = checkpoint_inputs(stage.prerequisites)
        start = time.monotonic()
        with (output / 'console.log').open('w') as stream:
            result = subprocess.run(stage.command, cwd=plan['repository'], stdout=stream, stderr=subprocess.STDOUT)
        report = {'stage': stage.as_record(), 'returncode': result.returncode,
                  'elapsed_seconds': time.monotonic() - start, 'prerequisites': prerequisites}
        (output / 'study-stage.json').write_text(json.dumps(report, indent=2) + '\n')
        results.append(report)
        if result.returncode:
            raise RuntimeError(f'Stage failed: {stage.identity}. Inspect {output / "console.log"}')
    return results


def trajectory(history):
    previous = {}
    rows = []
    for record in history:
        epoch = int(record['epoch'])
        state = record.get('curriculum', {})
        unlocked = {int(key): int(value) for key, value in state.get('unlock_epoch', {}).items()}
        if any(level in previous and previous[level] != start for level, start in unlocked.items()):
            raise ValueError('Unlock epochs changed within the training history.')
        if not set(previous) <= set(unlocked):
            raise ValueError('A curriculum level was relocked in the recorded history.')
        previous = unlocked
        row = {'epoch': epoch, 'loss': record['loss'],
               'highest_unlocked': max(unlocked, default=1),
               'explicit_probability': record.get('explicit_probability')}
        for level in (1, 2, 3):
            row[f'level{level}_unlocked_at'] = unlocked.get(level)
            row[f'level{level}_loss_ema'] = state.get('ema', {}).get(str(level), state.get('ema', {}).get(level))
            row[f'level{level}_probe_accuracy'] = record.get('probe_accuracy', {}).get(str(level), record.get('probe_accuracy', {}).get(level))
        if not math.isfinite(float(row['loss'])):
            raise ValueError('Training history contains a nonfinite loss.')
        rows.append(row)
    return rows


def export_history(path, output):
    history = json.loads(Path(path).read_text())
    if not isinstance(history, list) or not history:
        raise ValueError('Expected the nonempty metrics.json list from preference training.')
    rows = trajectory(history)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'curriculum.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'epochs': len(rows), 'initial_loss': rows[0]['loss'], 'final_loss': rows[-1]['loss'],
               'best_recorded_loss': min(row['loss'] for row in rows),
               'unlocked': {str(level): rows[-1][f'level{level}_unlocked_at'] for level in (1, 2, 3)}}
    (output / 'curriculum.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest='operation', required=True)
    plan = operations.add_parser('plan')
    plan.add_argument('--config', action='append', required=True)
    plan.add_argument('--model', default='qwen3-4b')
    plan.add_argument('--sft-data')
    plan.add_argument('--sft-checkpoint')
    plan.add_argument('--preferences', required=True)
    plan.add_argument('--validation', required=True)
    plan.add_argument('--test')
    plan.add_argument('--local-model')
    plan.add_argument('--offline', action='store_true')
    plan.add_argument('--seeds', nargs='+', type=int, default=[42])
    plan.add_argument('--device', default='cuda')
    plan.add_argument('--output', required=True)
    run = operations.add_parser('run')
    run.add_argument('--plan', required=True)
    run.add_argument('--skip-complete', action='store_true')
    history = operations.add_parser('history')
    history.add_argument('--history', required=True)
    history.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.operation == 'plan':
        if not args.sft_checkpoint and not args.sft_data:
            parser.error('Provide --sft-data or --sft-checkpoint.')
        result = build_study(args)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / 'study.json').write_text(json.dumps(result, indent=2) + '\n')
        (output / 'commands.txt').write_text('\n'.join(shlex.join(row['command']) for row in result['stages']) + '\n')
        print(json.dumps({'stages': len(result['stages']), 'output': str(output)}))
    elif args.operation == 'run':
        result = execute(json.loads(Path(args.plan).read_text()), args.skip_complete)
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(export_history(args.history, args.output), indent=2))
