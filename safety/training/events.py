"""Append-only update records and competence transition summaries."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time


def finite_tree(value, path='event'):
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f'{path}.{key}')
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            finite_tree(child, f'{path}[{index}]')
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'Nonfinite telemetry value at {path}.')


class TrainingEvents:
    def __init__(self, destination, resume_epoch=0):
        self.path = Path(destination)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.sequence = 0
        self.rows = Counter()
        self.updates = Counter()
        self._stream = None
        if self.path.exists():
            entries = read_events(self.path, allow_incomplete_tail=resume_epoch > 0)
            committed = [row for row in entries if row.get('epoch', -1) < resume_epoch]
            if entries and resume_epoch == 0:
                raise FileExistsError('Training events already exist. Resume or choose another output folder.')
            if len(committed) != len(entries) or not self.path.read_bytes().endswith(b'\n'):
                temporary = self.path.with_name(self.path.name + '.partial')
                temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in committed), encoding='utf-8')
                temporary.replace(self.path)
            self.sequence = max((row['sequence'] for row in committed), default=-1) + 1

    def open(self):
        if self._stream is not None:
            raise RuntimeError('The event stream is already open.')
        self._stream = self.path.open('a', encoding='utf-8')
        return self

    def close(self):
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self):
        return self.open()

    def __exit__(self, error_type, error, traceback):
        self.close()

    def write(self, kind, epoch, **values):
        if self._stream is None:
            raise RuntimeError('Open the event stream before writing.')
        event = {'sequence': self.sequence, 'kind': kind, 'epoch': int(epoch),
                 'elapsed_seconds': time.monotonic() - self.started, **values}
        finite_tree(event)
        self._stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
        self._stream.flush()
        self.sequence += 1
        return event

    def update(self, epoch, step, loss, rate, gradient, records):
        levels = Counter(str(row['level']) for row in records)
        sources = Counter(str(row.get('source', 'unspecified')) for row in records)
        self.rows.update(levels)
        self.updates[epoch] += 1
        return self.write('update', epoch, step=step, loss=loss, learning_rate=rate,
                          gradient_norm=gradient, examples=len(records), levels=dict(levels), sources=dict(sources))

    def epoch(self, result, sampler):
        return self.write('epoch', result['epoch'], result=result, sampling=sampler.report())


def read_events(path, allow_incomplete_tail=False):
    events = []
    lines = Path(path).read_bytes().splitlines(keepends=True)
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if allow_incomplete_tail and number == len(lines) and not line.endswith(b'\n'):
                break
            raise ValueError(f'Invalid event at {path}:{number}.') from error
        finite_tree(row)
        if not isinstance(row, dict):
            raise ValueError('Training events must contain JSON objects.')
        if row.get('kind') not in ('update', 'epoch'):
            raise ValueError('Unknown training event kind.')
        if row.get('sequence') != len(events):
            raise ValueError('Event sequence must start at zero and remain consecutive.')
        events.append(row)
    return events


def summarize_events(events):
    epochs = defaultdict(list)
    outcomes = {}
    for row in events:
        if row['kind'] == 'update':
            epochs[row['epoch']].append(row)
        else:
            if row['epoch'] in outcomes:
                raise ValueError('An epoch has multiple completion records.')
            outcomes[row['epoch']] = row['result']
    summaries = []
    previous_levels = {1}
    for epoch in sorted(set(epochs) | set(outcomes)):
        updates = epochs[epoch]
        result = outcomes.get(epoch)
        levels, sources = Counter(), Counter()
        for row in updates:
            levels.update(row['levels'])
            sources.update(row['sources'])
        current_levels = {int(level) for level in result['curriculum']['unlock_epoch']} if result else previous_levels
        summaries.append({'epoch': epoch, 'complete': result is not None, 'logged_updates': len(updates),
                          'logged_examples': sum(row['examples'] for row in updates),
                          'mean_update_loss': sum(row['loss'] for row in updates) / len(updates) if updates else None,
                          'maximum_gradient_norm': max((row['gradient_norm'] for row in updates), default=None),
                          'first_learning_rate': updates[0]['learning_rate'] if updates else None,
                          'last_learning_rate': updates[-1]['learning_rate'] if updates else None,
                          'level_draws': dict(levels), 'source_draws': dict(sources),
                          'newly_unlocked': sorted(current_levels - previous_levels),
                          'probe_accuracy': result.get('probe_accuracy') if result else None})
        previous_levels = current_levels
    return {'epochs': summaries, 'completed_epochs': len(outcomes), 'events': len(events)}


def export_updates(events, destination):
    import csv
    fields = ['epoch', 'step', 'loss', 'learning_rate', 'gradient_norm', 'examples', 'elapsed_seconds']
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in events:
            if row['kind'] == 'update':
                writer.writerow({field: row[field] for field in fields})


def competence_diagnostics(events, regression_tolerance=.05):
    if not 0 <= regression_tolerance <= 1:
        raise ValueError('Regression tolerance must lie in [0,1].')
    history = [row['result'] for row in events if row['kind'] == 'epoch']
    traces = {level: [] for level in (1, 2, 3)}
    best = {}
    regressions = []
    explicit = []
    for result in history:
        epoch = result['epoch']
        losses = {int(level): value for level, value in result.get('level_losses', {}).items()}
        accuracy = {int(level): value for level, value in result['probe_accuracy'].items()}
        ema = {int(level): value for level, value in result['curriculum']['ema'].items()}
        unlock = {int(level): value for level, value in result['curriculum']['unlock_epoch'].items()}
        for level in (1, 2, 3):
            value = accuracy.get(level)
            if value is not None and not 0 <= value <= 1:
                raise ValueError('Probe accuracy lies outside [0,1].')
            if value is not None:
                previous_best = best.get(level, value)
                if previous_best - value > regression_tolerance:
                    regressions.append({'epoch': epoch, 'level': level,
                                        'best_accuracy': previous_best, 'accuracy': value,
                                        'difference': value - previous_best})
                best[level] = max(previous_best, value)
            traces[level].append({'epoch': epoch, 'loss': losses.get(level), 'loss_ema': ema.get(level),
                                  'probe_accuracy': value, 'active': unlock.get(level, math.inf) <= epoch})
        probability = result['explicit_probability']
        if not 0 <= probability <= 1:
            raise ValueError('Explicit reasoning probability lies outside [0,1].')
        explicit.append({'epoch': epoch, 'probability': probability})
    return {'levels': traces, 'regressions': regressions, 'regression_tolerance': regression_tolerance,
            'best_probe_accuracy': best, 'explicit_reasoning': explicit}


def update_outliers(events, window=20, multiplier=3.0):
    import statistics
    if window < 3 or multiplier <= 0:
        raise ValueError('Use a window of at least three updates and a positive multiplier.')
    updates = [row for row in events if row['kind'] == 'update']
    outliers = []
    for index in range(window, len(updates)):
        previous = updates[index - window:index]
        values = [row['loss'] for row in previous]
        center = statistics.median(values)
        deviations = [abs(value - center) for value in values]
        deviation = statistics.median(deviations)
        scale = max(1e-12, 1.4826 * deviation)
        current = updates[index]
        standardized = (current['loss'] - center) / scale
        if standardized > multiplier:
            outliers.append({'epoch': current['epoch'], 'step': current['step'],
                             'loss': current['loss'], 'rolling_median': center,
                             'rolling_mad': deviation, 'standardized_excess': standardized})
    return {'window': window, 'multiplier': multiplier, 'outliers': outliers}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--updates-csv')
    parser.add_argument('--regression-tolerance', type=float, default=.05)
    args = parser.parse_args(argv)
    events = read_events(args.events)
    report = summarize_events(events)
    report['competence'] = competence_diagnostics(events, args.regression_tolerance)
    report['loss_outliers'] = update_outliers(events)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    if args.updates_csv:
        export_updates(events, args.updates_csv)
    print(json.dumps({'output': str(output), 'events': len(events)}))
