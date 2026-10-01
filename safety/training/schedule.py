"""Controlled curriculum ablations and replay of observed competence histories."""
import argparse
import copy
import json
import math
from pathlib import Path

import yaml


POLICIES = ('full', 'no_curriculum', 'fixed_schedule', 'no_margin_decay', 'no_ccr', 'explicit_only', 'implicit_only')


def configure_ablation(config, name='full'):
    if name not in POLICIES:
        raise ValueError(f'Unknown ablation {name}.')
    result = copy.deepcopy(config)
    options = {'name': name, 'schedule': 'competence', 'reasoning': 'fade'}
    if name == 'no_curriculum':
        options['schedule'] = 'all_levels'
    elif name == 'fixed_schedule':
        options['schedule'] = 'fixed'
        options['unlock_epochs'] = [0, 16, 32]
    elif name == 'no_margin_decay':
        result['curriculum']['kappa'] = 0.0
    elif name == 'no_ccr':
        result['lambda_ccr'] = 0.0
    elif name == 'explicit_only':
        options['reasoning'] = 'explicit'
    elif name == 'implicit_only':
        options['reasoning'] = 'implicit'
    result['ablation'] = options
    return result


class CurriculumPolicy:
    def __init__(self, curriculum, options=None):
        self.base = curriculum
        self.options = options or {'name': 'full', 'schedule': 'competence', 'reasoning': 'fade'}
        mode = self.options.get('schedule', 'competence')
        if mode not in ('competence', 'all_levels', 'fixed'):
            raise ValueError('Invalid curriculum schedule.')
        reasoning = self.options.get('reasoning', 'fade')
        if reasoning not in ('fade', 'explicit', 'implicit'):
            raise ValueError('Invalid reasoning schedule.')
        if mode == 'all_levels':
            self.base.unlock_epoch = {1: 0, 2: 0, 3: 0}
        elif mode == 'fixed':
            epochs = self.options.get('unlock_epochs', [0, 16, 32])
            if len(epochs) != 3 or epochs[0] != 0 or any(not isinstance(e, int) or e < 0 for e in epochs):
                raise ValueError('Three nonnegative unlock epochs starting at zero are required.')
            if epochs != sorted(epochs):
                raise ValueError('Fixed unlock epochs must be ordered.')
            self.base.unlock_epoch = dict(zip((1, 2, 3), epochs))

    def __getattr__(self, name):
        if name == 'base':
            raise AttributeError(name)
        return getattr(self.base, name)

    def update(self, epoch, level_losses, probe_accuracy):
        if self.options.get('schedule', 'competence') == 'competence':
            return self.base.update(epoch, level_losses, probe_accuracy)
        for level, loss in level_losses.items():
            if self.base.unlock_epoch.get(level, math.inf) <= epoch:
                previous = self.base.ema.get(level, float(loss))
                self.base.ema[level] = (1 - self.base.rho) * previous + self.base.rho * float(loss)

    def level_probabilities(self, epoch):
        if self.options.get('schedule') == 'all_levels':
            return {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}
        return self.base.level_probabilities(epoch)

    def explicit_probability(self, epoch):
        if self.options.get('reasoning') == 'explicit':
            return 1.0
        if self.options.get('reasoning') == 'implicit':
            return 0.0
        return self.base.explicit_probability(epoch)

    def state_dict(self):
        return self.base.state_dict()

    def load_state_dict(self, state):
        self.base.load_state_dict(state)


def replay_history(config, history):
    from safety.models.learner import HierarchicalCurriculum
    curriculum = CurriculumPolicy(HierarchicalCurriculum(**config['curriculum']), config.get('ablation'))
    result = []
    for expected, row in enumerate(history):
        epoch = row['epoch']
        if epoch != expected:
            raise ValueError('Competence history must contain consecutive epochs starting at zero.')
        probabilities = curriculum.level_probabilities(epoch)
        margins = {level: curriculum.margin(level, epoch) for level in probabilities}
        hard = {level: curriculum.hard_probability(level, epoch) for level in probabilities}
        losses = {int(level): float(loss) for level, loss in row['level_losses'].items()}
        accuracy = {int(level): float(value) for level, value in row['probe_accuracy'].items()}
        if any(not math.isfinite(value) or value < 0 for value in losses.values()):
            raise ValueError('Observed competence losses must be finite and nonnegative.')
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in accuracy.values()):
            raise ValueError('Observed probe accuracies must lie in [0,1].')
        before = curriculum.state_dict()
        explicit = curriculum.explicit_probability(epoch)
        curriculum.update(epoch, losses, accuracy)
        result.append({'epoch': epoch, 'level_probabilities': probabilities, 'margins': margins,
                       'hard_probabilities': hard, 'explicit_probability': explicit,
                       'before_update': before, 'after_update': curriculum.state_dict()})
    return result


def compare_replays(config, history, policies):
    reports = {}
    for name in policies:
        reports[name] = replay_history(configure_ablation(config, name), history)
    transitions = {}
    for name, rows in reports.items():
        transitions[name] = rows[-1]['after_update']['unlock_epoch'] if rows else {1: 0}
    return {'policies': reports, 'unlock_epochs': transitions, 'observations': len(history)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--history', required=True)
    parser.add_argument('--policies', nargs='+', choices=POLICIES, default=['full'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    from safety.study import load_recipe
    config = load_recipe(args.config)
    text = Path(args.history).read_text(encoding='utf-8')
    history = yaml.safe_load(text) if Path(args.history).suffix in ('.yaml', '.yml') else json.loads(text)
    if not isinstance(history, list) or not history:
        raise ValueError('History must be a nonempty list of observed epoch losses and probe accuracies.')
    report = compare_replays(config, history, args.policies)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'policies': args.policies}))
