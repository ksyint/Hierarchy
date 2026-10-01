"""Build preference recipes as dataset/margin/temperature/gate/decay/replay axes."""
import copy
from itertools import product

import yaml

from .catalog import CATALOG, ROOT, validate_recipe


GAMMA = {'g05': 0.5, 'g10': 1.0, 'g20': 2.0}
BETA = {'b005': 0.05, 'b010': 0.1, 'b020': 0.2}
GATES = {'conservative': ([0.25, 0.20], 0.90),
         'standard': ([0.35, 0.30], 0.85),
         'permissive': ([0.45, 0.40], 0.80)}
DECAY = {'k004': 0.04, 'k008': 0.08, 'k012': 0.12}
REPLAY = {'r020': 0.20, 'r030': 0.30, 'r040': 0.40}


def main():
    base = yaml.safe_load((ROOT / 'configs/korean/harm.yaml').read_text())
    count = 0
    for gamma, beta, gate, decay, replay in product(GAMMA, BETA, GATES, DECAY, REPLAY):
        config = copy.deepcopy(base)
        config['beta'] = BETA[beta]
        thresholds, probe_threshold = GATES[gate]
        config['curriculum'].update(gamma0=GAMMA[gamma], thresholds=list(thresholds),
                                    probe_threshold=probe_threshold, kappa=DECAY[decay],
                                    lower_replay=REPLAY[replay])
        validate_recipe(config)
        path = CATALOG / 'korean' / gamma / beta / gate / decay / (replay + '.yaml')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        count += 1
    print(f'Built {count} preference recipes.')


if __name__ == '__main__':
    main()
