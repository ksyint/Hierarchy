"""Native recipe discovery and validation for Korean preference experiments."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'configs' / 'experiments'


def recipes():
    return {path.relative_to(CATALOG).with_suffix('').as_posix(): path
            for path in sorted(CATALOG.rglob('*.yaml'))}


def resolve_recipe(name):
    available = recipes()
    if name not in available:
        raise ValueError(f'Unknown recipe {name!r}. Use --list-recipes to inspect available settings.')
    return available[name]


def validate_recipe(config):
    required = {'seed', 'model_dim', 'batch_size', 'epochs', 'steps_per_epoch', 'lr',
                'weight_decay', 'beta', 'lambda_ccr', 'max_length', 'decision_tokens', 'curriculum'}
    if not required <= config.keys():
        raise ValueError(f'Missing recipe keys: {sorted(required - config.keys())}')
    if any(config[key] <= 0 for key in ('model_dim', 'batch_size', 'epochs', 'steps_per_epoch', 'lr', 'beta', 'max_length')):
        raise ValueError('Model, batching, optimization, beta, and sequence length settings must be positive.')
    if len(config['decision_tokens']) != 2 or not all(isinstance(token, str) and token for token in config['decision_tokens']):
        raise ValueError('Specify two nonempty decision verbalizers in [comply, refuse] order.')
    curriculum = config['curriculum']
    if len(curriculum['thresholds']) != 2 or min(curriculum['thresholds']) <= 0:
        raise ValueError('The hierarchy requires positive Level-1 and Level-2 loss thresholds.')
    if not 0 <= curriculum['probe_threshold'] <= 1 or not 0 <= curriculum['lower_replay'] < 1:
        raise ValueError('Probe threshold and replay probability are outside their valid ranges.')
    if len(curriculum['ramps']) != 3 or min(curriculum['ramps']) <= 0:
        raise ValueError('Specify a positive hard-negative ramp for each of three levels.')
    if curriculum['gamma0'] < 0 or curriculum['kappa'] < 0 or not 0 < curriculum['rho'] <= 1:
        raise ValueError('Invalid margin decay or competence EMA settings.')
    if config['lambda_ccr'] < 0 or config['weight_decay'] < 0 or curriculum['explicit_fade'] <= 0:
        raise ValueError('Regularizer weights must be nonnegative and the fade duration positive.')
    return config
