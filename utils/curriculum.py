import math

import torch


class HierarchicalCurriculum:
    def __init__(self, rho=0.1, thresholds=(0.35, 0.30), probe_threshold=0.85,
                 gamma0=2.0, kappa=0.08, ramps=(8, 8, 8), lower_replay=0.2,
                 explicit_start=48, explicit_fade=32):
        if not 0 < rho <= 1 or not 0 <= lower_replay < 1:
            raise ValueError('Invalid EMA rate or replay floor.')
        if len(ramps) != 3 or min(ramps) <= 0 or explicit_fade <= 0:
            raise ValueError('Three positive ramp lengths and a positive fade length are required.')
        self.rho, self.thresholds, self.probe_threshold = rho, thresholds, probe_threshold
        self.gamma0, self.kappa, self.ramps = gamma0, kappa, ramps
        self.lower_replay = lower_replay
        self.explicit_start, self.explicit_fade = explicit_start, explicit_fade
        self.unlock_epoch = {1: 0}
        self.ema = {}

    def update(self, epoch, level_losses, probe_accuracy):
        # At most one sequential unlock after the epoch; use the next epoch as start.
        for level, loss in level_losses.items():
            if level in self.unlock_epoch:
                previous = self.ema.get(level, float(loss))
                self.ema[level] = (1 - self.rho) * previous + self.rho * float(loss)
        highest = max(self.unlock_epoch)
        if (highest < 3 and self.ema.get(highest, math.inf) < self.thresholds[highest - 1]
                and probe_accuracy.get(highest, 0) > self.probe_threshold):
            self.unlock_epoch[highest + 1] = epoch + 1

    def margin(self, level, epoch):
        if level not in self.unlock_epoch:
            raise ValueError('Cannot train a locked level.')
        return self.gamma0 * math.exp(-self.kappa * max(0, epoch - self.unlock_epoch[level]))

    def hard_probability(self, level, epoch):
        if level not in self.unlock_epoch:
            return 0.0
        return min(1.0, max(0.0, (epoch - self.unlock_epoch[level]) / self.ramps[level - 1]))

    def level_probabilities(self, epoch):
        available = sorted(level for level, start in self.unlock_epoch.items() if start <= epoch)
        # The manuscript leaves w_l(e) unspecified. This explicit default increases
        # the newest level's logit during its hard-negative ramp.
        logits = torch.tensor([(level - 1) * self.hard_probability(level, epoch) for level in available])
        probabilities = logits.softmax(0)
        if len(available) > 1 and probabilities[:-1].sum() < self.lower_replay:
            probabilities[:-1] *= self.lower_replay / probabilities[:-1].sum()
            probabilities[-1] = 1 - self.lower_replay
        return dict(zip(available, probabilities.tolist()))

    def explicit_probability(self, epoch):
        if 3 not in self.unlock_epoch:
            return 1.0
        start = max(self.explicit_start, self.unlock_epoch[3] + self.ramps[2])
        return max(0.0, 1 - max(0, epoch - start) / self.explicit_fade)

    def state_dict(self):
        return {'unlock_epoch': self.unlock_epoch.copy(), 'ema': self.ema.copy()}

    def load_state_dict(self, state):
        self.unlock_epoch = {int(k): v for k, v in state['unlock_epoch'].items()}
        self.ema = {int(k): v for k, v in state['ema'].items()}
