import random

from .records import load_records, synthetic_records


class LevelExperience:
    def __init__(self, epoch, records):
        self.current_experience = epoch
        self.records = records

    def sample(self, batch_size, curriculum, stage):
        probabilities = ({1: 1 / 3, 2: 1 / 3, 3: 1 / 3} if stage == 'sft'
                         else curriculum.level_probabilities(self.current_experience))
        levels = random.choices(list(probabilities), weights=list(probabilities.values()), k=batch_size)
        return [random.choice(self.records[level]) for level in levels]


class LevelBenchmark:
    """Disjoint safety-level train/evaluation streams for a preference experiment."""
    def __init__(self, train, validation, epochs):
        self.levels = {level: [row for row in train if row['level'] == level] for level in (1, 2, 3)}
        if any(not rows for rows in self.levels.values()):
            raise ValueError('Training data must contain all three levels.')
        if {row['level'] for row in validation} != {1, 2, 3}:
            raise ValueError('Validation data must contain all three levels for competence gates.')
        if {row['prompt'] for row in train} & {row['prompt'] for row in validation}:
            raise ValueError('Training and validation prompts must be disjoint.')
        self.epochs = epochs
        self.test_stream = validation

    @property
    def train_stream(self):
        return (LevelExperience(epoch, self.levels) for epoch in range(self.epochs))

    @classmethod
    def from_paths(cls, data, validation, epochs):
        if data and not validation:
            raise ValueError('Real-data training requires a disjoint --validation JSONL.')
        train = load_records(data) if data else synthetic_records()
        test = load_records(validation) if validation else synthetic_records(24, 1000)
        return cls(train, test, epochs)
