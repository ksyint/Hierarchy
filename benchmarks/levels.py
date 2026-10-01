import random

from .records import load_records, record_prompts


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
        training_prompts = {prompt for row in train for prompt in record_prompts(row)}
        validation_prompts = {prompt for row in validation for prompt in record_prompts(row)}
        if training_prompts & validation_prompts:
            raise ValueError('Training and validation primary/counterfactual prompts must be disjoint.')
        self.epochs = epochs
        self.test_stream = validation

    @property
    def train_stream(self):
        return (LevelExperience(epoch, self.levels) for epoch in range(self.epochs))

    @classmethod
    def from_paths(cls, data, validation, epochs):
        if not data or not validation:
            raise ValueError('Training requires --data and a disjoint --validation JSONL.')
        train = load_records(data)
        test = load_records(validation)
        return cls(train, test, epochs)
