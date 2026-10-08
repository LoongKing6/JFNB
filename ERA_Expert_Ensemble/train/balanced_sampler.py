import math

import numpy as np
from torch.utils.data import Sampler


class SubjectClassBalancedSampler(Sampler):
    """Sample subject-class groups uniformly, then sample within each group."""

    def __init__(self, subject_ids, labels, num_samples=None, seed=0):
        self.subject_ids = np.asarray(subject_ids).reshape(-1)
        self.labels = np.asarray(labels).reshape(-1)
        if len(self.subject_ids) != len(self.labels):
            raise ValueError('subject_ids and labels must have the same length.')
        self.num_samples = int(num_samples or len(self.labels))
        self.seed = int(seed)
        self.epoch = 0

        self.groups = {}
        for index, key in enumerate(zip(self.subject_ids.tolist(), self.labels.tolist())):
            self.groups.setdefault(key, []).append(index)
        self.group_keys = sorted(self.groups)
        if not self.group_keys:
            raise ValueError('Cannot build a sampler for an empty dataset.')

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        group_order = np.arange(len(self.group_keys))
        output = []
        while len(output) < self.num_samples:
            rng.shuffle(group_order)
            for group_index in group_order:
                members = self.groups[self.group_keys[int(group_index)]]
                output.append(int(rng.choice(members)))
                if len(output) >= self.num_samples:
                    break
        self.epoch += 1
        return iter(output)

    def __len__(self):
        return self.num_samples


def balanced_steps(num_samples, batch_size):
    return int(math.ceil(float(num_samples) / float(batch_size)))
