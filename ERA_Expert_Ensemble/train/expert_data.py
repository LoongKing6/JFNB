import json
import os
from dataclasses import dataclass

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


ERA_GRAPH_LAYOUTS = {
    'hem': [
        ['Fp1'], ['F3', 'F7'], ['FT7', 'FC3'], ['C3', 'T3'], ['TP7', 'CP3'],
        ['T5', 'P3'], ['O1'], ['Fpz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'Oz'],
        ['A1'], ['Fp2'], ['F4', 'F8'], ['FT8', 'FC4'], ['C4', 'T4'],
        ['CP4', 'TP8'], ['P4', 'T6'], ['O2']
    ],
    'fro': [
        ['Fp1'], ['Fp2'], ['F3', 'F7'], ['F4', 'F8'], ['Fpz', 'Fz', 'FCz'],
        ['FT7', 'FC3'], ['FT8', 'FC4'], ['C3', 'Cz', 'C4'],
        ['TP7', 'CP3', 'CPz', 'CP4', 'TP8'], ['T5', 'P3', 'Pz', 'P4', 'T6'],
        ['O1', 'Oz', 'O2'], ['T3'], ['T4'], ['A1']
    ],
    'gen': [
        ['Fp1', 'Fp2', 'Fpz'], ['F3', 'F7', 'Fz', 'F4', 'F8'],
        ['FT7', 'FC3', 'FCz', 'FT8', 'FC4'], ['C3', 'Cz', 'C4'],
        ['TP7', 'CP3', 'CPz', 'CP4', 'TP8'], ['T5', 'P3', 'Pz', 'P4', 'T6'],
        ['O1', 'Oz', 'O2'], ['T3'], ['T4'], ['A1']
    ]
}


@dataclass
class SubjectData:
    data: np.ndarray
    label: np.ndarray
    sample_id: np.ndarray
    trial_id: np.ndarray
    segment_id: np.ndarray
    subject_id: np.ndarray
    channel_order: list
    graph_structure: list
    source_path: str


class ExpertDataset(Dataset):
    def __init__(self, data, labels, teacher_logits=None, subject_ids=None,
                 sample_ids=None, teacher_weights=None):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.teacher_logits = None if teacher_logits is None else torch.as_tensor(
            teacher_logits, dtype=torch.float32)
        self.subject_ids = None if subject_ids is None else torch.as_tensor(subject_ids, dtype=torch.long)
        self.sample_ids = None if sample_ids is None else torch.as_tensor(sample_ids, dtype=torch.long)
        self.teacher_weights = None if teacher_weights is None else torch.as_tensor(
            teacher_weights, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        result = [self.data[index], self.labels[index]]
        if self.teacher_logits is not None:
            result.append(self.teacher_logits[index])
        if self.subject_ids is not None:
            result.append(self.subject_ids[index])
        if self.sample_ids is not None:
            result.append(self.sample_ids[index])
        if self.teacher_weights is not None:
            result.append(self.teacher_weights[index])
        return tuple(result)


def parse_subject_list(args):
    if args.subject_list:
        return [int(value.strip()) for value in args.subject_list.split(',') if value.strip()]
    return list(range(args.subjects))


def resolve_data_dir(args):
    if args.data_dir:
        return os.path.abspath(args.data_dir)
    name = 'data_{}_{}_{}'.format(args.data_format, args.dataset, args.label_type)
    return os.path.abspath(name)


def _flatten_data(data):
    if data.ndim == 5:
        return data.reshape((-1,) + data.shape[2:])
    if data.ndim == 4:
        return data
    if data.ndim == 3:
        return np.expand_dims(data, axis=1)
    raise ValueError('Unsupported EEG data shape: {}'.format(data.shape))


def _flatten_label(label):
    return np.asarray(label, dtype=np.int64).reshape(-1)


def _default_metadata(data, flat_count, subject):
    if data.ndim == 5:
        trial_count, segment_count = data.shape[:2]
        trial_id = np.repeat(np.arange(trial_count, dtype=np.int64), segment_count)
        segment_id = np.tile(np.arange(segment_count, dtype=np.int64), trial_count)
    else:
        trial_id = np.arange(flat_count, dtype=np.int64)
        segment_id = np.zeros(flat_count, dtype=np.int64)
    return {
        'sample_id': np.arange(flat_count, dtype=np.int64),
        'trial_id': trial_id,
        'segment_id': segment_id,
        'subject_id': np.full(flat_count, subject, dtype=np.int64)
    }


def _read_json_attr(dataset, name, default):
    value = dataset.attrs.get(name)
    if value is None:
        return default
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def load_subject_hdf(args, subject):
    path = os.path.join(resolve_data_dir(args), 'sub{}.hdf'.format(subject))
    if not os.path.exists(path):
        raise FileNotFoundError('Subject HDF not found: {}'.format(path))

    with h5py.File(path, 'r') as dataset:
        raw_data = np.asarray(dataset['data'], dtype=np.float32)
        data = _flatten_data(raw_data)
        label = _flatten_label(dataset['label'])
        if len(data) != len(label):
            raise ValueError('Data/label length mismatch in {}: {} vs {}'.format(path, len(data), len(label)))

        defaults = _default_metadata(raw_data, len(data), subject)
        metadata = {}
        for key, default in defaults.items():
            value = np.asarray(dataset[key]) if key in dataset else default
            metadata[key] = value.reshape(-1).astype(np.int64)
            if len(metadata[key]) != len(data):
                metadata[key] = default

        channel_order = _read_json_attr(dataset, 'channel_order', [])
        graph_structure = _read_json_attr(dataset, 'graph_structure', [])

    if not graph_structure and args.dataset == 'EMO':
        graph_structure = ERA_GRAPH_LAYOUTS.get(args.graph_type, [])
    if not channel_order and graph_structure:
        channel_order = [channel for group in graph_structure for channel in group]

    return SubjectData(
        data=data,
        label=label,
        sample_id=metadata['sample_id'],
        trial_id=metadata['trial_id'],
        segment_id=metadata['segment_id'],
        subject_id=metadata['subject_id'],
        channel_order=channel_order,
        graph_structure=graph_structure,
        source_path=path
    )


def compute_normalization(data):
    if data.ndim != 4:
        raise ValueError('Expected data shaped [sample, feature, channel, time], got {}'.format(data.shape))
    mean = np.mean(data, axis=(0, 1, 3), keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.std(data, axis=(0, 1, 3), keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
    return mean, std


def apply_normalization(data, mean, std):
    return ((np.asarray(data, dtype=np.float32) - mean) / std).astype(np.float32)


def normalization_to_checkpoint(mean, std):
    return {
        'normalization_mean': torch.as_tensor(mean.reshape(-1), dtype=torch.float32),
        'normalization_std': torch.as_tensor(std.reshape(-1), dtype=torch.float32)
    }


def normalization_from_checkpoint(checkpoint):
    mean = checkpoint.get('normalization_mean')
    std = checkpoint.get('normalization_std')
    if mean is None or std is None:
        raise KeyError('Checkpoint does not contain normalization_mean/normalization_std.')
    mean = np.asarray(mean.detach().cpu() if torch.is_tensor(mean) else mean, dtype=np.float32)
    std = np.asarray(std.detach().cpu() if torch.is_tensor(std) else std, dtype=np.float32)
    return mean.reshape(1, 1, -1, 1), std.reshape(1, 1, -1, 1)


def concatenate_subjects(subject_data):
    return {
        'data': np.concatenate([item.data for item in subject_data], axis=0),
        'label': np.concatenate([item.label for item in subject_data], axis=0),
        'subject_id': np.concatenate([item.subject_id for item in subject_data], axis=0),
        'sample_id': np.concatenate([item.sample_id for item in subject_data], axis=0)
    }
