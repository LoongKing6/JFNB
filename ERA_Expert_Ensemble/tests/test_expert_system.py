import os
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import torch.nn as nn

from adaptive_fusion import (
    adaptive_probability_fusion,
    known_subject_probability_fusion,
    softmax_similarity,
)
from train.balanced_sampler import SubjectClassBalancedSampler
from train.expert_data import (
    apply_normalization,
    compute_normalization,
    load_subject_hdf,
)
from train.distillation import train_global_expert
from train.era_geometry import era_geometry_from_probability
from train.expert_training import train_individual_experts
import train.expert_runtime as expert_runtime


def build_args(data_dir):
    return SimpleNamespace(
        data_dir=str(data_dir), data_format='eeg', dataset='EMO',
        label_type='Axis', graph_type='hem')


def test_legacy_hdf_metadata_and_normalization(tmp_path):
    raw = np.arange(4 * 2 * 1 * 3 * 5, dtype=np.float32).reshape(4, 2, 1, 3, 5)
    label = np.tile(np.arange(4, dtype=np.int64)[:, None], (1, 2))
    with h5py.File(tmp_path / 'sub0.hdf', 'w') as dataset:
        dataset['data'] = raw
        dataset['label'] = label

    subject = load_subject_hdf(build_args(tmp_path), 0)
    assert subject.data.shape == (8, 1, 3, 5)
    assert np.array_equal(subject.trial_id, np.repeat(np.arange(4), 2))
    assert np.array_equal(subject.segment_id, np.tile(np.arange(2), 4))

    mean, std = compute_normalization(subject.data)
    normalized = apply_normalization(subject.data, mean, std)
    assert np.allclose(np.mean(normalized, axis=(0, 1, 3)), 0.0, atol=1e-6)
    assert np.allclose(np.std(normalized, axis=(0, 1, 3)), 1.0, atol=1e-6)


def test_subject_class_balanced_sampler_visits_all_groups():
    subjects = np.repeat([0, 1], 6)
    labels = np.tile(np.repeat([0, 1, 2], 2), 2)
    sampler = SubjectClassBalancedSampler(subjects, labels, num_samples=12, seed=7)
    indices = list(iter(sampler))
    groups = {(int(subjects[index]), int(labels[index])) for index in indices}
    assert groups == {(subject, label) for subject in [0, 1] for label in [0, 1, 2]}


def test_adaptive_fusion_is_normalized_and_can_fallback():
    global_probability = np.asarray([[0.7, 0.1, 0.1, 0.05, 0.05]], dtype=np.float32)
    individual_probability = np.asarray([[
        [0.1, 0.7, 0.1, 0.05, 0.05],
        [0.1, 0.1, 0.7, 0.05, 0.05]
    ]], dtype=np.float32)
    similarity = softmax_similarity(np.asarray([[0.8, 0.2]], dtype=np.float32), 0.15)
    final, global_weight, individual_weight = adaptive_probability_fusion(
        global_probability, individual_probability,
        0.8, [0.7, 0.6], similarity,
        global_min_weight=0.35, similarity_threshold=0.05,
        similarity_max=np.asarray([0.8]))
    assert np.allclose(np.sum(final, axis=1), 1.0)
    assert global_weight[0] >= 0.35
    assert np.all(individual_weight >= 0)

    fallback, fallback_global, fallback_individual = adaptive_probability_fusion(
        global_probability, individual_probability,
        0.8, [0.7, 0.6], similarity,
        similarity_threshold=0.05,
        similarity_max=np.asarray([-0.2]))
    assert np.allclose(fallback, global_probability)
    assert fallback_global[0] == 1.0
    assert np.allclose(fallback_individual, 0.0)


def test_known_subject_fusion_tracks_reliability():
    global_probability = np.asarray([[0.7, 0.1, 0.1, 0.05, 0.05]], dtype=np.float32)
    individual_probability = np.asarray([[0.1, 0.7, 0.1, 0.05, 0.05]], dtype=np.float32)
    final, global_weight, individual_weight = known_subject_probability_fusion(
        global_probability, individual_probability, 0.8)
    assert np.allclose(np.sum(final, axis=1), 1.0)
    assert np.allclose(global_weight + individual_weight, 1.0)
    assert individual_weight[0] > global_weight[0]


def test_era_geometry_is_symmetric_and_rest_normalized():
    labels = np.repeat(np.arange(5), 2)
    probability = np.eye(5, dtype=np.float32)[labels] * 0.8 + 0.04
    geometry = era_geometry_from_probability(probability, labels)
    assert geometry['class_centers'].shape == (5, 5)
    assert np.allclose(geometry['distance_matrix'], geometry['distance_matrix'].T)
    assert np.allclose(np.diag(geometry['js_matrix']), 0.0)
    assert np.isclose(np.mean(geometry['rest_normalized_distance_matrix'][4, :4]), 1.0)


def test_top_k_fusion_zeroes_unselected_experts():
    global_probability = np.asarray([[0.6, 0.1, 0.1, 0.1, 0.1]], dtype=np.float32)
    individual_probability = np.repeat(
        np.asarray([[[0.1, 0.6, 0.1, 0.1, 0.1]]], dtype=np.float32), 4, axis=1)
    similarity = softmax_similarity(
        np.asarray([[0.9, 0.8, 0.2, 0.1]], dtype=np.float32), 0.15)
    _, _, individual_weight = adaptive_probability_fusion(
        global_probability, individual_probability, 0.8,
        [0.8, 0.8, 0.8, 0.8], similarity, top_k=2)
    assert np.count_nonzero(individual_weight[0]) == 2


class TinyClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(nn.Flatten(), nn.Linear(12, 10), nn.ReLU())
        self.classifier = nn.Linear(10, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


def _expert_args(tmp_path):
    return SimpleNamespace(
        data_dir=str(tmp_path / 'data'), data_format='eeg', dataset='EMO',
        label_type='Axis', graph_type='hem', subject_list='0,1,2', subjects=3,
        checkpoint_dir=str(tmp_path / 'checkpoints'), oof_dir=str(tmp_path / 'oof'),
        geometry_dir=str(tmp_path / 'subject_geometry'),
        shared_geometry_path='shared_era_geometry.npy',
        oof_folds=3, num_class=5, random_seed=13, batch_size=15, num_workers=0,
        learning_rate=0.02, weight_decay=0.0, LS=False, LS_rate=0.0,
        expert_max_epoch=3, expert_patience=2, distill_lambda=0.5,
        distill_temperature=2.0, geometry_lambda=0.5,
        global_objective='full', run_global_ablations=False,
        teacher_gate_floor=0.05, global_validation_subjects=1,
        input_shape=(1, 3, 4), model='AT-DGNN', target_rate=500,
        T=8, hidden=4, dropout=0.1, pool=2, pool_step_rate=0.5,
        temporal_kernel=3, gpu='0'
    )


def _write_synthetic_subject(path, subject):
    rng = np.random.default_rng(100 + subject)
    labels = np.repeat(np.arange(5, dtype=np.int64), 3)
    data = rng.normal(0, 0.15, size=(15, 1, 1, 3, 4)).astype(np.float32)
    for index, label in enumerate(labels):
        data[index, 0, 0, label % 3, :] += float(label + 1)
    with h5py.File(path, 'w') as dataset:
        dataset['data'] = data
        dataset['label'] = labels.reshape(-1, 1)


def test_individual_and_global_training_pipeline(tmp_path, monkeypatch):
    args = _expert_args(tmp_path)
    os.makedirs(args.data_dir)
    for subject in range(3):
        _write_synthetic_subject(os.path.join(args.data_dir, 'sub{}.hdf'.format(subject)), subject)

    monkeypatch.setattr(expert_runtime, 'get_model', lambda model_args: TinyClassifier(model_args.num_class))
    train_individual_experts(args)
    train_global_expert(args)

    for subject in range(3):
        assert os.path.exists(os.path.join(args.oof_dir, 'subject{}_oof_logits.npy'.format(subject)))
        checkpoint = torch.load(
            os.path.join(args.checkpoint_dir, 'Expert_{}_final.pth'.format(subject)),
            map_location='cpu', weights_only=False)
        for key in [
            'state_dict', 'config', 'label_mapping', 'era_mapping', 'channel_order',
            'graph_structure', 'normalization_mean', 'normalization_std',
            'temperature', 'validation_reliability', 'subject_prototype'
        ]:
            assert key in checkpoint

    global_checkpoint = torch.load(
        os.path.join(args.checkpoint_dir, 'Global_final.pth'),
        map_location='cpu', weights_only=False)
    assert global_checkpoint['subject_prototypes'].shape[0] == 3
    assert global_checkpoint['prototype_subject_ids'].tolist() == [0, 1, 2]
    assert global_checkpoint['global_objective'] == 'full'
    assert 'model_state_dict' in global_checkpoint
    assert 'shared_geometry' in global_checkpoint
    assert 'subject_reliability' in global_checkpoint


def test_global_ablation_objectives_run(tmp_path, monkeypatch):
    args = _expert_args(tmp_path)
    os.makedirs(args.data_dir)
    for subject in range(3):
        _write_synthetic_subject(os.path.join(args.data_dir, 'sub{}.hdf'.format(subject)), subject)
    monkeypatch.setattr(expert_runtime, 'get_model', lambda model_args: TinyClassifier(model_args.num_class))
    train_individual_experts(args)
    for objective in ('ce', 'kd', 'reliability_kd'):
        args.global_objective = objective
        summary = train_global_expert(args)
        assert summary['global_objective'] == objective
