import argparse
import glob
import json
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

from adaptive_fusion import (
    adaptive_probability_fusion,
    cosine_similarity,
    known_subject_probability_fusion,
    reliability_score,
    softmax_similarity,
)
from config.config import set_config
from train.expert_data import (
    apply_normalization,
    load_subject_hdf,
    normalization_from_checkpoint,
)
from train.era_geometry import pairwise_distance_matrix
from train.expert_runtime import evaluate_model, probability_from_logits, probability_metrics
from train.train_model import LABEL_MAPPING, compute_era_from_prob, load_model_checkpoint
from utils.utils import ensure_path, get_metrics, get_model


ERA_PLOT_LABELS = ['LEA', 'HEA', 'ACP', 'GDP', 'REST']
ERA_PLOT_COLORS = ['tab:blue', '#FFD700', 'tab:green', 'tab:red', 'tab:purple']


def build_parser():
    parser = argparse.ArgumentParser(description='Adaptive ERA expert inference')
    parser.add_argument('--model-dir', default='./checkpoints')
    parser.add_argument('--global-checkpoint', default='Global_final.pth',
                        help='Global checkpoint filename, including an ablation checkpoint.')
    parser.add_argument('--dataset', default='RESB', choices=['RESB', 'EMO', 'DEAP', 'MEEG', 'EEGMAT'])
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--subjects', default=None, help='Comma-separated zero-based subject ids.')
    parser.add_argument('--results-dir', default='./ERA_results')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--global-min-weight', type=float, default=0.35)
    parser.add_argument('--similarity-temperature', type=float, default=0.15)
    parser.add_argument('--similarity-threshold', type=float, default=0.05)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--unknown-subject', action='store_true', default=False,
                        help='Use Global + top-k similar Experts instead of the matched Expert.')
    return parser


def _checkpoint_args(checkpoint, overrides):
    config = dict(checkpoint.get('config', {}))
    config.update(overrides)
    if isinstance(config.get('input_shape'), list):
        config['input_shape'] = tuple(config['input_shape'])
    return SimpleNamespace(**config)


def _load_expert(path, overrides):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    args = _checkpoint_args(checkpoint, overrides)
    model = get_model(args)
    if torch.cuda.is_available():
        model = model.cuda()
    load_model_checkpoint(model, path)
    mean, std = normalization_from_checkpoint(checkpoint)
    return model, checkpoint, mean, std, args


def _run_expert(path, raw_data, labels, overrides, batch_size, return_features=False):
    model, checkpoint, mean, std, model_args = _load_expert(path, overrides)
    normalized = apply_normalization(raw_data, mean, std)
    result = evaluate_model(
        model, normalized, labels, batch_size,
        return_features=return_features,
        num_workers=getattr(model_args, 'num_workers', 0))
    probability = probability_from_logits(
        result['logits'], checkpoint.get('temperature', 1.0))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probability, result, checkpoint


def _era_numpy(probability):
    tensor = torch.as_tensor(probability, dtype=torch.float32)
    _, e_score, r_score, _, _ = compute_era_from_prob(tensor)
    return e_score.numpy(), r_score.numpy()


def _centers(labels, e_score, r_score):
    centers = {}
    labels = np.asarray(labels)
    for class_index in range(5):
        mask = labels == class_index
        name = LABEL_MAPPING['index_to_name'][class_index]
        centers[name] = {
            'class_index': class_index,
            'count': int(np.sum(mask)),
            'mean_E': float(np.mean(e_score[mask])) if np.any(mask) else None,
            'mean_R': float(np.mean(r_score[mask])) if np.any(mask) else None
        }
    return centers


def _era_geometry_summary(labels, e_score, r_score):
    labels = np.asarray(labels, dtype=np.int64)
    coordinates = np.stack([e_score, r_score], axis=1).astype(np.float32)
    centers = []
    within_variance = {}
    for class_index in range(5):
        mask = labels == class_index
        name = LABEL_MAPPING['index_to_name'][class_index]
        if not np.any(mask):
            centers.append(np.full(2, np.nan, dtype=np.float32))
            within_variance[name] = None
            continue
        center = np.mean(coordinates[mask], axis=0)
        centers.append(center)
        within_variance[name] = float(np.mean(np.sum((coordinates[mask] - center) ** 2, axis=1)))
    centers = np.asarray(centers, dtype=np.float32)
    distance = pairwise_distance_matrix(centers)
    return {
        'class_distance_matrix': distance.tolist(),
        'within_class_variance': within_variance
    }


def _plot_era(e_score, r_score, labels, output_path):
    plt.figure(figsize=(7, 6))
    for class_index in range(5):
        mask = np.asarray(labels) == class_index
        if np.any(mask):
            plt.scatter(e_score[mask], r_score[mask], s=12, alpha=0.45,
                        color=ERA_PLOT_COLORS[class_index],
                        label=ERA_PLOT_LABELS[class_index])
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.xlabel('Emotional Axis (E)')
    plt.ylabel('Rational Axis (R)')
    plt.title('Adaptive Expert ERA Space')
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()


def run_subject(args, base_args, subject, individual_paths, global_path):
    subject_data = load_subject_hdf(base_args, subject)
    overrides = {
        'dataset': base_args.dataset,
        'data_dir': base_args.data_dir,
        'checkpoint_dir': args.model_dir
    }
    global_probability, global_result, global_checkpoint = _run_expert(
        global_path, subject_data.data, subject_data.label, overrides,
        args.batch_size, return_features=True)

    prototype_ids = np.asarray(global_checkpoint['prototype_subject_ids'], dtype=np.int64)
    prototypes = np.asarray(global_checkpoint['subject_prototypes'], dtype=np.float32)
    raw_similarity = cosine_similarity(global_result['feature'], prototypes)
    if not args.unknown_subject and subject in individual_paths:
        probability, _, checkpoint = _run_expert(
            individual_paths[subject], subject_data.data, subject_data.label,
            overrides, args.batch_size)
        matched_reliability = reliability_score(
            checkpoint.get('validation_reliability'))
        final_probability, global_weight, matched_weight = known_subject_probability_fusion(
            global_probability, probability, matched_reliability)
        selected_expert_ids = np.asarray([subject], dtype=np.int64)
        individual_weight = matched_weight[:, None]
        fusion_mode = 'known_subject_matched'
    else:
        available = np.asarray([
            index for index, expert_subject in enumerate(prototype_ids.tolist())
            if expert_subject in individual_paths
        ], dtype=np.int64)
        if len(available) == 0:
            raise FileNotFoundError('No Individual checkpoint matches the Global subject prototypes.')
        top_k = min(max(1, args.top_k), len(available))
        subject_similarity = np.mean(raw_similarity[:, available], axis=0)
        selected_local = np.argpartition(subject_similarity, -top_k)[-top_k:]
        selected_indices = available[selected_local]
        selected_expert_ids = prototype_ids[selected_indices]
        selected_raw_similarity = raw_similarity[:, selected_indices]
        similarity = softmax_similarity(
            selected_raw_similarity, args.similarity_temperature)
        ordered_probability = []
        ordered_reliability = []
        for expert_subject in selected_expert_ids.tolist():
            probability, _, checkpoint = _run_expert(
                individual_paths[expert_subject], subject_data.data, subject_data.label,
                overrides, args.batch_size)
            ordered_probability.append(probability)
            ordered_reliability.append(reliability_score(
                checkpoint.get('validation_reliability')))
        individual_probability = np.stack(ordered_probability, axis=1)
        final_probability, global_weight, individual_weight = adaptive_probability_fusion(
            global_probability,
            individual_probability,
            reliability_score(global_checkpoint.get('validation_reliability')),
            ordered_reliability,
            similarity,
            global_min_weight=args.global_min_weight,
            similarity_threshold=args.similarity_threshold,
            similarity_max=np.max(selected_raw_similarity, axis=1))
        fusion_mode = 'unknown_subject_top_k'
    prediction = np.argmax(final_probability, axis=1)
    e_score, r_score = _era_numpy(final_probability)
    metrics = probability_metrics(subject_data.label, final_probability)

    result = {
        'subject': subject,
        'samples': len(subject_data.label),
        'fusion_mode': fusion_mode,
        'ACC': metrics['accuracy'],
        'Macro-F1': metrics['macro_f1'],
        'Kappa': metrics['kappa'],
        'NLL': metrics['nll'],
        'ECE': metrics['ece'],
        'Brier': metrics['brier'],
        'mean_E': float(np.mean(e_score)),
        'mean_R': float(np.mean(r_score)),
        'ERA_center': _centers(subject_data.label, e_score, r_score),
        **_era_geometry_summary(subject_data.label, e_score, r_score)
    }
    ensure_path(args.results_dir)
    with open(os.path.join(args.results_dir, 'subject_{}_result.json'.format(subject)),
              'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    np.save(os.path.join(args.results_dir, 'subject_{}_sample_result.npy'.format(subject)), {
        'sample_id': subject_data.sample_id,
        'label': subject_data.label,
        'prediction': prediction,
        'probability': final_probability,
        'E_score': e_score,
        'R_score': r_score,
        'global_weight': global_weight,
        'individual_weight': individual_weight,
        'prototype_subject_ids': selected_expert_ids
    }, allow_pickle=True)
    return result, {
        'subject_id': np.full(len(subject_data.label), subject, dtype=np.int64),
        'sample_id': subject_data.sample_id,
        'label': subject_data.label,
        'prediction': prediction,
        'probability': final_probability,
        'E_score': e_score,
        'R_score': r_score
    }


def main():
    args = build_parser().parse_args()
    base_args, _ = set_config([])
    base_args.dataset = 'EMO' if args.dataset == 'RESB' else args.dataset
    if args.data_dir:
        base_args.data_dir = args.data_dir
    if args.subjects:
        subjects = [int(value.strip()) for value in args.subjects.split(',') if value.strip()]
    else:
        subjects = list(range(base_args.subjects))

    global_path = os.path.join(args.model_dir, args.global_checkpoint)
    if not os.path.exists(global_path):
        raise FileNotFoundError('Global checkpoint not found: {}'.format(global_path))
    individual_paths = {}
    for path in glob.glob(os.path.join(args.model_dir, 'Expert_*_final.pth')):
        name = os.path.basename(path)
        expert_subject = int(name[len('Expert_'):-len('_final.pth')])
        individual_paths[expert_subject] = path
    if not individual_paths:
        raise FileNotFoundError('No Expert_*_final.pth checkpoints found in {}'.format(args.model_dir))

    summaries = []
    sample_results = []
    for subject in subjects:
        summary, sample_result = run_subject(
            args, base_args, subject, individual_paths, global_path)
        summaries.append(summary)
        sample_results.append(sample_result)

    combined = {
        key: np.concatenate([result[key] for result in sample_results], axis=0)
        for key in sample_results[0]
    }

    metrics = probability_metrics(combined['label'], combined['probability'])
    overall = {
        'subjects': summaries,
        'ACC': metrics['accuracy'],
        'Macro-F1': metrics['macro_f1'],
        'Kappa': metrics['kappa'],
        'NLL': metrics['nll'],
        'ECE': metrics['ece'],
        'Brier': metrics['brier'],
        'ERA_center': _centers(combined['label'], combined['E_score'], combined['R_score']),
        **_era_geometry_summary(combined['label'], combined['E_score'], combined['R_score'])
    }
    ensure_path(args.results_dir)
    with open(os.path.join(args.results_dir, 'subject_result.json'), 'w', encoding='utf-8') as file:
        json.dump(overall, file, ensure_ascii=False, indent=2)
    with open(os.path.join(args.results_dir, 'ERA_center.json'), 'w', encoding='utf-8') as file:
        json.dump(overall['ERA_center'], file, ensure_ascii=False, indent=2)
    np.save(os.path.join(args.results_dir, 'sample_result.npy'), combined, allow_pickle=True)
    _plot_era(combined['E_score'], combined['R_score'], combined['label'],
              os.path.join(args.results_dir, 'ERA_scatter.png'))
    print(json.dumps({
        key: overall[key]
        for key in ['ACC', 'Macro-F1', 'Kappa', 'NLL', 'ECE', 'Brier']
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
