import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib.pyplot as plt

from config.config import set_config
from train.era_geometry import pairwise_distance_matrix
from train.expert_data import apply_normalization, load_subject_hdf, normalization_from_checkpoint
from train.expert_runtime import evaluate_model, probability_from_logits, probability_metrics
from train.train_model import compute_era_from_prob, load_model_checkpoint
from utils.utils import ensure_path, get_model


ERA_PLOT_LABELS = ['LEA', 'HEA', 'ACP', 'GDP', 'REST']
ERA_PLOT_COLORS = ['tab:blue', '#FFD700', 'tab:green', 'tab:red', 'tab:purple']


def _checkpoint_args(checkpoint, overrides):
    config = dict(checkpoint.get('config', {}))
    config.update(overrides)
    if isinstance(config.get('input_shape'), list):
        config['input_shape'] = tuple(config['input_shape'])
    return SimpleNamespace(**config)


def _plot_era(coordinates, labels, output_path, title):
    plt.figure(figsize=(7, 6))
    for class_index in range(5):
        mask = labels == class_index
        plt.scatter(
            coordinates[mask, 0], coordinates[mask, 1],
            s=12, alpha=0.45,
            color=ERA_PLOT_COLORS[class_index],
            label=ERA_PLOT_LABELS[class_index])
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.xlabel('Emotional Axis (E)')
    plt.ylabel('Rational Axis (R)')
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()


def _evaluate_checkpoint(path, base_args, subjects, batch_size, scatter_path=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    args = _checkpoint_args(checkpoint, {
        'dataset': base_args.dataset,
        'data_dir': base_args.data_dir
    })
    model = get_model(args)
    if torch.cuda.is_available():
        model = model.cuda()
    load_model_checkpoint(model, path)
    mean, std = normalization_from_checkpoint(checkpoint)
    labels = []
    probabilities = []
    for subject in subjects:
        subject_data = load_subject_hdf(base_args, subject)
        normalized = apply_normalization(subject_data.data, mean, std)
        result = evaluate_model(
            model, normalized, subject_data.label, batch_size,
            num_workers=getattr(args, 'num_workers', 0))
        probabilities.append(probability_from_logits(
            result['logits'], checkpoint.get('temperature', 1.0)))
        labels.append(subject_data.label)
    labels = np.concatenate(labels)
    probability = np.concatenate(probabilities)
    metrics = probability_metrics(labels, probability)
    _, e_score, r_score, _, _ = compute_era_from_prob(
        torch.as_tensor(probability, dtype=torch.float32))
    coordinates = np.stack([e_score.numpy(), r_score.numpy()], axis=1)
    if scatter_path is not None:
        _plot_era(
            coordinates, labels, scatter_path,
            '{} ERA Space'.format(checkpoint.get('global_objective', 'Global')))
    centers = np.stack([np.mean(coordinates[labels == index], axis=0) for index in range(5)])
    within_variance = [
        float(np.mean(np.sum((coordinates[labels == index] - centers[index]) ** 2, axis=1)))
        for index in range(5)
    ]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        'ACC': metrics['accuracy'],
        'Macro-F1': metrics['macro_f1'],
        'Kappa': metrics['kappa'],
        'NLL': metrics['nll'],
        'ECE': metrics['ece'],
        'Brier': metrics['brier'],
        'ERA_centers': centers.tolist(),
        'ERA_distance_matrix': pairwise_distance_matrix(centers).tolist(),
        'ERA_within_class_variance': within_variance
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Global ERA ablation checkpoints')
    parser.add_argument('--model-dir', default='./checkpoints')
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--subjects', default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--results-dir', default='./ERA_results/ablations')
    cli = parser.parse_args()
    base_args, _ = set_config([])
    if cli.data_dir:
        base_args.data_dir = cli.data_dir
    subjects = (
        [int(value.strip()) for value in cli.subjects.split(',') if value.strip()]
        if cli.subjects else list(range(base_args.subjects))
    )
    results = {}
    ensure_path(cli.results_dir)
    for objective in ('ce', 'kd', 'reliability_kd', 'full'):
        path = os.path.join(cli.model_dir, 'Global_{}_final.pth'.format(objective))
        if os.path.exists(path):
            results[objective] = _evaluate_checkpoint(
                path, base_args, subjects, cli.batch_size,
                scatter_path=os.path.join(
                    cli.results_dir, 'Global_{}_ERA_scatter.png'.format(objective)))
    if not results:
        raise FileNotFoundError('No Global ablation checkpoints found in {}'.format(cli.model_dir))
    output_path = os.path.join(cli.results_dir, 'global_ablation_results.json')
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
