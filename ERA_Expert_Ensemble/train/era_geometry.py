import argparse
import json
import os

import numpy as np

from config.config import set_config
from train.expert_data import parse_subject_list
from utils.utils import ensure_path


def class_probability_centers(probability, labels, num_classes=5):
    probability = np.asarray(probability, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if probability.ndim != 2 or probability.shape[1] != num_classes:
        raise ValueError('Expected probability shaped [sample, {}].'.format(num_classes))
    centers = []
    counts = []
    for class_index in range(num_classes):
        mask = labels == class_index
        if not np.any(mask):
            raise ValueError('Geometry requires samples for class {}.'.format(class_index))
        centers.append(np.mean(probability[mask], axis=0))
        counts.append(int(np.sum(mask)))
    return np.asarray(centers, dtype=np.float32), np.asarray(counts, dtype=np.int64)


def pairwise_distance_matrix(centers):
    centers = np.asarray(centers, dtype=np.float32)
    difference = centers[:, None, :] - centers[None, :, :]
    return np.sqrt(np.sum(difference ** 2, axis=-1)).astype(np.float32)


def pairwise_js_matrix(centers, epsilon=1e-8):
    centers = np.clip(np.asarray(centers, dtype=np.float32), epsilon, 1.0)
    centers = centers / np.sum(centers, axis=1, keepdims=True)
    left = centers[:, None, :]
    right = centers[None, :, :]
    mixture = 0.5 * (left + right)
    left_kl = np.sum(left * (np.log(left) - np.log(mixture)), axis=-1)
    right_kl = np.sum(right * (np.log(right) - np.log(mixture)), axis=-1)
    return (0.5 * (left_kl + right_kl)).astype(np.float32)


def era_geometry_from_probability(probability, labels, num_classes=5):
    centers, counts = class_probability_centers(probability, labels, num_classes)
    distance = pairwise_distance_matrix(centers)
    rest_index = num_classes - 1
    rest_scale = max(float(np.mean(distance[rest_index, :rest_index])), 1e-8)
    return {
        'class_centers': centers,
        'class_counts': counts,
        'distance_matrix': distance,
        'js_matrix': pairwise_js_matrix(centers),
        'rest_anchored_centers': (centers - centers[rest_index]).astype(np.float32),
        'rest_distance_scale': np.asarray(rest_scale, dtype=np.float32),
        'rest_normalized_distance_matrix': (distance / rest_scale).astype(np.float32)
    }


def save_subject_geometry(args, subject):
    stem = os.path.join(args.oof_dir, 'subject{}_oof'.format(subject))
    probability_path = stem + '_probability.npy'
    metadata_path = stem + '_metadata.npz'
    if not os.path.exists(probability_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError('Missing OOF probability or metadata for subject {}.'.format(subject))
    probability = np.load(probability_path).astype(np.float32)
    metadata = np.load(metadata_path)
    geometry = era_geometry_from_probability(probability, metadata['label'], args.num_class)
    geometry['subject'] = np.asarray(subject, dtype=np.int64)

    ensure_path(args.geometry_dir)
    path = os.path.join(args.geometry_dir, 'subject_{}_geometry.npy'.format(subject))
    np.save(path, geometry, allow_pickle=True)
    return path, geometry


def generate_subject_geometries(args):
    manifest = []
    for subject in parse_subject_list(args):
        path, geometry = save_subject_geometry(args, subject)
        manifest.append({
            'subject': subject,
            'path': path,
            'class_counts': geometry['class_counts'].tolist()
        })
    ensure_path(args.geometry_dir)
    manifest_path = os.path.join(args.geometry_dir, 'subject_geometry_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description='Extract subject OOF ERA geometry')
    parser.add_argument('--oof-dir', default='./oof')
    parser.add_argument('--geometry-dir', default='./subject_geometry')
    parser.add_argument('--subject-list', default=None)
    parser.add_argument('--subjects', type=int, default=32)
    parser.add_argument('--num-class', type=int, default=5)
    cli = parser.parse_args()
    args, _ = set_config([])
    for key, value in vars(cli).items():
        setattr(args, key, value)
    generate_subject_geometries(args)


if __name__ == '__main__':
    main()
