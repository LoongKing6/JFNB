import argparse
import json
import os

import numpy as np

from config.config import set_config
from train.expert_data import parse_subject_list
from train.era_geometry import pairwise_distance_matrix, pairwise_js_matrix, save_subject_geometry
from utils.utils import ensure_path


def reliability_weight(metrics):
    metrics = metrics or {}
    macro_f1 = max(float(metrics.get('macro_f1', 0.0)), 1e-4)
    nll = max(float(metrics.get('nll', 0.0)), 0.0)
    ece = min(max(float(metrics.get('ece', 0.0)), 0.0), 1.0)
    return float(macro_f1 * np.exp(-nll) * (1.0 - ece))


def _load_reliability(oof_dir, subject):
    path = os.path.join(oof_dir, 'subject{}_oof_summary.json'.format(subject))
    if not os.path.exists(path):
        raise FileNotFoundError('Missing OOF summary for subject {}: {}'.format(subject, path))
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)['validation_reliability']


def build_shared_geometry(args):
    subjects = parse_subject_list(args)
    centers = []
    distance = []
    js = []
    rest_distance = []
    rest_centers = []
    raw_weights = []
    reliabilities = []
    for subject in subjects:
        geometry_path = os.path.join(args.geometry_dir, 'subject_{}_geometry.npy'.format(subject))
        if not os.path.exists(geometry_path):
            save_subject_geometry(args, subject)
        geometry = np.load(geometry_path, allow_pickle=True).item()
        reliability = _load_reliability(args.oof_dir, subject)
        centers.append(geometry['class_centers'])
        distance.append(geometry['distance_matrix'])
        js.append(geometry['js_matrix'])
        rest_distance.append(geometry['rest_normalized_distance_matrix'])
        rest_centers.append(geometry['rest_anchored_centers'])
        raw_weights.append(reliability_weight(reliability))
        reliabilities.append(reliability)

    raw_weights = np.asarray(raw_weights, dtype=np.float32)
    if not np.any(raw_weights > 0):
        raw_weights = np.ones_like(raw_weights)
    weights = raw_weights / np.sum(raw_weights)
    shared_centers = np.average(np.stack(centers), axis=0, weights=weights).astype(np.float32)
    shared = {
        'subject_ids': np.asarray(subjects, dtype=np.int64),
        'subject_weights': weights.astype(np.float32),
        'subject_reliability': np.asarray(reliabilities, dtype=object),
        'shared_class_centers': shared_centers,
        'shared_distance_matrix': np.average(
            np.stack(distance), axis=0, weights=weights).astype(np.float32),
        'shared_js_matrix': np.average(
            np.stack(js), axis=0, weights=weights).astype(np.float32),
        'shared_rest_anchored_centers': np.average(
            np.stack(rest_centers), axis=0, weights=weights).astype(np.float32),
        'shared_rest_normalized_distance_matrix': np.average(
            np.stack(rest_distance), axis=0, weights=weights).astype(np.float32),
        'distance_from_shared_centers': pairwise_distance_matrix(shared_centers),
        'js_from_shared_centers': pairwise_js_matrix(shared_centers)
    }
    ensure_path(args.geometry_dir)
    output_path = args.shared_geometry_path
    if not os.path.isabs(output_path):
        output_path = os.path.join(args.geometry_dir, output_path)
    np.save(output_path, shared, allow_pickle=True)
    return output_path, shared


def main():
    parser = argparse.ArgumentParser(description='Build shared ERA geometry template')
    parser.add_argument('--oof-dir', default='./oof')
    parser.add_argument('--geometry-dir', default='./subject_geometry')
    parser.add_argument('--shared-geometry-path', default='shared_era_geometry.npy')
    parser.add_argument('--subject-list', default=None)
    parser.add_argument('--subjects', type=int, default=32)
    parser.add_argument('--num-class', type=int, default=5)
    cli = parser.parse_args()
    args, _ = set_config([])
    for key, value in vars(cli).items():
        setattr(args, key, value)
    path, _ = build_shared_geometry(args)
    print('Shared ERA geometry saved:', path)


if __name__ == '__main__':
    main()
