import json
import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

from train.expert_data import (
    apply_normalization,
    compute_normalization,
    load_subject_hdf,
    normalization_to_checkpoint,
    parse_subject_list,
)
from train.expert_runtime import (
    classification_metrics,
    evaluate_model,
    fit_temperature,
    probability_from_logits,
    train_classifier,
)
from train.era_geometry import era_geometry_from_probability
from train.train_model import build_checkpoint
from utils.utils import ensure_path


def _safe_oof_folds(labels, groups, requested):
    unique_groups = np.unique(groups)
    per_class_groups = []
    for cls in np.unique(labels):
        per_class_groups.append(len(np.unique(groups[labels == cls])))
    available = min(len(unique_groups), min(per_class_groups))
    if available < 2:
        raise ValueError('OOF training needs at least two independent trial groups per class.')
    return min(int(requested), available)


def _subject_prototype(model, data, label, args):
    result = evaluate_model(
        model, data, label, args.batch_size,
        return_features=True, num_workers=args.num_workers)
    features = result['feature']
    return torch.as_tensor(np.mean(features, axis=0), dtype=torch.float32)


def train_individual_subject(args, subject):
    subject_data = load_subject_hdf(args, subject)
    folds = _safe_oof_folds(subject_data.label, subject_data.trial_id, args.oof_folds)
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=args.random_seed)

    oof_logits = np.zeros((len(subject_data.label), args.num_class), dtype=np.float32)
    oof_seen = np.zeros(len(subject_data.label), dtype=bool)
    best_epochs = []
    fold_metrics = []
    fold_dir = os.path.join(args.checkpoint_dir, 'oof_folds')
    ensure_path(fold_dir)

    for fold, (train_idx, val_idx) in enumerate(splitter.split(
            subject_data.data, subject_data.label, groups=subject_data.trial_id)):
        mean, std = compute_normalization(subject_data.data[train_idx])
        data_train = apply_normalization(subject_data.data[train_idx], mean, std)
        data_val = apply_normalization(subject_data.data[val_idx], mean, std)
        model, best_epoch = train_classifier(
            args, data_train, subject_data.label[train_idx],
            data_val, subject_data.label[val_idx])
        prediction = evaluate_model(
            model, data_val, subject_data.label[val_idx], args.batch_size,
            num_workers=args.num_workers)
        oof_logits[val_idx] = prediction['logits']
        oof_seen[val_idx] = True
        best_epochs.append(best_epoch)
        fold_metrics.append(classification_metrics(prediction['label'], prediction['logits']))

        fold_checkpoint = build_checkpoint(
            model, args, subject=subject, fold=fold,
            extra_metadata={
                **normalization_to_checkpoint(mean, std),
                'channel_order': subject_data.channel_order,
                'graph_structure': subject_data.graph_structure,
                'checkpoint_role': 'individual_oof_fold'
            })
        torch.save(fold_checkpoint, os.path.join(
            fold_dir, 'Expert_{}_fold{}.pth'.format(subject, fold)))

    if not np.all(oof_seen):
        missing = np.where(~oof_seen)[0].tolist()
        raise RuntimeError('OOF predictions are missing samples: {}'.format(missing[:10]))

    temperature = fit_temperature(oof_logits, subject_data.label)
    oof_probability = probability_from_logits(oof_logits, temperature)
    reliability = classification_metrics(subject_data.label, oof_logits / temperature)

    ensure_path(args.oof_dir)
    stem = os.path.join(args.oof_dir, 'subject{}_oof'.format(subject))
    np.save(stem + '_logits.npy', oof_logits)
    np.save(stem + '_probability.npy', oof_probability)
    np.savez_compressed(
        stem + '_metadata.npz',
        sample_id=subject_data.sample_id,
        trial_id=subject_data.trial_id,
        segment_id=subject_data.segment_id,
        subject_id=subject_data.subject_id,
        label=subject_data.label
    )
    geometry = era_geometry_from_probability(
        oof_probability, subject_data.label, args.num_class)
    geometry['subject'] = np.asarray(subject, dtype=np.int64)
    geometry_dir = getattr(args, 'geometry_dir', './subject_geometry')
    ensure_path(geometry_dir)
    geometry_path = os.path.join(
        geometry_dir, 'subject_{}_geometry.npy'.format(subject))
    np.save(geometry_path, geometry, allow_pickle=True)

    final_mean, final_std = compute_normalization(subject_data.data)
    final_data = apply_normalization(subject_data.data, final_mean, final_std)
    final_epochs = max(1, int(round(float(np.median(best_epochs)))))
    final_model, _ = train_classifier(
        args, final_data, subject_data.label, fixed_epochs=final_epochs)
    prototype = _subject_prototype(final_model, final_data, subject_data.label, args)

    ensure_path(args.checkpoint_dir)
    checkpoint = build_checkpoint(
        final_model, args, subject=subject,
        extra_metadata={
            **normalization_to_checkpoint(final_mean, final_std),
            'channel_order': subject_data.channel_order,
            'graph_structure': subject_data.graph_structure,
            # OOF temperature belongs to the fold ensemble, not this separately retrained model.
            'temperature': 1.0,
            'oof_temperature': float(temperature),
            'validation_reliability': reliability,
            'subject_prototype': prototype,
            'training_epochs': final_epochs,
            'checkpoint_role': 'individual_final'
        })
    checkpoint_path = os.path.join(
        args.checkpoint_dir, 'Expert_{}_final.pth'.format(subject))
    torch.save(checkpoint, checkpoint_path)

    summary = {
        'subject': subject,
        'checkpoint': checkpoint_path,
        'oof_probability': stem + '_probability.npy',
        'geometry': geometry_path,
        'temperature': 1.0,
        'oof_temperature': temperature,
        'validation_reliability': reliability,
        'fold_metrics': fold_metrics,
        'training_epochs': final_epochs
    }
    with open(stem + '_summary.json', 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def train_individual_experts(args):
    summaries = []
    for subject in parse_subject_list(args):
        print('Training Individual Expert {}...'.format(subject))
        summaries.append(train_individual_subject(args, subject))
    manifest_path = os.path.join(args.checkpoint_dir, 'individual_manifest.json')
    ensure_path(args.checkpoint_dir)
    with open(manifest_path, 'w', encoding='utf-8') as file:
        json.dump(summaries, file, ensure_ascii=False, indent=2)
    print('Individual expert manifest saved:', manifest_path)
    return summaries
