import json
import os
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train.balanced_sampler import SubjectClassBalancedSampler
from train.expert_data import (
    ExpertDataset,
    apply_normalization,
    compute_normalization,
    load_subject_hdf,
    normalization_to_checkpoint,
    parse_subject_list,
)
from train.expert_runtime import (
    classification_metrics,
    create_model,
    evaluate_model,
    fit_temperature,
    get_device,
    probability_metrics,
)
from train.shared_geometry import build_shared_geometry, reliability_weight
from train.train_model import build_checkpoint
from utils.utils import ensure_path, seed_all


GLOBAL_OBJECTIVES = ('ce', 'kd', 'reliability_kd', 'full')


def _load_oof_summary(args, subject):
    path = os.path.join(args.oof_dir, 'subject{}_oof_summary.json'.format(subject))
    if not os.path.exists(path):
        raise FileNotFoundError('OOF summary missing for subject {}: {}'.format(subject, path))
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def _load_oof_teacher(args, subject, subject_data):
    stem = os.path.join(args.oof_dir, 'subject{}_oof'.format(subject))
    probability_path = stem + '_probability.npy'
    metadata_path = stem + '_metadata.npz'
    if not os.path.exists(probability_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(
            'OOF calibrated probability missing for subject {}. Run --mode individual first.'.format(subject))

    probability = np.load(probability_path).astype(np.float32)
    metadata = np.load(metadata_path)
    if not np.array_equal(metadata['sample_id'], subject_data.sample_id):
        raise ValueError('OOF sample_id mismatch for subject {}.'.format(subject))
    if not np.array_equal(metadata['label'], subject_data.label):
        raise ValueError('OOF label mismatch for subject {}.'.format(subject))
    if probability.shape != (len(subject_data.label), args.num_class):
        raise ValueError('OOF probability shape mismatch for subject {}.'.format(subject))

    reliability = _load_oof_summary(args, subject)['validation_reliability']
    return probability, reliability


def _collect_subjects(args, subjects):
    entries = []
    for subject in subjects:
        subject_data = load_subject_hdf(args, subject)
        probability, reliability = _load_oof_teacher(args, subject, subject_data)
        entries.append((subject, subject_data, probability, reliability))
    return entries


def _combine_entries(entries, mean, std):
    return {
        'data': np.concatenate([
            apply_normalization(subject_data.data, mean, std)
            for _, subject_data, _, _ in entries
        ], axis=0),
        'label': np.concatenate([subject_data.label for _, subject_data, _, _ in entries], axis=0),
        'teacher_probability': np.concatenate([probability for _, _, probability, _ in entries], axis=0),
        'subject_reliability': np.concatenate([
            np.full(len(subject_data.label), reliability_weight(reliability), dtype=np.float32)
            for _, subject_data, _, reliability in entries
        ], axis=0),
        'subject_id': np.concatenate([subject_data.subject_id for _, subject_data, _, _ in entries], axis=0),
        'sample_id': np.concatenate([subject_data.sample_id for _, subject_data, _, _ in entries], axis=0)
    }


def _soften_calibrated_probability(probability, temperature):
    probability = torch.clamp(probability, min=1e-8)
    temperature = max(float(temperature), 1e-6)
    return torch.softmax(torch.log(probability) / temperature, dim=1)


def _normalized_entropy(probability):
    probability = torch.clamp(probability, min=1e-8)
    entropy = -torch.sum(probability * torch.log(probability), dim=1)
    return entropy / float(np.log(probability.shape[1]))


def _teacher_gate(teacher_probability, labels, subject_reliability, floor):
    confidence = teacher_probability.gather(1, labels[:, None]).squeeze(1)
    certainty = 1.0 - _normalized_entropy(teacher_probability)
    gate = subject_reliability * confidence * certainty
    return torch.clamp(gate, min=float(floor), max=1.0)


def _pairwise_student_geometry(student_probability, labels, num_classes, target):
    centers = []
    valid = []
    for class_index in range(num_classes):
        mask = labels == class_index
        if torch.any(mask):
            centers.append(torch.mean(student_probability[mask], dim=0))
            valid.append(class_index)
    if len(valid) < 2:
        return None, None
    centers = torch.stack(centers)
    distance = torch.cdist(centers, centers, p=2)
    rest_index = num_classes - 1
    if rest_index in valid:
        local_rest = valid.index(rest_index)
        non_rest = [index for index in range(len(valid)) if index != local_rest]
        scale = torch.mean(distance[local_rest, non_rest]) if non_rest else distance.new_tensor(1.0)
        distance = distance / torch.clamp(scale, min=1e-8)
    target_indices = torch.as_tensor(valid, dtype=torch.long, device=distance.device)
    target = target.index_select(0, target_indices).index_select(1, target_indices)
    off_diagonal = ~torch.eye(len(valid), dtype=torch.bool, device=distance.device)
    return distance[off_diagonal], target[off_diagonal]


def _distillation_loss(student_logits, labels, teacher_probability,
                       subject_reliability, shared_distance, args):
    ce = F.cross_entropy(
        student_logits, labels,
        label_smoothing=args.LS_rate if args.LS else 0.0)
    objective = args.global_objective
    kd = student_logits.new_tensor(0.0)
    geometry = student_logits.new_tensor(0.0)

    if objective in ('kd', 'reliability_kd', 'full'):
        teacher_soft = _soften_calibrated_probability(
            teacher_probability, args.distill_temperature)
        student_log_probability = F.log_softmax(
            student_logits / args.distill_temperature, dim=1)
        per_sample_kd = F.kl_div(
            student_log_probability, teacher_soft,
            reduction='none').sum(dim=1)
        if objective in ('reliability_kd', 'full'):
            gate = _teacher_gate(
                teacher_probability, labels, subject_reliability,
                args.teacher_gate_floor)
            kd = torch.sum(gate * per_sample_kd) / torch.clamp(torch.sum(gate), min=1e-8)
        else:
            kd = torch.mean(per_sample_kd)

    if objective == 'full' and shared_distance is not None:
        student_probability = torch.softmax(student_logits, dim=1)
        student_relation, target_relation = _pairwise_student_geometry(
            student_probability, labels, args.num_class, shared_distance)
        if student_relation is not None:
            geometry = F.mse_loss(student_relation, target_relation)

    total = ce
    if objective in ('kd', 'reliability_kd', 'full'):
        total = total + args.distill_lambda * (args.distill_temperature ** 2) * kd
    if objective == 'full':
        total = total + args.geometry_lambda * geometry
    return total, {'ce': ce.detach(), 'kd': kd.detach(), 'geometry': geometry.detach()}


def _train_distilled(args, bundle, shared_distance=None, val_bundle=None, fixed_epochs=None):
    seed_all(args.random_seed)
    device = get_device()
    model = create_model(args)
    sampler = SubjectClassBalancedSampler(
        bundle['subject_id'], bundle['label'], seed=args.random_seed)
    dataset = ExpertDataset(
        bundle['data'], bundle['label'],
        teacher_logits=bundle['teacher_probability'],
        subject_ids=bundle['subject_id'],
        teacher_weights=bundle['subject_reliability'])
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    target_distance = None
    if shared_distance is not None:
        target_distance = torch.as_tensor(shared_distance, dtype=torch.float32, device=device)

    max_epochs = int(fixed_epochs or args.expert_max_epoch)
    best_state = None
    best_epoch = max_epochs
    best_score = -np.inf
    patience = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        totals = {'ce': 0.0, 'kd': 0.0, 'geometry': 0.0, 'batches': 0}
        for x_batch, y_batch, teacher_probability, _, subject_reliability in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            teacher_probability = teacher_probability.to(device)
            subject_reliability = subject_reliability.to(device).float()
            optimizer.zero_grad()
            logits = model(x_batch)
            loss, parts = _distillation_loss(
                logits, y_batch, teacher_probability, subject_reliability,
                target_distance, args)
            loss.backward()
            optimizer.step()
            for key in ('ce', 'kd', 'geometry'):
                totals[key] += float(parts[key].cpu())
            totals['batches'] += 1

        record = {
            'epoch': epoch,
            'train_ce': totals['ce'] / max(totals['batches'], 1),
            'train_kd': totals['kd'] / max(totals['batches'], 1),
            'train_geometry': totals['geometry'] / max(totals['batches'], 1)
        }
        if val_bundle is not None:
            validation = evaluate_model(
                model, val_bundle['data'], val_bundle['label'], args.batch_size,
                num_workers=args.num_workers)
            metrics = classification_metrics(validation['label'], validation['logits'])
            record['validation'] = metrics
            score = metrics['macro_f1']
            if score > best_score + 1e-8:
                best_score = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= args.expert_patience:
                    history.append(record)
                    break
        history.append(record)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, history


def _extract_subject_prototypes(model, entries, mean, std, args):
    prototypes = []
    subject_ids = []
    for subject, subject_data, _, _ in entries:
        data = apply_normalization(subject_data.data, mean, std)
        result = evaluate_model(
            model, data, subject_data.label, args.batch_size,
            return_features=True, num_workers=args.num_workers)
        prototype = np.mean(result['feature'], axis=0)
        norm = np.linalg.norm(prototype)
        if norm > 1e-8:
            prototype = prototype / norm
        prototypes.append(prototype.astype(np.float32))
        subject_ids.append(subject)
    return np.asarray(subject_ids, dtype=np.int64), np.stack(prototypes, axis=0)


def _resolve_shared_geometry(args):
    output_path = args.shared_geometry_path
    if not os.path.isabs(output_path):
        output_path = os.path.join(args.geometry_dir, output_path)
    if not os.path.exists(output_path):
        output_path, shared = build_shared_geometry(args)
    else:
        shared = np.load(output_path, allow_pickle=True).item()
    return output_path, shared


def _train_one_global_expert(args, checkpoint_name='Global_final.pth', summary_name='global_summary.json'):
    if args.global_objective not in GLOBAL_OBJECTIVES:
        raise ValueError('Unknown global objective: {}'.format(args.global_objective))
    subjects = parse_subject_list(args)
    if len(subjects) < 3:
        raise ValueError('Global expert training requires at least three subjects.')
    entries = _collect_subjects(args, subjects)
    shared_geometry_path, shared_geometry = _resolve_shared_geometry(args)
    shared_distance = shared_geometry['shared_rest_normalized_distance_matrix']

    rng = np.random.default_rng(args.random_seed)
    shuffled = np.asarray(subjects, dtype=np.int64)
    rng.shuffle(shuffled)
    val_count = min(max(1, args.global_validation_subjects), len(subjects) - 2)
    validation_subjects = set(shuffled[:val_count].tolist())
    train_entries = [entry for entry in entries if entry[0] not in validation_subjects]
    val_entries = [entry for entry in entries if entry[0] in validation_subjects]

    selection_raw = np.concatenate([entry[1].data for entry in train_entries], axis=0)
    selection_mean, selection_std = compute_normalization(selection_raw)
    train_bundle = _combine_entries(train_entries, selection_mean, selection_std)
    val_bundle = _combine_entries(val_entries, selection_mean, selection_std)
    selection_model, best_epoch, selection_history = _train_distilled(
        args, train_bundle, shared_distance=shared_distance, val_bundle=val_bundle)
    validation = evaluate_model(
        selection_model, val_bundle['data'], val_bundle['label'], args.batch_size,
        num_workers=args.num_workers)
    selection_temperature = fit_temperature(validation['logits'], validation['label'])
    reliability = probability_metrics(
        validation['label'],
        torch.softmax(torch.as_tensor(validation['logits']) / selection_temperature, dim=1).numpy())

    all_raw = np.concatenate([entry[1].data for entry in entries], axis=0)
    final_mean, final_std = compute_normalization(all_raw)
    final_bundle = _combine_entries(entries, final_mean, final_std)
    final_model, _, final_history = _train_distilled(
        args, final_bundle, shared_distance=shared_distance,
        fixed_epochs=max(1, best_epoch))
    prototype_subject_ids, subject_prototypes = _extract_subject_prototypes(
        final_model, entries, final_mean, final_std, args)

    channel_order = entries[0][1].channel_order
    graph_structure = entries[0][1].graph_structure
    subject_reliability = {
        str(subject): reliability_data
        for subject, _, _, reliability_data in entries
    }
    checkpoint = build_checkpoint(
        final_model, args, subject='global',
        extra_metadata={
            **normalization_to_checkpoint(final_mean, final_std),
            'channel_order': channel_order,
            'graph_structure': graph_structure,
            # The selection-model temperature is not transferable to a separately retrained final model.
            'temperature': 1.0,
            'selection_temperature': float(selection_temperature),
            'validation_reliability': reliability,
            'subject_reliability': subject_reliability,
            'prototype_subject_ids': torch.as_tensor(prototype_subject_ids, dtype=torch.long),
            'subject_prototypes': torch.as_tensor(subject_prototypes, dtype=torch.float32),
            'shared_geometry': {
                key: torch.as_tensor(value) if isinstance(value, np.ndarray) and value.dtype != object else value
                for key, value in shared_geometry.items()
                if key != 'subject_reliability'
            },
            'shared_geometry_path': shared_geometry_path,
            'distill_lambda': float(args.distill_lambda),
            'geometry_lambda': float(args.geometry_lambda),
            'global_objective': args.global_objective,
            'validation_subjects': sorted(validation_subjects),
            'training_epochs': int(best_epoch),
            'checkpoint_role': 'global_final_{}'.format(args.global_objective)
        })
    checkpoint['model_state_dict'] = checkpoint['state_dict']
    ensure_path(args.checkpoint_dir)
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    torch.save(checkpoint, checkpoint_path)

    summary = {
        'checkpoint': checkpoint_path,
        'global_objective': args.global_objective,
        'validation_subjects': sorted(validation_subjects),
        'training_epochs': int(best_epoch),
        'temperature': 1.0,
        'selection_temperature': selection_temperature,
        'validation_reliability': reliability,
        'subjects': subjects,
        'shared_geometry': shared_geometry_path,
        'selection_history': selection_history,
        'final_history': final_history
    }
    summary_path = os.path.join(args.checkpoint_dir, summary_name)
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print('Global expert saved:', checkpoint_path)
    return summary


def train_global_expert(args):
    if not getattr(args, 'run_global_ablations', False):
        return _train_one_global_expert(args)

    summaries = {}
    for objective in GLOBAL_OBJECTIVES:
        objective_args = copy.copy(args)
        objective_args.global_objective = objective
        summaries[objective] = _train_one_global_expert(
            objective_args,
            checkpoint_name='Global_{}_final.pth'.format(objective),
            summary_name='global_{}_summary.json'.format(objective))
    full_path = summaries['full']['checkpoint']
    checkpoint = torch.load(full_path, map_location='cpu', weights_only=False)
    torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'Global_final.pth'))
    with open(os.path.join(args.checkpoint_dir, 'global_ablation_manifest.json'),
              'w', encoding='utf-8') as file:
        json.dump(summaries, file, ensure_ascii=False, indent=2)
    return summaries
