import math

import numpy as np


def normalized_entropy(probability):
    probability = np.clip(np.asarray(probability, dtype=np.float32), 1e-8, 1.0)
    entropy = -np.sum(probability * np.log(probability), axis=1)
    return entropy / math.log(probability.shape[1])


def confidence_from_probability(probability):
    return np.clip(1.0 - normalized_entropy(probability), 0.05, 1.0)


def reliability_score(metadata):
    metrics = metadata or {}
    macro_f1 = max(float(metrics.get('macro_f1', metrics.get('accuracy', 0.5))), 1e-3)
    nll = max(float(metrics.get('nll', 1.0)), 0.0)
    ece = min(max(float(metrics.get('ece', 0.0)), 0.0), 1.0)
    return float(macro_f1 * np.exp(-min(nll, 10.0)) * (1.0 - ece))


def known_subject_probability_fusion(global_probability, individual_probability,
                                     individual_reliability, minimum_global_weight=0.1):
    global_probability = np.asarray(global_probability, dtype=np.float32)
    individual_probability = np.asarray(individual_probability, dtype=np.float32)
    reliability = float(np.clip(individual_reliability, 0.0, 1.0))
    alpha = np.clip(reliability, 0.0, 1.0 - float(minimum_global_weight))
    final = alpha * individual_probability + (1.0 - alpha) * global_probability
    final /= np.maximum(np.sum(final, axis=1, keepdims=True), 1e-8)
    return final, np.full(len(final), 1.0 - alpha, dtype=np.float32), np.full(
        len(final), alpha, dtype=np.float32)


def cosine_similarity(features, prototypes):
    features = np.asarray(features, dtype=np.float32)
    prototypes = np.asarray(prototypes, dtype=np.float32)
    feature_norm = np.linalg.norm(features, axis=1, keepdims=True)
    prototype_norm = np.linalg.norm(prototypes, axis=1, keepdims=True).T
    denominator = np.maximum(feature_norm * prototype_norm, 1e-8)
    return np.matmul(features, prototypes.T) / denominator


def softmax_similarity(similarity, temperature):
    scaled = similarity / max(float(temperature), 1e-6)
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_value = np.exp(scaled)
    return exp_value / np.maximum(np.sum(exp_value, axis=1, keepdims=True), 1e-8)


def adaptive_probability_fusion(global_probability, individual_probability,
                                global_reliability, individual_reliability,
                                similarity, global_min_weight=0.35,
                                similarity_threshold=0.05,
                                similarity_max=None, top_k=None):
    global_probability = np.asarray(global_probability, dtype=np.float32)
    individual_probability = np.asarray(individual_probability, dtype=np.float32)
    individual_reliability = np.asarray(individual_reliability, dtype=np.float32).reshape(1, -1)
    similarity = np.asarray(similarity, dtype=np.float32)

    global_confidence = confidence_from_probability(global_probability)
    individual_confidence = np.stack([
        confidence_from_probability(individual_probability[:, index, :])
        for index in range(individual_probability.shape[1])
    ], axis=1)

    individual_weight = individual_reliability * similarity * individual_confidence
    if top_k is not None and int(top_k) > 0 and int(top_k) < individual_weight.shape[1]:
        keep = np.argpartition(similarity, -int(top_k), axis=1)[:, -int(top_k):]
        top_k_mask = np.zeros_like(individual_weight, dtype=bool)
        rows = np.arange(len(individual_weight))[:, None]
        top_k_mask[rows, keep] = True
        individual_weight = np.where(top_k_mask, individual_weight, 0.0)
    global_weight = max(float(global_reliability), 1e-6) * global_confidence
    total = global_weight + np.sum(individual_weight, axis=1)
    global_weight = global_weight / np.maximum(total, 1e-8)
    individual_weight = individual_weight / np.maximum(total[:, None], 1e-8)

    if similarity_max is None:
        similarity_max = np.max(similarity, axis=1)
    low_similarity = np.asarray(similarity_max).reshape(-1) < float(similarity_threshold)
    global_weight = np.maximum(global_weight, float(global_min_weight))
    individual_total = np.sum(individual_weight, axis=1)
    target_individual_total = 1.0 - global_weight
    scale = target_individual_total / np.maximum(individual_total, 1e-8)
    individual_weight = individual_weight * scale[:, None]

    global_weight[low_similarity] = 1.0
    individual_weight[low_similarity] = 0.0

    final_probability = global_weight[:, None] * global_probability
    final_probability += np.sum(
        individual_weight[:, :, None] * individual_probability, axis=1)
    final_probability /= np.maximum(np.sum(final_probability, axis=1, keepdims=True), 1e-8)
    return final_probability, global_weight, individual_weight
