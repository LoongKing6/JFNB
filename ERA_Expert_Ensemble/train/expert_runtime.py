import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss
from torch.utils.data import DataLoader

from train.expert_data import ExpertDataset
from utils.utils import get_model, seed_all


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_model(args):
    model = get_model(args)
    return model.to(get_device())


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def find_last_linear(model):
    linear = None
    for module in unwrap_model(model).modules():
        if isinstance(module, nn.Linear):
            linear = module
    if linear is None:
        raise ValueError('The selected model does not contain a Linear classifier.')
    return linear


class FeatureCapture:
    """Capture the input of the final Linear layer without changing forward()."""

    def __init__(self, model):
        self.feature = None
        self.handle = find_last_linear(model).register_forward_pre_hook(self._capture)

    def _capture(self, module, inputs):
        self.feature = inputs[0]

    def close(self):
        self.handle.remove()


def expected_calibration_error(labels, probability, bins=15):
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float32)
    confidence = np.max(probability, axis=1)
    predictions = np.argmax(probability, axis=1)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    ece = 0.0
    for index in range(len(edges) - 1):
        lower, upper = edges[index], edges[index + 1]
        if index == 0:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence > lower) & (confidence <= upper)
        if not np.any(mask):
            continue
        bin_accuracy = np.mean(predictions[mask] == labels[mask])
        bin_confidence = np.mean(confidence[mask])
        ece += float(np.mean(mask)) * abs(float(bin_accuracy) - float(bin_confidence))
    return float(ece)


def brier_score(labels, probability):
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float32)
    target = np.eye(probability.shape[1], dtype=np.float32)[labels]
    return float(np.mean(np.sum((probability - target) ** 2, axis=1)))


def probability_metrics(labels, probability):
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float32)
    predictions = np.argmax(probability, axis=1)
    return {
        'accuracy': float(accuracy_score(labels, predictions)),
        'macro_f1': float(f1_score(labels, predictions, average='macro', zero_division=0)),
        'kappa': float(cohen_kappa_score(labels, predictions)),
        'nll': float(log_loss(labels, probability, labels=list(range(probability.shape[1])))),
        'ece': expected_calibration_error(labels, probability),
        'brier': brier_score(labels, probability)
    }


def classification_metrics(labels, logits):
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float32)
    probability = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
    return probability_metrics(labels, probability)


def fit_temperature(logits, labels):
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    candidates = torch.logspace(math.log10(0.5), math.log10(5.0), steps=80)
    losses = torch.stack([F.cross_entropy(logits / value, labels) for value in candidates])
    return float(candidates[int(torch.argmin(losses))].item())


def evaluate_model(model, data, labels, batch_size, return_features=False, num_workers=0):
    loader = DataLoader(
        ExpertDataset(data, labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    device = get_device()
    model.eval()
    logits_out = []
    labels_out = []
    features_out = []
    capture = FeatureCapture(model) if return_features else None
    try:
        with torch.no_grad():
            for x_batch, y_batch in loader:
                logits = model(x_batch.to(device))
                logits_out.append(logits.detach().cpu().numpy())
                labels_out.append(y_batch.numpy())
                if capture is not None:
                    features_out.append(capture.feature.detach().cpu().numpy())
    finally:
        if capture is not None:
            capture.close()

    result = {
        'logits': np.concatenate(logits_out, axis=0),
        'label': np.concatenate(labels_out, axis=0)
    }
    if return_features:
        result['feature'] = np.concatenate(features_out, axis=0)
    return result


def train_classifier(args, train_data, train_label, val_data=None, val_label=None,
                     fixed_epochs=None, sampler=None):
    seed_all(args.random_seed)
    device = get_device()
    model = create_model(args)
    train_dataset = ExpertDataset(train_data, train_label)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.LS_rate if args.LS else 0.0)

    max_epochs = int(fixed_epochs or args.expert_max_epoch)
    best_state = None
    best_epoch = max_epochs
    best_score = -np.inf
    patience = 0

    for epoch in range(1, max_epochs + 1):
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        if val_data is None:
            continue

        validation = evaluate_model(
            model, val_data, val_label, args.batch_size, num_workers=args.num_workers)
        metrics = classification_metrics(validation['label'], validation['logits'])
        score = metrics['macro_f1']
        if score > best_score + 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= args.expert_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


def probability_from_logits(logits, temperature=1.0):
    return torch.softmax(torch.as_tensor(logits, dtype=torch.float32) / float(temperature), dim=1).numpy()
