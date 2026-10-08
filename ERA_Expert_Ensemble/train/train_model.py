import sys
import os
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from utils.utils import *
from config.config import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Check CUDA availability and set device

CUDA = torch.cuda.is_available()


_, os.environ['CUDA_VISIBLE_DEVICES'] = set_config([])
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LABEL_MAPPING = {
    'index_to_name': {
        0: '感性弱',
        1: '感性强',
        2: '理性强',
        3: '理性弱',
        4: '静息态'
    },
    'name_to_index': {
        '感性弱': 0,
        '感性强': 1,
        '理性强': 2,
        '理性弱': 3,
        '静息态': 4
    }
}

ERA_MAPPING = {
    'class_order': [0, 1, 2, 3, 4],
    'class_order_name': ['感性弱', '感性强', '理性强', '理性弱', '静息态'],
    'emotional_axis': {
        '静息态': 0.0,
        '感性弱': 1.0,
        '感性强': 2.0
    },
    'rational_axis': {
        '静息态': 0.0,
        '理性弱': 1.0,
        '理性强': 2.0
    },
    'coordinate_range': [0.0, 2.0],
    'normalized_coordinate_range': [0.0, 1.0],
    'formula': {
        'E': '0*P(静息态) + 1*P(感性弱) + 2*P(感性强)',
        'R': '0*P(静息态) + 1*P(理性弱) + 2*P(理性强)'
    }
}
EXTERNAL_INFERENCE_DATASETS = {'DEAP', 'MEEG', 'EEGMAT'}


def compute_era_from_prob(prob):
    """
    Compute ERA coordinates from softmax probabilities.
    Fixed class order:
    0=感性弱, 1=感性强, 2=理性强, 3=理性弱, 4=静息态
    """
    e_score = prob[:, 0] + 2.0 * prob[:, 1]
    r_score = prob[:, 3] + 2.0 * prob[:, 2]
    return prob, e_score, r_score, e_score / 2.0, r_score / 2.0


def compute_era_from_logits(logits):
    prob = torch.softmax(logits, dim=1)
    return compute_era_from_prob(prob)


def summarize_era_scores(e_scores, r_scores):
    if len(e_scores) == 0:
        return {'mean_E': 0.0, 'mean_R': 0.0}
    return {
        'mean_E': float(np.mean(e_scores)),
        'mean_R': float(np.mean(r_scores))
    }


def summarize_era_distribution(e_scores, r_scores, labels, num_classes):
    e_scores = np.asarray(e_scores, dtype=np.float32).reshape(-1)
    r_scores = np.asarray(r_scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    summary = {
        'mean_E': float(np.mean(e_scores)) if e_scores.size > 0 else 0.0,
        'mean_R': float(np.mean(r_scores)) if r_scores.size > 0 else 0.0,
        'class_centers': {}
    }

    for cls in range(num_classes):
        cls_mask = labels == cls
        cls_name = LABEL_MAPPING['index_to_name'].get(cls, f'class_{cls}')
        count = int(np.sum(cls_mask))
        if count > 0:
            summary['class_centers'][cls_name] = {
                'class_index': cls,
                'count': count,
                'mean_E': float(np.mean(e_scores[cls_mask])),
                'mean_R': float(np.mean(r_scores[cls_mask]))
            }
        else:
            summary['class_centers'][cls_name] = {
                'class_index': cls,
                'count': 0,
                'mean_E': None,
                'mean_R': None
            }
    return summary


def print_era_class_centers(prefix, summary):
    for class_name, center in summary['class_centers'].items():
        print('{} {} center -> E:{} R:{} count:{}'.format(
            prefix, class_name, center['mean_E'], center['mean_R'], center['count']))


def is_public_run(subject):
    return str(subject) == 'public'


def get_model_state_dict(model):
    return model.module.state_dict() if hasattr(model, 'module') else model.state_dict()


def maybe_parallelize_model(model, args):
    if CUDA:
        model = model.cuda()
        visible_gpu_count = torch.cuda.device_count()
        requested_multi_gpu = isinstance(args.gpu, str) and ',' in args.gpu
        if requested_multi_gpu and visible_gpu_count > 1 and not isinstance(model, nn.DataParallel):
            model = nn.DataParallel(model)
    return model


def align_state_dict_for_model(model, state_dict):
    model_has_module = hasattr(model, 'module')
    state_has_module = any(key.startswith('module.') for key in state_dict.keys())

    if model_has_module and not state_has_module:
        return {'module.' + key: value for key, value in state_dict.items()}
    if not model_has_module and state_has_module:
        return {key.replace('module.', '', 1): value for key, value in state_dict.items()}
    return state_dict


def build_checkpoint(model, args, subject=None, fold=None, extra_metadata=None):
    checkpoint = {
        'state_dict': get_model_state_dict(model),
        'config': vars(args),
        'label_mapping': LABEL_MAPPING,
        'era_mapping': ERA_MAPPING
    }
    if subject is not None:
        checkpoint['subject'] = subject
    if fold is not None:
        checkpoint['fold'] = fold
    if extra_metadata:
        checkpoint.update(extra_metadata)
    return checkpoint


def load_model_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if CUDA else 'cpu')
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    state_dict = align_state_dict_for_model(model, state_dict)
    model.load_state_dict(state_dict)
    return checkpoint


def resolve_test_checkpoint_path(args, reproduce, subject, fold):
    if reproduce and is_public_run(subject):
        data_type = f'model_{args.data_format}_{args.label_type}'
        experiment_setting = f'T_{args.T}_pool_{args.pool}'
        return os.path.join(args.save_path, experiment_setting, data_type, f'public_fold{fold}.pth')
    if reproduce and args.dataset not in EXTERNAL_INFERENCE_DATASETS:
        model_name = f'sub{subject}_fold{fold}.pth'
        data_type = f'model_{args.data_format}_{args.label_type}'
        experiment_setting = f'T_{args.T}_pool_{args.pool}'
        return os.path.join(args.save_path, experiment_setting, data_type, model_name)
    return args.load_path_final


def save_test_outputs(args, subject, fold, payload, summary):
    era_dir = os.path.join(args.save_path, 'era_results')
    ensure_path(era_dir)

    npz_path = os.path.join(era_dir, f'sub{subject}_fold{fold}_era_outputs.npz')
    np.savez_compressed(
        npz_path,
        prediction=np.asarray(payload['prediction'], dtype=np.int64),
        probability=np.asarray(payload['probability'], dtype=np.float32),
        E_score=np.asarray(payload['E_score'], dtype=np.float32),
        R_score=np.asarray(payload['R_score'], dtype=np.float32),
        E_score_norm=np.asarray(payload['E_score_norm'], dtype=np.float32),
        R_score_norm=np.asarray(payload['R_score_norm'], dtype=np.float32),
        label=np.asarray(payload['label'], dtype=np.int64)
    )

    summary_path = os.path.join(era_dir, f'sub{subject}_fold{fold}_era_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return npz_path, summary_path

def train_one_epoch(data_loader, net, loss_fn, optimizer):
    net.train()
    tl = Averager()
    pred_train = []
    act_train = []
    e_scores = []
    r_scores = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        if CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        out = net(x_batch)
        _, e_score, r_score, _, _ = compute_era_from_logits(out)
        loss = loss_fn(out, y_batch)
        _, pred = torch.max(out, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
        e_scores.extend(e_score.detach().cpu().tolist())
        r_scores.extend(r_score.detach().cpu().tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tl.add(loss.item())
    return tl.item(), pred_train, act_train, summarize_era_scores(e_scores, r_scores)


def predict(data_loader, net, loss_fn):
    net.eval()
    pred_val = []
    act_val = []
    vl = Averager()
    e_scores = []
    r_scores = []
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(data_loader):
            if CUDA:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            out = net(x_batch)
            _, e_score, r_score, _, _ = compute_era_from_logits(out)
            loss = loss_fn(out, y_batch)
            _, pred = torch.max(out, 1)
            vl.add(loss.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
            e_scores.extend(e_score.detach().cpu().tolist())
            r_scores.extend(r_score.detach().cpu().tolist())
    return vl.item(), pred_val, act_val, summarize_era_scores(e_scores, r_scores)


def set_up(args):
    """
    Set up GPU, random seed, and save path.
    """
    set_gpu(args.gpu)
    ensure_path(args.save_path)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True


def train(args, data_train, label_train, data_val, label_val, subject, fold):
    """
    Train model for one stage with early stopping.
    Records train and validation metrics to files.
    """
    seed_all(args.random_seed)
    save_name = f'_sub{subject}_fold{fold}'
    set_up(args)

    train_loader = get_dataloader(data_train, label_train, args.batch_size)
    val_loader = get_dataloader(data_val, label_val, args.batch_size)

    model = get_model(args)
    model = maybe_parallelize_model(model, args)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = LabelSmoothing(args.LS_rate) if args.LS else nn.CrossEntropyLoss()

    trlog = {
        'args': vars(args),
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'max_acc': 0.0,
        'F1': 0.0
    }

    timer = Timer()
    patient = args.patient
    counter = 0

    # Paths for logging results
    train_res_file = os.path.join(args.save_path, 'train_result.txt')
    val_res_file = os.path.join(args.save_path, 'validation_result.txt')

    for epoch in range(1, args.max_epoch + 1):
        # Training step
        loss_train, pred_train, act_train, train_era = train_one_epoch(
            data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer)
        acc_train, f1_train, _, kappa_train = get_metrics(y_pred=pred_train, y_true=act_train)
        print(
            f'epoch {epoch}, for the train set, loss={loss_train:.4f} acc={acc_train:.4f} '
            f'f1={f1_train:.4f} kappa={kappa_train:.4f} meanE={train_era["mean_E"]:.4f} meanR={train_era["mean_R"]:.4f}'
        )
        with open(train_res_file, 'a', encoding='utf-8') as f:
            f.write(f"SUB:{subject} FOLD:{fold}的训练集周期epoch {epoch}, for the train set, "
                    f"loss={loss_train:.4f} acc={acc_train:.4f} f1={f1_train:.4f} "
                    f"kappa={kappa_train:.4f} meanE={train_era['mean_E']:.4f} meanR={train_era['mean_R']:.4f}\n")

        # Validation step
        loss_val, pred_val, act_val, val_era = predict(
            data_loader=val_loader, net=model, loss_fn=loss_fn)
        acc_val, f1_val, _, kappa_val = get_metrics(y_pred=pred_val, y_true=act_val)
        print(
            f'epoch {epoch}, for the validation set, loss={loss_val:.4f} acc={acc_val:.4f} '
            f'f1={f1_val:.4f} kappa={kappa_val:.4f} meanE={val_era["mean_E"]:.4f} meanR={val_era["mean_R"]:.4f}'
        )
        with open(val_res_file, 'a', encoding='utf-8') as f:
            f.write(f"SUB:{subject} FOLD:{fold}的验证集周期epoch {epoch}, for the validation set, "
                    f"loss={loss_val:.4f} acc={acc_val:.4f} f1={f1_val:.4f} "
                    f"kappa={kappa_val:.4f} meanE={val_era['mean_E']:.4f} meanR={val_era['mean_R']:.4f}\n")

        # Early stopping logic
        if acc_val >= trlog['max_acc']:
            trlog['max_acc'] = acc_val
            trlog['F1'] = f1_val
            trlog['Kappa'] = kappa_val
            # Save candidate model
            candidate_path = os.path.join(args.save_path, 'candidate.pth')
            torch.save(get_model_state_dict(model), candidate_path)
            counter = 0
        else:
            counter += 1
            if counter >= patient:
                print('early stopping')
                break

        # Update training log
        trlog['train_loss'].append(loss_train)
        trlog['train_acc'].append(acc_train)
        trlog['val_loss'].append(loss_val)
        trlog['val_acc'].append(acc_val)
        print(f'ETA:{timer.measure()}/{timer.measure(epoch / args.max_epoch)} SUB:{subject} FOLD:{fold}')

    # Save training log object
    save_name = 'trlog' + save_name
    experiment_setting = f'T_{args.T}_pool_{args.pool}'
    log_dir = os.path.join(args.save_path, experiment_setting, 'log_train')
    ensure_path(log_dir)
    torch.save(trlog, os.path.join(log_dir, save_name))

    return trlog['max_acc'], trlog['F1'], trlog['Kappa']


def test(args, data, label, reproduce, subject, fold):
    """
    Test the model and record results.
    """
    set_up(args)
    seed_all(args.random_seed)

    test_loader = get_dataloader(data, label, args.batch_size, shuffle=False)
    model = get_model(args)
    model = maybe_parallelize_model(model, args)
    loss_fn = nn.CrossEntropyLoss()

    # Load the correct model weights
    load_path = resolve_test_checkpoint_path(args, reproduce, subject, fold)
    load_model_checkpoint(model, load_path)

    pred_test = []
    act_test = []
    prob_test = []
    e_scores = []
    r_scores = []
    e_scores_norm = []
    r_scores_norm = []
    loss_meter = Averager()

    model.eval()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            if CUDA:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            logits = model(x_batch)
            prob, e_score, r_score, e_score_norm, r_score_norm = compute_era_from_logits(logits)
            loss = loss_fn(logits, y_batch)
            _, pred = torch.max(logits, 1)

            loss_meter.add(loss.item())
            pred_test.extend(pred.detach().cpu().tolist())
            act_test.extend(y_batch.detach().cpu().tolist())
            prob_test.append(prob.detach().cpu().numpy())
            e_scores.extend(e_score.detach().cpu().tolist())
            r_scores.extend(r_score.detach().cpu().tolist())
            e_scores_norm.extend(e_score_norm.detach().cpu().tolist())
            r_scores_norm.extend(r_score_norm.detach().cpu().tolist())

    loss_test = loss_meter.item()
    prob_test = np.concatenate(prob_test, axis=0) if prob_test else np.empty((0, args.num_class), dtype=np.float32)

    """老版get_metrics"""
    # acc, f1, cm, kappa = get_metrics(y_pred=pred_test, y_true=act_test)
    """适应500HZ的6分类gen脑区的get_metrics"""
    acc, f1, cm, kappa = get_metrics(y_pred=pred_test, y_true=act_test, classes=list(range(args.num_class)))
    era_summary = summarize_era_distribution(e_scores, r_scores, act_test, args.num_class)
    era_summary.update({
        'loss': float(loss_test),
        'acc': float(acc),
        'f1': float(f1),
        'kappa': float(kappa),
        'label_mapping': LABEL_MAPPING,
        'era_mapping': ERA_MAPPING
    })

    print(
        f'>>> Test:  loss={loss_test:.4f} acc={acc:.4f} f1={f1:.4f} '
        f'kappa={kappa:.4f} meanE={era_summary["mean_E"]:.4f} meanR={era_summary["mean_R"]:.4f}'
    )
    print_era_class_centers('>>> Test:', era_summary)

    # Write to test_result.txt
    test_res_file = os.path.join(args.save_path, 'test_result.txt')
    with open(test_res_file, 'a', encoding='utf-8') as f:
        f.write(f"SUB:{subject} FOLD:{fold}的测试集Test:  loss={loss_test:.4f} "
                f"acc={acc:.4f} f1={f1:.4f} kappa={kappa:.4f} "
                f"meanE={era_summary['mean_E']:.4f} meanR={era_summary['mean_R']:.4f}\n")

    payload = {
        'prediction': pred_test,
        'probability': prob_test,
        'E_score': e_scores,
        'R_score': r_scores,
        'E_score_norm': e_scores_norm,
        'R_score_norm': r_scores_norm,
        'label': act_test
    }
    npz_path, summary_path = save_test_outputs(args, subject, fold, payload, era_summary)
    era_summary['npz_path'] = npz_path
    era_summary['summary_path'] = summary_path

    return acc, pred_test, act_test, payload, era_summary


def combine_train(args, data_train, label_train, subject, fold, target_acc):
    """
    Second-stage training (fine-tuning) until target accuracy.
    """
    save_name = f'_sub{subject}_fold{fold}'
    set_up(args)
    seed_all(args.random_seed)

    train_loader = get_dataloader(data_train, label_train, args.batch_size)
    model = get_model(args)
    model = maybe_parallelize_model(model, args)
    # Load initial model
    load_model_checkpoint(model, args.load_path)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate * 1e-1)
    loss_fn = LabelSmoothing(args.LS_rate) if args.LS else nn.CrossEntropyLoss()

    trlog = {
        'args': vars(args),
        'train_loss': [],
        'train_acc': [],
        'max_acc': 0.0,
        'F1': 0.0,
        'kappa': 0.0
    }

    timer = Timer()

    train_res_file = os.path.join(args.save_path, 'train_result.txt')

    for epoch in range(1, args.max_epoch_cmb + 1):
        loss_cmb, pred_cmb, act_cmb, cmb_era = train_one_epoch(
            data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer)
        acc_cmb, f1_cmb, _, kappa_cmb = get_metrics(y_pred=pred_cmb, y_true=act_cmb)
        print(
            f'Stage 2 : epoch {epoch}, for train set loss={loss_cmb:.4f} acc={acc_cmb:.4f} '
            f'f1={f1_cmb:.4f} kappa={kappa_cmb:.4f} meanE={cmb_era["mean_E"]:.4f} meanR={cmb_era["mean_R"]:.4f}'
        )
        with open(train_res_file, 'a', encoding='utf-8') as f:
            f.write(f"SUB:{subject} FOLD:{fold}的训练集周期epoch {epoch}, for train set "
                    f"loss={loss_cmb:.4f} acc={acc_cmb:.4f} f1={f1_cmb:.4f} "
                    f"kappa={kappa_cmb:.4f} meanE={cmb_era['mean_E']:.4f} meanR={cmb_era['mean_R']:.4f}\n")

        # Early stopping or target reached
        if acc_cmb >= target_acc or epoch == args.max_epoch_cmb:
            print('early stopping!')
            # Save final model for inference
            # final_name = 'public_final_model.pth' if is_public_run(subject) else 'final_model.pth'
            final_name = 'final_model.pth'
            final_checkpoint = build_checkpoint(model, args, subject=subject, fold=fold)
            torch.save(final_checkpoint, os.path.join(args.save_path, final_name))
            # Save reproduce model
            model_name = f'public_fold{fold}.pth' if is_public_run(subject) else f'sub{subject}_fold{fold}.pth'
            data_type = f'model_{args.data_format}_{args.label_type}'
            experiment_setting = f'T_{args.T}_pool_{args.pool}'
            save_dir = os.path.join(args.save_path, experiment_setting, data_type)
            ensure_path(save_dir)
            reproduce_checkpoint = build_checkpoint(model, args, subject=subject, fold=fold)
            torch.save(reproduce_checkpoint, os.path.join(save_dir, model_name))
            break

        trlog['train_loss'].append(loss_cmb)
        trlog['train_acc'].append(acc_cmb)
        print(f'ETA:{timer.measure()}/{timer.measure(epoch / args.max_epoch_cmb)} SUB:{subject} FOLD:{fold}')

    # Save combine training log
    log_name = 'trlog_comb' + save_name
    experiment_setting = f'T_{args.T}_pool_{args.pool}'
    log_dir = os.path.join(args.save_path, experiment_setting, 'log_train_cmb')
    ensure_path(log_dir)
    torch.save(trlog, os.path.join(log_dir, log_name))


"""Tecption不兼容extract_features版"""
# def extract_features(data_loader, net):
#     """提取模型分类层之前的特征向量"""
#     net.eval()
#     features = []
#     labels = []
#     with torch.no_grad():
#         for x_batch, y_batch in data_loader:
#             if CUDA:
#                 x_batch = x_batch.cuda()
#             # 手动前向传播，在 fc 层之前截取特征
#             x = x_batch
#             # AT-DGNN / LGGNet 的前向传播
#             if hasattr(net, 'Tception1'):
#                 y = net.Tception1(x)
#                 out = y
#                 y = net.Tception2(x)
#                 out = torch.cat((out, y), dim=-1)
#                 y = net.Tception3(x)
#                 out = torch.cat((out, y), dim=-1)
#                 if hasattr(net, 'feature_integrator'):
#                     out = net.feature_integrator(out)
#                     out = net.sliding_window_processor(out)
#                 out = torch.reshape(out, (out.size(0), out.size(1), -1))
#                 out = net.local_filter_fun(out, net.local_filter_weight)
#                 out = net.aggregate.forward(out)
#                 out = net.bn(out)
#                 if hasattr(net, 'dynamic_gcn'):
#                     out = net.dynamic_gcn(out)
#                 else:
#                     out = net.GCN(out)
#                 out = net.bn_(out)
#                 out = out.view(out.size(0), -1)
#             else:
#                 # 其他模型：直接用 forward 然后取 fc 之前的输出
#                 out = net(x)
#
#             features.append(out.cpu().numpy())
#             labels.append(y_batch.numpy())
#     return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

"""Tecption兼容extract_features版"""
def extract_features(data_loader, net):
    """提取模型分类层之前的特征向量"""
    net.eval()
    features = []
    labels = []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            if CUDA:
                x_batch = x_batch.cuda()
            x = x_batch

            # 兼容 TSception / AT-DGNN / LGGNet / 其他模型
            if all(hasattr(net, a) for a in ['Tception1', 'Tception2', 'Tception3']):
                y1 = net.Tception1(x)
                y2 = net.Tception2(x)
                y3 = net.Tception3(x)
                out = torch.cat((y1, y2, y3), dim=-1)

                if hasattr(net, 'feature_integrator'):
                    out = net.feature_integrator(out)

                if hasattr(net, 'sliding_window_processor'):
                    out = net.sliding_window_processor(out)

                if hasattr(net, 'local_filter_fun') and hasattr(net, 'local_filter_weight'):
                    out = net.local_filter_fun(out, net.local_filter_weight)

                if hasattr(net, 'aggregate'):
                    out = net.aggregate(out)
                elif hasattr(net, 'aggregate_fun'):
                    out = net.aggregate_fun(out)

                if hasattr(net, 'bn'):
                    out = net.bn(out)

                if hasattr(net, 'dynamic_gcn'):
                    out = net.dynamic_gcn(out)
                elif hasattr(net, 'GCN'):
                    out = net.GCN(out)

                if hasattr(net, 'bn_'):
                    out = net.bn_(out)

                out = out.view(out.size(0), -1)

            elif hasattr(net, 'forward_features'):
                out = net.forward_features(x)
                out = out.view(out.size(0), -1)

            else:
                out = net(x)
                if out.dim() > 2:
                    out = out.view(out.size(0), -1)

            features.append(out.cpu().numpy())
            labels.append(y_batch.cpu().numpy())

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def tsne_visualize(args, all_features, all_labels):
    """对所有受试者的特征进行 t-SNE 可视化"""
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    n_samples = all_features.shape[0]
    perplexity = min(args.tsne_perplexity, n_samples - 1)
    if n_samples < 2:
        print(f'Skipping t-SNE: only {n_samples} samples')
        return

    # t-SNE 降维
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=args.tsne_lr,
                n_iter=1000, random_state=args.random_seed)
    features_2d = tsne.fit_transform(all_features)

    # 绘制散点图
    plt.figure(figsize=(10, 8))
    num_classes = args.num_class
    class_names = getattr(args, 'class_names', None)

    if class_names is None:
        if args.dataset == 'EMO' and args.label_type == 'S':
            class_names = ['IEE', 'MEE', 'IIW', 'MIW', 'CAL', 'IMA']
        elif args.dataset == 'EMO' and args.label_type == 'A':
            class_names = ['Negative', 'Positive']
        elif args.dataset == 'EMO' and args.label_type == 'V':
            class_names = ['Weak', 'Strong']
        elif args.dataset == 'EMO' and args.label_type == 'D':
            class_names = ['Emotional', 'Rational']
        elif args.dataset == 'EMO' and args.label_type == 'L':
            class_names = ['Weak Rational', 'Strong Rational']

    # 高对比度颜色列表
    high_contrast_colors = [
        '#E63946',  # 红色
        '#A8DADC',  # 浅蓝
        '#2A9D8F',  # 青绿
        '#AB83A1',  # 淡紫
        '#F4A261',  # 橙色
        '#264653',  # 深青
        '#1D3557',  # 深蓝
        '#6A0572',  # 紫色
        '#E9C46A',  # 金黄
        '#D62828',  # 深红
    ]
    colors = high_contrast_colors[:num_classes]

    for cls in range(num_classes):
        idx = all_labels == cls
        name = class_names[cls] if class_names else f'Class {cls}'
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1],
                    c=colors[cls], label=name, alpha=0.75, s=30,
                    edgecolors='white', linewidths=0.5)

    plt.legend(fontsize=10)
    plt.title('t-SNE (All Subjects)', fontsize=14)
    plt.xlabel('t-SNE Dim 1', fontsize=12)
    plt.ylabel('t-SNE Dim 2', fontsize=12)
    plt.tight_layout()

    # 保存图片
    tsne_path = os.path.join(args.save_path, 'tsne')
    ensure_path(tsne_path)
    save_file = os.path.join(tsne_path, 'tsne_all_subjects.png')
    plt.savefig(save_file, dpi=600)
    plt.close()
    print(f't-SNE visualization saved to {save_file}')

def plot_confusion_matrix(args, y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    # 转换为百分比
    cm = cm.astype(np.float32)
    cm = cm / cm.sum(axis=1, keepdims=True)

    class_names = getattr(args, 'class_names', None)

    if class_names is None:
        if args.dataset == 'EMO' and args.label_type == 'S':
            class_names = ['IEE', 'MEE', 'IIW', 'MIW', 'CAL', 'IMA']


    plt.figure(figsize=(7,6))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        square=True
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")

    save_path = os.path.join(args.save_path, "confusion_matrix")
    ensure_path(save_path)

    plt.savefig(
        os.path.join(save_path, "confusion_matrix_all_subjects.png"),
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()
