import sys
import os
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

'''如果想用第二张卡跑，就打开下面这个'''
os.environ['CUDA_VISIBLE_DEVICES'] = "1"



# Check CUDA availability and set device

CUDA = torch.cuda.is_available()


# _, os.environ['CUDA_VISIBLE_DEVICES'] = set_config()

def train_one_epoch(data_loader, net, loss_fn, optimizer):
    net.train()
    tl = Averager()
    pred_train = []
    act_train = []
    for i, (x_batch, y_batch) in enumerate(data_loader):
        if CUDA:
            x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

        out = net(x_batch)
        loss = loss_fn(out, y_batch)
        _, pred = torch.max(out, 1)
        pred_train.extend(pred.data.tolist())
        act_train.extend(y_batch.data.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tl.add(loss.item())
    return tl.item(), pred_train, act_train


def predict(data_loader, net, loss_fn):
    net.eval()
    pred_val = []
    act_val = []
    vl = Averager()
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(data_loader):
            if CUDA:
                x_batch, y_batch = x_batch.cuda(), y_batch.cuda()

            out = net(x_batch)
            loss = loss_fn(out, y_batch)
            _, pred = torch.max(out, 1)
            vl.add(loss.item())
            pred_val.extend(pred.data.tolist())
            act_val.extend(y_batch.data.tolist())
    return vl.item(), pred_val, act_val


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
    if CUDA:
        model = model.cuda()

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
        loss_train, pred_train, act_train = train_one_epoch(
            data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer)
        acc_train, f1_train, _, kappa_train = get_metrics(y_pred=pred_train, y_true=act_train)
        print(f'epoch {epoch}, for the train set, loss={loss_train:.4f} acc={acc_train:.4f} f1={f1_train:.4f} kappa={kappa_train:.4f}')
        with open(train_res_file, 'a', encoding='utf-8') as f:
            f.write(f"SUB:{subject} FOLD:{fold}的训练集周期epoch {epoch}, for the train set, "
                    f"loss={loss_train:.4f} acc={acc_train:.4f} f1={f1_train:.4f} kappa={kappa_train:.4f}\n")

        # Validation step
        loss_val, pred_val, act_val = predict(
            data_loader=val_loader, net=model, loss_fn=loss_fn)
        acc_val, f1_val, _, kappa_val = get_metrics(y_pred=pred_val, y_true=act_val)
        print(f'epoch {epoch}, for the validation set, loss={loss_val:.4f} acc={acc_val:.4f} f1={f1_val:.4f} kappa={kappa_val:.4f}')
        with open(val_res_file, 'a', encoding='utf-8') as f:
            f.write(f"SUB:{subject} FOLD:{fold}的验证集周期epoch {epoch}, for the validation set, "
                    f"loss={loss_val:.4f} acc={acc_val:.4f} f1={f1_val:.4f} kappa={kappa_val:.4f}\n")

        # Early stopping logic
        if acc_val >= trlog['max_acc']:
            trlog['max_acc'] = acc_val
            trlog['F1'] = f1_val
            trlog['Kappa'] = kappa_val
            # Save candidate model
            candidate_path = os.path.join(args.save_path, 'candidate.pth')
            torch.save(model.state_dict(), candidate_path)
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

    test_loader = get_dataloader(data, label, args.batch_size)
    model = get_model(args)
    if CUDA:
        model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()

    # Load the correct model weights
    if reproduce:
        model_name = f'sub{subject}_fold{fold}.pth'
        data_type = f'model_{args.data_format}_{args.label_type}'
        experiment_setting = f'T_{args.T}_pool_{args.pool}'
        load_path = os.path.join(args.save_path, experiment_setting, data_type, model_name)
        model.load_state_dict(torch.load(load_path))
    else:
        model.load_state_dict(torch.load(args.load_path_final))

    loss_test, pred_test, act_test = predict(
        data_loader=test_loader, net=model, loss_fn=loss_fn)
    acc, f1, cm, kappa = get_metrics(y_pred=pred_test, y_true=act_test)
    print(f'>>> Test:  loss={loss_test:.4f} acc={acc:.4f} f1={f1:.4f} kappa={kappa:.4f}')

    # Write to test_result.txt
    test_res_file = os.path.join(args.save_path, 'test_result.txt')
    with open(test_res_file, 'a', encoding='utf-8') as f:
        f.write(f"SUB:{subject} FOLD:{fold}的测试集Test:  loss={loss_test:.4f} "
                f"acc={acc:.4f} f1={f1:.4f}\n")

    return acc, pred_test, act_test


def combine_train(args, data_train, label_train, subject, fold, target_acc):
    """
    Second-stage training (fine-tuning) until target accuracy.
    """
    save_name = f'_sub{subject}_fold{fold}'
    set_up(args)
    seed_all(args.random_seed)

    train_loader = get_dataloader(data_train, label_train, args.batch_size)
    model = get_model(args)
    if CUDA:
        model = model.cuda()
    # Load initial model
    model.load_state_dict(torch.load(args.load_path))

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
        loss_cmb, pred_cmb, act_cmb = train_one_epoch(
            data_loader=train_loader, net=model, loss_fn=loss_fn, optimizer=optimizer)
        acc_cmb, f1_cmb, _, kappa_cmb = get_metrics(y_pred=pred_cmb, y_true=act_cmb)
        print(f'Stage 2 : epoch {epoch}, for train set loss={loss_cmb:.4f} acc={acc_cmb:.4f} f1={f1_cmb:.4f} kappa={kappa_cmb:.4f}')
        with open(train_res_file, 'a', encoding='utf-8') as f:
            f.write(f"SUB:{subject} FOLD:{fold}的训练集周期epoch {epoch}, for train set "
                    f"loss={loss_cmb:.4f} acc={acc_cmb:.4f} f1={f1_cmb:.4f} kappa={kappa_cmb:.4f}\n")

        # Early stopping or target reached
        if acc_cmb >= target_acc or epoch == args.max_epoch_cmb:
            print('early stopping!')
            # Save final model for inference
            final_name = 'final_model.pth'
            torch.save(model.state_dict(), os.path.join(args.save_path, final_name))
            # Save reproduce model
            model_name = f'sub{subject}_fold{fold}.pth'
            data_type = f'model_{args.data_format}_{args.label_type}'
            experiment_setting = f'T_{args.T}_pool_{args.pool}'
            save_dir = os.path.join(args.save_path, experiment_setting, data_type)
            ensure_path(save_dir)
            torch.save(model.state_dict(), os.path.join(save_dir, model_name))
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


def extract_features(data_loader, net):
    """提取模型分类层之前的特征向量"""
    net.eval()
    features = []
    labels = []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            if CUDA:
                x_batch = x_batch.cuda()
            # 手动前向传播，在 fc 层之前截取特征
            x = x_batch
            # AT-DGNN / LGGNet 的前向传播
            if hasattr(net, 'Tception1'):
                y = net.Tception1(x)
                out = y
                y = net.Tception2(x)
                out = torch.cat((out, y), dim=-1)
                y = net.Tception3(x)
                out = torch.cat((out, y), dim=-1)
                if hasattr(net, 'feature_integrator'):
                    out = net.feature_integrator(out)
                    out = net.sliding_window_processor(out)
                out = torch.reshape(out, (out.size(0), out.size(1), -1))
                out = net.local_filter_fun(out, net.local_filter_weight)
                out = net.aggregate.forward(out)
                out = net.bn(out)
                if hasattr(net, 'dynamic_gcn'):
                    out = net.dynamic_gcn(out)
                else:
                    out = net.GCN(out)
                out = net.bn_(out)
                out = out.view(out.size(0), -1)
            else:
                # 其他模型：直接用 forward 然后取 fc 之前的输出
                out = net(x)

            features.append(out.cpu().numpy())
            labels.append(y_batch.numpy())
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
        '#6A0572',  # 紫色

        '#1D3557',  # 深蓝
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
