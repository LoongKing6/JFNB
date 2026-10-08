import pprint
import random
import time
import h5py
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
from torch.utils.data import DataLoader
from models.networks import *
OPTIONAL_MODELS_AVAILABLE = True
try:
    from models.aaa import *
except ImportError:
    # Optional comparison models depend on torch_geometric. AT-DGNN does not.
    OPTIONAL_MODELS_AVAILABLE = False
from config.config import *
from types import SimpleNamespace

_, os.environ['CUDA_VISIBLE_DEVICES'] = set_config([])
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class EEGTensorDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


def set_gpu(x):
    torch.set_num_threads(1)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.max_split_size_mb = 1000
    np.random.seed(seed)


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def get_model(args):
    if args.model != 'AT-DGNN' and not OPTIONAL_MODELS_AVAILABLE:
        raise ImportError(
            'The selected comparison model requires optional dependencies from requirements.txt.')
    idx_local_graph = 0
    if args.model == 'LGGNet' or args.model == 'AT-DGNN' or args.model == 'TFDEEG':
        graph_file = 'num_chan_local_graph_{}.hdf'.format(args.graph_type)
        if not os.path.exists(graph_file):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            graph_file = os.path.join(project_root, graph_file)
        with h5py.File(graph_file, 'r') as graph_dataset:
            idx_local_graph = list(np.array(graph_dataset['data']))
    if args.model == 'AT-DGNN':
        model = ATDGNN(num_classes=args.num_class, input_size=args.input_shape, sampling_rate=args.target_rate,
                       num_T=args.T, out_graph=args.hidden, dropout_rate=args.dropout,
                       pool=args.pool, pool_step_rate=args.pool_step_rate, idx_graph=idx_local_graph, temporal_kernel=args.temporal_kernel,
                       layers_transformer=2, num_head=8, layers_len = 721) #EEGMat的layers_len = 181,

        '''BHTMNet'''
        # model = ATDGNN(
        #     num_chan=32, num_time=512, layers_transformer=1, hidden_graph=64, num_head=16,
        #     alpha=0.25, temporal_kernel=args.temporal_kernel, num_kernel=args.T,
        #     num_classes=args.num_class, depth=int(args.num_layers - 2), heads=args.AT,
        #     mlp_dim=args.AT, dim_head=args.AT, dropout=args.dropout)

    elif args.model == 'LGGNet':
        model = LGGNet(
            num_classes=args.num_class, input_size=args.input_shape,
            sampling_rate=args.target_rate,
            num_T=args.T, out_graph=args.hidden,
            dropout_rate=args.dropout,
            pool=args.pool, pool_step_rate=args.pool_step_rate,
            idx_graph=idx_local_graph)
    elif args.model == 'EEGNet':
        model = EEGNet(
            n_classes=args.num_class, channels=args.channels, sampling_rate=args.target_rate,
            input_size=args.input_shape,
            kernLength=0.25 * args.target_rate
        )
    elif args.model == 'DeepConvNet':
        model = DeepConvNet(
            n_classes=args.num_class, channels=args.channels,
            nTime=args.input_shape[2], dropout_rate=args.dropout)
    elif args.model == 'ShallowConvNet':
        model = ShallowConvNet(
            n_classes=args.num_class, channels=args.channels,
            nTime=args.input_shape[2], dropout_rate=args.dropout)
    elif args.model == 'EEG-TCNet':
        model = EEGTCNet(
            n_classes=args.num_class, in_channels=args.channels, kernLength=int(args.target_rate * 0.25))
    elif args.model == 'TCNet-Fusion':
        model = TCNet_Fusion(
            input_size=args.input_shape, n_classes=args.num_class, channels=args.channels,
            sampling_rate=args.target_rate)
    elif args.model == "TSception":
        model = TSception(
            input_size=args.input_shape, num_classes=args.num_class, sampling_rate=args.target_rate,
            num_T=args.T, num_S=args.T, hidden=args.hidden, dropout_rate=args.dropout)
    elif args.model == "ATCNet":
        model = ATCNet(input_size=args.input_shape, n_channel=args.channels, n_classes=args.num_class,
                       eegn_F1=50, eegn_D=2, eegn_kernelSize=50,
                       tcn_depth=2, activation='elu')
    elif args.model == "DGCNN":
        model = DGCNN(input_size=args.input_shape, batch_size=args.batch_size, k_adj=40, num_out=64, nclass=2)

    elif args.model == "TCANet":
        model = TCANet(
            input_size= args.input_shape,
            out_features=args.num_class,

            # 如需覆盖其他参数，可在这里传入
            # pooling_size=POOLING_SIZE, attn_heads=HEADS, attn_depth=DEPTH
        )
    elif args.model == "SFSWTS":
        channel_map = [(i // 6, i % 6) for i in range(32)]  # 6x6 网格
        model = SFSWTS(channel_map, H=6, W=6, seq_len=800, num_classes=2)


    elif args.model == "CCMTL":
        CCMTL_args = SimpleNamespace(
            eeg_channels=32,
            seq_len=512,
            lstm_hidden_size=64,
            n_units=128,
            n_classes=2,
            reduction_ratio=16,
            use_lstm=True,
            use_modulator=True,
            cnn_out_channels=64
        )

        # 构建模型
        model = CCMTL(CCMTL_args)

    elif args.model == "CFBM":
        if isinstance(args.input_shape, str):
            shape = tuple(map(int, args.input_shape.split(',')))
        else:
            # 假如在外部已解析为 tuple
            shape = args.input_shape
        model = CFBM(input_size=shape)

    elif args.model == "EmT":
        model = EmT(
            layers_graph=[1, 2],
            layers_transformer=4,
            num_adj=2,
            num_chan=args.input_shape[1],
            ###换数据时记得改这个
            num_feature=args.input_shape[2],
            hidden_graph=args.input_shape[1],
            K=4,
            num_head=8,
            dim_head=16,
            dropout=0.25,
            num_class=2,
            graph2token='Linear',
            encoder_type='GCN'
        )

    elif args.model == "EEGDeformer":
        model = Deformer(num_chan=32, num_time=800, temporal_kernel=11, num_kernel=64,
                       num_classes=2, depth=4, heads=16,
                       mlp_dim=16, dim_head=16, dropout=0.5)

    elif args.model == "DBGCN":
        bands = [(0.5, 4), (4, 8), (8, 12), (12, 30)]
        in_bands = len(bands)
        model = DBGCN(in_bands=in_bands, bilstm_hidden=64, node_feat_dim=128, gcn_hidden=128, n_classes=2,
                      dropout=0.5)

    elif args.model == "ACCNet":
        model = ACCNet(args, in_channels=args.input_shape[1], hidden_channels=args.input_shape[1], out_channels=2, num_channels=args.input_shape[1], signal_length=800)  # signal_length为数据长度

    elif args.model == "FG_HANet_Simple":
        model = FG_HANet_Simple(E=args.input_shape[1], F1=41, F2=38, L=41, fs=200, num_classes=2) #E为通道数

    elif args.model == "TFDEEG":
        model = TFDEEG(
            num_heads = 8, #EEGMat=4
            num_classes=args.num_class, input_size=args.input_shape,
            sampling_rate=args.target_rate,
            num_T=args.T, out_graph=args.hidden,
            dropout_rate=args.dropout,
            pool=args.pool, pool_step_rate=args.pool_step_rate,
            idx_graph=idx_local_graph)

    elif args.model == "MoCE":
        model = MoCE(
            num_regions=1,
            channels=args.input_shape[1],
            feature_dim=args.input_shape[1],
            hyper_dim=args.input_shape[1],
            num_classes=args.num_class
        )

    return model


def get_dataloader(data, label, batch_size, shuffle=True, num_workers=0):
    # load the data
    dataset = EEGTensorDataset(data, label)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers
    )
    return loader

"""老版get_metrics"""
# def get_metrics(y_pred, y_true, classes=None):
#     """
#     This function calculates the accuracy, f1 score and confusion matrix
#     Parameters
#     ----------
#     y_pred: the predicted label
#     y_true: the ground truth label
#     classes: the class labels
#     return: the accuracy, f1 score and confusion matrix
#     """
#     acc = accuracy_score(y_true, y_pred)
#     labels = np.unique(np.concatenate((np.asarray(y_true), np.asarray(y_pred))))
#     average = 'binary' if len(labels) <= 2 else 'macro'
#     f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
#
#     #Cohen's Kappa的代码
#     kappa = cohen_kappa_score(y_true, y_pred)
#
#     if classes is not None:
#         cm = confusion_matrix(y_true, y_pred, labels=classes)
#     else:
#         cm = confusion_matrix(y_true, y_pred)
#     return acc, f1, cm, kappa

"""适应500HZ的6分类gen脑区的get_metrics"""
def get_metrics(y_pred, y_true, classes=None):
    """
    Calculate accuracy, F1, confusion matrix and kappa.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    acc = accuracy_score(y_true, y_pred)

    # 六分类：统一用 macro，避免 binary 的 pos_label 问题
    if classes is not None:
        f1 = f1_score(y_true, y_pred, average='macro', labels=classes, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=classes)
    else:
        labels = np.unique(np.concatenate((y_true, y_pred)))
        f1 = f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels)

    kappa = cohen_kappa_score(y_true, y_pred)

    return acc, f1, cm, kappa

def get_trainable_parameter_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def L1Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(torch.abs(w))
    return err


def L2Loss(model, Lambda):
    w = torch.cat([x.view(-1) for x in model.parameters()])
    err = Lambda * torch.sum(w.pow(2))
    return err


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
       refer to: https://github.com/NVIDIA/DeepLearningExamples/blob/8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/ConvNets/image_classification/smoothing.py#L18
    """

    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
