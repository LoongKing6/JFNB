import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
import numpy as np
from config.config import *
try:
    from torch_geometric.utils import to_dense_batch
except ImportError:
    to_dense_batch = None

import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.utils import weight_norm
try:
    from einops import rearrange
    from einops.layers.torch import Rearrange
except ImportError:
    rearrange = None
    Rearrange = None

_, os.environ['CUDA_VISIBLE_DEVICES'] = set_config([])
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# STFT after Attention(preprocessing jian fa)
class ATDGNN(nn.Module):
#temporal_learner 方法定义了一个时间卷积块，用于提取时间特征。包含一个二维卷积层和一个功率层，用于对卷积后的特征进行幂变换。
    def temporal_learner(
            self, in_chan, out_chan, kernel, pool, pool_step_rate):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
            # PowerLayer(dim=-1, length=pool, step=int(pool_step_rate * pool))
            TeLU(),
            nn.MaxPool2d(kernel_size=(1, pool), stride=(1, int(pool_step_rate * pool)))

        )
#__init__ 方法初始化了 ATDGNN 模型的各种参数和层
    def __init__(self, num_classes, input_size, sampling_rate, num_T,
                 out_graph, dropout_rate, pool, pool_step_rate, idx_graph, temporal_kernel, layers_transformer, num_head, layers_len):
        super(ATDGNN, self).__init__()

        self.num_T = num_T
        self.out_graph = out_graph
        self.dropout_rate = dropout_rate
        self.window = [0.5, 0.25, 0.125]
        self.pool = pool
        self.pool_step_rate = pool_step_rate
        self.idx = idx_graph
        self.channel = input_size[1]
        self.brain_area = len(self.idx)
        ###################
        # 多头注意力相关参数
        self.model_dim = round(num_T / 2)
        self.num_heads = 8
        if sampling_rate == 200:
            self.window_size = 100
            self.stride = 20
        else:
            self.window_size = 64
            self.stride = 16
        ###################
        hidden_features = input_size[2]

        # by setting the convolutional kernel being (1,lenght) and the strids being 1, we can use conv2d to
        # achieve the 1d convolution operation.
        self.Tception = self.temporal_learner(input_size[0], num_T,
                                              (1, int(0.125 * sampling_rate)), pool, pool_step_rate)
        # Batch normalization layers
        self.bn_t = nn.BatchNorm2d(num_T)
        self.bn_s = nn.BatchNorm2d(num_T)
        self.OneXOneConv = nn.Sequential(
            nn.Conv2d(num_T, num_T, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, 2))
        )
        #######################################
        # 特征整合、滑动窗口相关配置
        self.feature_integrator = FeatureIntegrator(sr=sampling_rate , in_channels=self.channel, out_channels=self.channel)

        self.transformer = TTransformer(
            temporal_kernel=temporal_kernel,
            depth=layers_transformer,
            dim=self.channel, heads=num_head,
            dim_head=self.channel, dropout=0.1, mlp_dim=self.channel,
            alpha=0.25
        )

        self.layers_len = layers_len # change it

        # self.linear_TM = nn.Linear(self.layers_len, 191, device=DEVICE)
        self.linear_TM = nn.Linear(self.layers_len, self.layers_len*2, device=DEVICE)
        #######################################
        # diag(W) to assign a weight to each local areas
        size = self.get_size_temporal(input_size)
        # 表示局部滤波器的权重。它被定义为一个形状为(self.channel, size[-1])的浮点型张量，并设置为需要梯度计算（requires_grad=True）
        self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel, size[-1]),
                                                requires_grad=True)
        # 用来对local_filter_weight进行初始化，采用的是Xavier均匀分布初始化方法
        nn.init.xavier_uniform_(self.local_filter_weight)
        # 表示局部滤波器的偏置。它被定义为一个形状为(1, self.channel, 1)的浮点型张量，初始值为全零，并设置为需要梯度计算
        self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel, 1), dtype=torch.float32),
                                              requires_grad=True)
        # aggregate function
        self.aggregate = Aggregator(self.idx)

        #树卷积
        # 自动划分半球 ——
        N = len(self.idx)
        half = N // 2
        left_regions = list(range(0, half))  # 前半子区
        right_regions = list(range(half, N))  # 后半子区
        self.tree_conv = GlobalTreeConvSimple(
            idx=self.idx,
            left_idxs=left_regions,
            right_idxs=right_regions,
            in_f=size[-1],
            out_f=out_graph
        )

        # 表示全局邻接矩阵。它被定义为浮点型张量，并设置为需要梯度计算（requires_grad=True）
        self.global_adj = nn.Parameter(torch.FloatTensor(self.brain_area, self.brain_area), requires_grad=True)
        # 根据给定的张量的形状和分布进行参数初始化。用来对global_adj进行初始化，采用的是Xavier均匀分布初始化方法。
        nn.init.xavier_uniform_(self.global_adj)
        # 改了： 14 子区 + 1 全局节点
        self.bn = nn.BatchNorm1d(len(self.idx) + 2)
        self.bn_ = nn.BatchNorm1d(len(self.idx) + 2)

        # 改了： 全连接，输入维度 = (14+1)*out_graph
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear((len(self.idx) + 2) * out_graph, num_classes)
        )
        self.to(DEVICE)

    def get_size_temporal(self, input_size):
        # input_size: frequency x channel x data point
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        out = self.Tception(data)  # 第一分支（dilation=1）(20,64,32,196)
        #######################################
        out = self.feature_integrator(out)  # 特征整合和降维out=(1,32,547)
        out = out.permute(0, 2, 1)
        out = self.transformer(out)
        out = out.permute(0, 2, 1)
        out = self.linear_TM(out)
        #######################################
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        size = out.size()
        return size

    # 定义局部滤波器的前向传播函数
    def local_filter_fun(self, x, w):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def forward(self, x):
        x = x.to(DEVICE)
        # Temporal convolution
        out = self.Tception(x)  # 第一分支（dilation=1）(20,64,32,196)
        ##############################
        out = self.feature_integrator(out)  # 特征整合和降维

        out = out.permute(0, 2, 1)
        out = self.transformer(out)
        out = out.permute(0, 2, 1)
        out = self.linear_TM(out)
        ##############################
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        out = self.local_filter_fun(out, self.local_filter_weight) #(20,1,32,800)
        out = self.aggregate.forward(out) # [B, 14, 2300]
        # 2) 树形聚合（方案二简单版）→ [B,15,out_graph]
        out = self.tree_conv(out)# [B,15,32]
        # 3) 标准化
        out = self.bn(out)
        out = self.bn_(out)
        # 4) 平坦化 + FC
        out = out.view(out.size(0), -1)  # [B, 15*out_graph]
        out = self.fc(out)  # [B, num_classes]
        return out

##############################

"""简化版量子扩散图卷积"""
class GraphConvolution_tree(torch.nn.Module):
    """
    简化版量子扩散图卷积：
    使用固定演化时间 t，计算量子扩散矩阵 Π 并归一化后聚合特征。
    """
    def __init__(self, in_features, out_features, t=1.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 可学习的权重和偏置，与原GraphConvolution一致
        self.weight = torch.nn.Parameter(torch.randn(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        # 固定的量子演化时间
        self.t = t

    def forward(self, x, adj):
        # x: (B, N, in_features), adj: (B, N, N) 或 (N, N)
        B, N, _ = x.shape
        # 保证批次维度
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
        # 计算图拉普拉斯 H = D - A
        deg = torch.sum(adj, dim=-1)            # (B, N)
        H = torch.diag_embed(deg) - adj         # (B, N, N)
        # 复数矩阵指数 U = exp(-i H t)
        H_c = torch.complex(H, torch.zeros_like(H))
        U = torch.linalg.matrix_exp(-1j * H_c * self.t)
        # 量子转移核 Π = |U|^2
        Pi = U.abs() ** 2                       # (B, N, N), 实值
        # 对称归一化 Ῠ = D_t^{-1/2} Π D_t^{-1/2}
        degPi = torch.sum(Pi, dim=-1)           # (B, N)
        degPi[degPi == 0] = 1                   # 避免除零
        d_inv = torch.pow(degPi, -0.5)
        D_inv = torch.diag_embed(d_inv)         # (B, N, N)
        normPi = torch.bmm(torch.bmm(D_inv, Pi), D_inv)  # (B, N, N)
        # 线性变换
        out_lin = torch.matmul(x, self.weight)  # (B, N, out_features)
        if self.bias is not None:
            out_lin = out_lin - self.bias       # 减去偏置（与原实现保持一致）
        # 聚合（消息传递）
        output = torch.bmm(normPi, out_lin)     # (B, N, out_features)
        return F.relu(output)

"""多频段量子扩散图卷积(DE), 在全脑"""
class GraphConvolution_tree1(nn.Module):
    """
    向量化高性能版 GraphConvolution
    - 保持接口: forward(x, adj)
    - 输入:  x -> (B, N, T) 或 (B, N, K)
    - 输出:  (B, N, out_features)

    feature:
        'de'    : 差分熵
        'de-c'  : 差分熵（高性能版里用频域掩码近似 band-pass，避免 SciPy 循环）
        'psd'   : PSD
        'rpsd'  : 相对 PSD
    """

    DEFAULT_BANDS = [
        (1.0, 4.0),
        (5.0, 8.0),
        (9.0, 12.0),
        (13.0, 16.0),
        (17.0, 20.0),
        (21.0, 24.0),
        (25.0, 28.0),
        (29.0, 32.0),
        (33.0, 36.0),
        (37.0, 40.0),
        (41.0, 44.0),
        (45.0, 50.0),
    ]



    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        feature="de",
        bands=None,
        fs=200,
        eps=1e-8,
        normalize_adj=False,
    ):
        super().__init__()
        self.in_features = in_features # 17472
        self.out_features = out_features # 32
        self.feature = feature.lower()   # 'de', 'de-c', 'psd', 'rpsd'
        self.fs = fs
        self.eps = eps
        self.normalize_adj = normalize_adj

        if bands is None:
            bands = self.DEFAULT_BANDS
        self.bands = [tuple(map(float, b)) for b in bands]
        self.num_bands = len(self.bands)

        # 频带特征投影权重：实际输入是 K 维 band 特征，因此这里按 K 初始化
        self.weight = Parameter(torch.empty(self.num_bands, out_features)) # (7,32)
        nn.init.xavier_uniform_(self.weight, gain=1.414)

        if bias:
            self.bias = Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        # 缓存不同 T 对应的 band mask，加速重复调用
        self._mask_cache = {}
        self.t = 1.0

    def _get_band_masks(self, T, device, dtype):
        """
        返回 shape = (K, F) 的 band mask
        K: 频带数
        F: rFFT 频点数 = T//2 + 1
        """
        cache_key = (int(T), str(device), str(dtype), float(self.fs))
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        freqs = torch.fft.rfftfreq(T, d=1.0 / self.fs).to(device=device)
        masks = []
        for low, high in self.bands:
            m = (freqs >= low) & (freqs < high)
            masks.append(m.to(dtype))
        masks = torch.stack(masks, dim=0)  # (K, F)

        self._mask_cache[cache_key] = masks
        return masks

    def _extract_band_features(self, x):
        """
        x: (B, N, T) 原始时间序列
        return:
            (B, N, K) 频带特征
        """
        B, N, T = x.shape

        # 如果输入已经是 band feature（例如你提前做成 7 维/9 维），直接返回
        if T == self.num_bands:
            return x

        fft = torch.fft.rfft(x, dim=-1)  # (B, N, F)
        masks = self._get_band_masks(T, x.device, fft.real.dtype)  # (K, F)

        if self.feature in ("psd", "rpsd"):
            # PSD: |FFT|^2
            psd = (fft.real.square() + fft.imag.square()) / float(T)  # (B, N, F)

            # (B, K, N, F) -> sum over F -> (B, K, N)
            band_power = (psd[:, None, :, :] * masks[None, :, None, :]).sum(dim=-1)
            band_power = band_power.permute(0, 2, 1).contiguous()  # (B, N, K)

            if self.feature == "rpsd":
                total_power = psd.sum(dim=-1, keepdim=True)  # (B, N, 1)
                band_power = band_power / (total_power + self.eps)

            return band_power

        elif self.feature in ("de", "de-c"):
            # 频域掩码版 band-pass，再做差分熵
            # 这样比 scipy 循环快很多，也能保持 batch 并行
            band_fft = fft[:, None, :, :] * masks[None, :, None, :]   # (B, K, N, F)
            band_sig = torch.fft.irfft(band_fft, n=T, dim=-1)         # (B, K, N, T)

            var = band_sig.var(dim=-1, unbiased=False).clamp_min(self.eps)  # (B, K, N)
            de = 0.5 * torch.log(2.0 * math.pi * math.e * var)              # (B, K, N)
            de = de.permute(0, 2, 1).contiguous()                           # (B, N, K)
            return de

        else:
            raise ValueError(f"Unsupported feature type: {self.feature}")

    def _normalize_adj(self, adj):
        """
        对称归一化 D^{-1/2} A D^{-1/2}
        adj: (B, N, N)
        """
        rowsum = adj.sum(dim=-1)  # (B, N)
        rowsum = rowsum.masked_fill(rowsum == 0, 1.0)
        d_inv_sqrt = rowsum.pow(-0.5)
        return d_inv_sqrt.unsqueeze(-1) * adj * d_inv_sqrt.unsqueeze(-2)

    def forward(self, x, adj):
        """
        x:   (B, N, T) 或 (B, N, K)
        adj: (N, N) 或 (B, N, N)
        return: (B, N, out_features)
        """
        if x.dim() != 3:
            raise ValueError(f"x must be 3D tensor (B, N, T), but got {x.shape}")

        B, N, _ = x.shape #(20,14,17472)

        # adj 统一成 (B, N, N)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, -1, -1)
        elif adj.dim() != 3:
            raise ValueError(f"adj must be 2D or 3D tensor, but got {adj.shape}")

        if self.normalize_adj:
            adj = self._normalize_adj(adj)

        # 1) 先把原始长序列压成 band 特征
        x = self._extract_band_features(x)  # (B, N, K)=(20,14,12)

        # 2) 频带特征做线性映射
        x = torch.matmul(x, self.weight)  # (B, N, out_features)=(20,14,32)

        # 计算图拉普拉斯 H = D - A
        deg = torch.sum(adj, dim=-1)  # (B, N)=(20,14)
        H = torch.diag_embed(deg) - adj  # (B, N, N)=(20,14,14)
        # 复数矩阵指数 U = exp(-i H t)
        H_c = torch.complex(H, torch.zeros_like(H))
        U = torch.linalg.matrix_exp(-1j * H_c * self.t)
        # 量子转移核 Π = |U|^2
        Pi = U.abs() ** 2  # (B, N, N), 实值
        # 对称归一化 Ῠ = D_t^{-1/2} Π D_t^{-1/2}
        degPi = torch.sum(Pi, dim=-1)  # (B, N)
        degPi[degPi == 0] = 1  # 避免除零
        d_inv = torch.pow(degPi, -0.5)
        D_inv = torch.diag_embed(d_inv)  # (B, N, N)
        normPi = torch.bmm(torch.bmm(D_inv, Pi), D_inv)  # (B, N, N)
        # 线性变换
        # out_lin = torch.matmul(x, self.weight)  # (B, N, out_features)
        if self.bias is not None:
            out_lin = x - self.bias  # 减去偏置（与原实现保持一致）
        # 聚合（消息传递）
        output = torch.bmm(normPi, out_lin)  # (B, N, out_features)
        return F.relu(output)


class GlobalTreeConvSimple(nn.Module):
    """
    hemi-first design:
      1) 半球融合（hemi_pair，从原始叶子特征计算 left/right 均值，在特征维拼接 -> 1-node fuse）
         输入 x: [B, N, in_f] -> hemi_fuse: [B,1,out_f]
      2) 将每个叶子用一个轻投影映射到 out_f（leaf projection）
      3) 拼接叶子 + hemi_fuse -> 形成 (N+1) 节点，计算 global（对 N+1 节点取均值）
      4) 拼成 N+2 节点（leaf + hemi + global）并做最终 GCN -> [B, N+2, out_f]

    使用说明：left_idxs/right_idxs 是叶子索引列表（0-based）。
    """
    def __init__(self, idx, left_idxs, right_idxs, in_f, out_f, device='cpu'):
        super().__init__()
        self.idx = idx
        self.N = len(idx)
        self.left_idxs = left_idxs
        self.right_idxs = right_idxs
        self.device = device

        # 半球融合：输入 2*in_f -> out_f
        self.gc_hemi_fuse = GraphConvolution_tree(2 * in_f, out_f)

        # 叶子投影（把原始 in_f→out_f）便于后续统一维度
        # self.gc_leaf_proj = GraphAttentionGaussianFast(in_f, out_f)
        self.gc_leaf_proj = GraphConvolution_tree1(in_f, out_f)

        # 全图最终层：在 (N+2) 个节点上做一次 GCN -> out_f
        self.gc_final = GraphConvolution_tree(out_f, out_f)

        # 邻接矩阵构建
        # adj_leaf_proj: identity for leaf projection (N x N)
        adj_leaf = torch.eye(self.N, dtype=torch.float32)
        self.register_buffer('adj_leaf', adj_leaf)

        # adj_hemi_fuse: single-node self-loop (1x1)
        self.register_buffer('adj_hemi', torch.eye(1, dtype=torch.float32))

        # adj_final: size (N+2)   indices: 0..N-1 leaves, N = hemi, N+1 = global
        M = self.N + 2
        adj_final = torch.zeros(M, M, dtype=torch.float32)
        # connect leaves and hemi to global (index = N+1)
        for i in list(range(self.N)) + [self.N]:
            adj_final[i, self.N + 1] = 1.0
            adj_final[self.N + 1, i] = 1.0
        # add self-loops
        adj_final += torch.eye(M, dtype=torch.float32)
        self.register_buffer('adj_final', adj_final)

    def forward(self, x):
        """
        x: [B, N, in_f]
        returns: out [B, N+2, out_f]  (leaves + hemi + global)
        """
        B, N, Fin = x.shape
        assert N == self.N, f"Expect N={self.N}, got {N}"
        # ---------- 1) 半球融合（在原始特征上做 hemi） ----------
        # left/right mean from original leaves
        left_feat = x[:, self.left_idxs, :].mean(dim=1, keepdim=True)   # [B,1,Fin]
        right_feat = x[:, self.right_idxs, :].mean(dim=1, keepdim=True) # [B,1,Fin]

        # hemi pair 拼接在特征维 -> [B,1,2*Fin]
        hemi_pair = torch.cat([left_feat, right_feat], dim=-1)

        # 单节点 GCN 融合 -> [B,1,out_f]
        hemi_fuse = self.gc_hemi_fuse(hemi_pair, self.adj_hemi)  # [B,1,out_f]

        # ---------- 2) 叶子投影到 out_f ----------
        h_leaves = self.gc_leaf_proj(x, self.adj_leaf)  # [B,N,out_f]


        # ---------- 3) 形成 N+1 节点并生成 global全局 ----------
        in_Np1 = torch.cat([h_leaves, hemi_fuse], dim=1)   # [B, N+1, out_f]
        global_feat = in_Np1.mean(dim=1, keepdim=True)    # [B,1,out_f]

        # ---------- 4) 拼接为 N+2 并做 final GCN ----------
        in_final = torch.cat([in_Np1, global_feat], dim=1)  # [B, N+2, out_f]
        out = self.gc_final(in_final, self.adj_final)       # [B, N+2, out_f]
        return out

#######################################################

class TeLU(nn.Module):
    """
    TeLU activation: x * tanh(exp(x))
    为了数值稳定性，对 exp 的输入做了截断（clamp）。
    clamp_range 可调，通常取 [-20, 20] 已足够安全。
    """
    def __init__(self, clamp_min: float = -20.0, clamp_max: float = 20.0):
        super().__init__()
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 截断后计算，避免 exp 导致的溢出或 inf
        x_clamped = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
        return x * torch.tanh(torch.exp(x_clamped))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.act = TeLU() #用TeLU
        # self.act = nn.GELU()
        # self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim, hidden_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.to(DEVICE)

    def forward(self, x, **kwargs):
        x = x.to(DEVICE)
        return self.fn(self.norm(x), **kwargs)

class STA(nn.Module):
    def __init__(self, temporal_kernel ,heads ,dropout, alpha):
        super().__init__()
        self.dropout = nn.Dropout(alpha * dropout)

        self.cnn_low = weight_norm(nn.Conv2d(heads, heads, (3, 1),
                              stride=1, padding=self.get_padding(3)))


        self.bn = nn.BatchNorm2d(heads)

    def forward(self, x):
        x = self.dropout(x) #(16,16,408,16)
        x1 = self.cnn_low(x)


        return x1
    def get_padding(self, kernel):
        return (int(0.5 * (kernel - 1)), 0)

class Attention(nn.Module):
    def __init__(self, dim, temporal_kernel ,heads = 8, dim_head = 64, anchor=3, dropout = 0., alpha=0.25):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.STA = STA(temporal_kernel, heads, dropout, alpha)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = self.STA(out)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FrequencyBranchSimp(nn.Module):
    def __init__(self, n_fft=64, hop_length=32, win_length=64):
        super(FrequencyBranchSimp, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        # 提前创建 Hann 窗口，并注册为 buffer，避免每次计算时重复创建
        self.register_buffer("hann_window", torch.hann_window(win_length))

    def forward(self, x):
        """
        输入 x: (B, model_dim, window_size)
        输出: (B, model_dim, window_size)
        处理流程：
         1. 将 x 重新 reshape 为 (B * model_dim, window_size)
         2. 计算 STFT 得到幅值谱，形状 (B * model_dim, F, T_f)，其中 F = n_fft//2 + 1
         3. 对频率维度求平均，得到 (B * model_dim, T_f)
         4. 恢复为 (B, model_dim, T_f)，若 T_f 与 window_size 不同则进行上采样
        """
        B, C, T = x.shape  # C == model_dim
        # 合并批次和通道，使用 reshape 替换 view
        x_reshaped = x.reshape(B * C, T)
        # 计算 STFT（批量计算）
        stft_result = torch.stft(
            x_reshaped,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window,
            return_complex=True
        )  # (B*C, F, T_f)
        mag = stft_result.abs()  # (B*C, F, T_f)
        # 对频率维度求平均，得到简单的频域特征
        mag_mean = mag.mean(dim=1)  # (B*C, T_f)
        # 恢复为 (B, C, T_f)
        mag_mean = mag_mean.reshape(B, C, -1)
        # 如果时间步 T_f 与原始窗口 T 不一致，则上采样回 T
        if mag_mean.size(-1) != T:
            mag_mean = F.interpolate(mag_mean, size=T, mode='nearest')
        return mag_mean  # (B, model_dim, window_size)


'''第四版 简单版：加入FrequencyBranchSimp分支'''
class TTransformer(nn.Module):
    def __init__(self, dim, temporal_kernel, depth, heads, dim_head, mlp_dim,
                 n_fft=64, hop_length=32, win_length=64, dropout=0.1, alpha=0.25):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.freqbranch = FrequencyBranchSimp(n_fft, hop_length, win_length)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, temporal_kernel, heads=heads, dim_head=dim_head,
                                       dropout=dropout, alpha=alpha)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]))
        self.to(DEVICE)
    def forward(self, x):
        x = x.to(DEVICE) # x=()
        stack_time =[]
        open = False
        # x: [batch, seq_len, dim]
        for attn, ff in self.layers:
            if(open):
                # 时间维度下采样
                x = F.max_pool1d(x.transpose(1,2), kernel_size=2, padding=0).transpose(1,2)
            # 在下采样序列上做注意力
            x_attn = attn(x) + x


            x_branch = self.freqbranch(x_attn.transpose(1,2)).transpose(1,2)
            # 融合所有分支
            x = x_attn + x_branch
            x = ff(x) + x
            stack_time.append(x)
            open = True
        x = torch.cat(stack_time, dim= 1) #(1,286,32)
        return x

#######################################################
#sample_rate=200
#简单版的两个原模块融合
class FrequencyDetailBranch(nn.Module):
    """
    频域细节提取分支——把谱图当 RGB 图像来用 CNN 抽特征
    输入: x (B, C, T)
    输出: y (B, C, T)  # 保持与支路 A 一致的形状
    """
    def __init__(self,channels, n_fft=64, hop_length=32, win_length=64):
        super().__init__()
        self.channels = channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        # Hann 窗口
        self.register_buffer('hann', torch.hann_window(win_length))
        # 用于高频滤波（Laplacian 核）
        lap = torch.tensor([[0., -1., 0.],
                            [-1., 4., -1.],
                            [0., -1., 0.]]).view(1,1,3,3)
        self.register_buffer('laplacian', lap)

        # 2D CNN：3C → C
        self.tcn = nn.Sequential(
            nn.Conv2d(4*channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(channels), nn.ELU(),nn.Dropout(0.25),
            nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(channels), nn.ELU(), nn.Dropout(0.25)
        )

    def forward(self, x):
        B, C, T = x.shape  # C == model_dim
        # 1) STFT 并取幅值谱
        x_flat = x.reshape(B*C, T)
        spec = torch.stft(x_flat,
                          n_fft=self.n_fft,
                          hop_length=self.hop_length,
                          win_length=self.win_length,
                          window=self.hann,
                          return_complex=True)        # (B*C, F, T_f)
        mag = spec.abs()                   # (B*C, F, T_f)
        F_bin, T_f = mag.shape[-2], mag.shape[-1]
        mag = mag.view(B, C, F_bin, T_f)   # (B, C, F, T_f)

        # 2) 频率尖峰通道：频率轴差分
        # pad 前后各一行，使 shape 不变
        pad_freq = F.pad(mag, (0,0,1,1), mode='replicate')
        diff_freq = pad_freq[:,:,2:,:] - pad_freq[:,:,:-2,:]  # (B,C,F,T_f)

        # 3) 快变通道：时间轴差分
        pad_time = F.pad(mag, (1,1,0,0), mode='replicate')
        diff_time = pad_time[:,:,:,2:] - pad_time[:,:,:,:-2]  # (B,C,F,T_f)

        # 4) 高频成分通道：Laplacian 高通滤波
        # 按通道卷积
        # laplacian: (1,1,3,3) -> (C,1,3,3)
        lap = self.laplacian.repeat(C,1,1,1)
        high_freq = F.conv2d(mag, weight=lap, padding=1, groups=C)  # (B,C,F,T_f)

        # 5) 当作“RGB”三通道，拼在一起： (B,4C,F,T_f)
        rgb = torch.cat([mag, diff_freq, diff_time, high_freq], dim=1)

        y = self.tcn(rgb)

        # 7) 沿频率维度池化 → (B, C, T_f)
        y = y.mean(dim=2)

        # 8) 插值对齐到原始窗口长度 T
        if T_f != T:
            y = F.interpolate(y, size=T, mode='linear', align_corners=False)

        return y  # (B, C, T)

class BandPassSimpleMA(nn.Module):
    def __init__(self, sr, bands=None, res_scale=0.3):
        super().__init__()
        self.sr = sr
        self.bands = bands if bands is not None else [(8.0,12.0),(12.0,30.0),(30.0,50.0)]
        self.res_scale = res_scale
        self.num_bands = len(self.bands)
        # 固定权重为1，可扩展为可学习
        self.register_buffer('band_w', torch.ones(self.num_bands))

    def _kernel_size(self, cutoff):
        k = max(3, int(round(self.sr / (cutoff + 1e-6))))
        return k + (k % 2 == 0)  # 保证为奇数

    def forward(self, x):
        # x: (B, C, T)
        bands_out = []
        for (f_low, f_high) in self.bands:
            k_high = self._kernel_size(f_high)
            k_low  = self._kernel_size(f_low)
            # 两次均值池化作为低通滤波
            lp_high = F.avg_pool1d(x, kernel_size=k_high, stride=1, padding=k_high//2)
            lp_low  = F.avg_pool1d(x, kernel_size=k_low,  stride=1, padding=k_low//2)
            # 带通信号 = 高截止低通 - 低截止低通
            band = lp_high - lp_low  # (B, C, T)
            bands_out.append(band)
        # 将各频段信号加权求和
        stacked = torch.stack(bands_out, dim=0)  # (num_bands, B, C, T)
        w = self.band_w.view(self.num_bands, 1, 1, 1)
        weighted = (stacked * w).sum(dim=0)     # (B, C, T)
        # 残差相加
        out = x + self.res_scale * weighted
        return out


class BandPassFreqDetailSimple(nn.Module):
    """
    Version 1 Simple: Directly combine BandPassSimpleMA and FrequencyDetailBranch.
    """
    def __init__(self, sr, bands=None, res_scale=0.3, channels=None,
                 n_fft=64, hop_length=32, win_length=64):
        super().__init__()
        self.sr = sr
        self.res_scale = res_scale
        self.bandpass = BandPassSimpleMA(sr, bands, res_scale)
        if channels is None:
            raise ValueError("channels must be specified for FrequencyDetailBranch")
        self.freq_detail = FrequencyDetailBranch(channels, n_fft, hop_length, win_length)
        self.detail_scale = res_scale

    def forward(self, x):
        # x: (B, C, T)
        band_out = self.bandpass(x)       # (B, C, T) includes residual x
        detail_out = self.freq_detail(band_out)  # (B, C, T)
        # Combine: add detail branch to band-enhanced output
        out = band_out + self.detail_scale * detail_out
        return out

class FeatureIntegrator(nn.Module):
    def __init__(self,sr, in_channels, out_channels, kernel_size=64, stride=64):
        super(FeatureIntegrator, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bandPass =  BandPassFreqDetailSimple(sr, res_scale=0.3, channels=out_channels)


    def forward(self, x):
        # 假设输入x的形状为 (batch_size, feature_dim, channels, length)
        batch_size, feature_dim, channels, length = x.size()

        # 你想将feature和length维度相结合
        # 首先，将x变形为 (batch_size, channels, feature_dim * length)
        x = x.reshape(batch_size, channels, feature_dim * length)

        # 然后，应用1D卷积
        x = self.conv(x)  # 卷积后的形状为 (batch_size, out_channels, new_length)

        x = self.bandPass(x)

        return x


########################################################################################

class PowerLayer(nn.Module):
    """
    The power layer: calculates the log-transformed power of the data
    """

    def __init__(self, dim, length, step):
        super(PowerLayer, self).__init__()
        self.dim = dim
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))

    def forward(self, x):
        return torch.log(self.pooling(x.pow(2)))

class Aggregator():

    def __init__(self, idx_area):
        # chan_in_area: a list of the number of channels within each area
        self.chan_in_area = idx_area
        self.idx = self.get_idx(idx_area)
        self.area = len(idx_area)

    def forward(self, x):
        # x: batch x channel x data
        data = []
        for i, area in enumerate(range(self.area)):
            if i < self.area - 1:
                data.append(self.aggr_fun(x[:, self.idx[i]:self.idx[i + 1], :], dim=1))
            else:
                data.append(self.aggr_fun(x[:, self.idx[i]:, :], dim=1))
        return torch.stack(data, dim=1)

    def get_idx(self, chan_in_area):
        idx = [0] + chan_in_area
        idx_ = [0]
        for i in idx:
            idx_.append(idx_[-1] + i)
        return idx_[1:]

    def aggr_fun(self, x, dim):
        # return torch.max(x, dim=dim).values
        return torch.mean(x, dim=dim)

if __name__ == '__main__':
    # 模拟数据：
    # 原始数据 shape 为 (20, 14, 1, 32, 800)
    # 20=batch_size，14=脑区数，1=频段数，32=通道数，800=时间点数
    data = torch.randn(20, 14, 1, 32, 800)
    label = torch.randint(0, 2, (20, 14))  # 示例标签
    print("原始数据 shape:", data.shape)
    print("标签 shape:", label.shape)

    # 选择处理第1个脑区的数据，得到 x 的 shape: (20, 1, 32, 800)
    x = data[:, 0, :, :, :]
    print("选择第1个脑区后 x 的 shape:", x.shape)

    # 定义模型的输入尺寸：(frequency, channels, data_points)
    input_size = (x.shape[1], x.shape[2], x.shape[3])
    print("模型输入尺寸:", input_size)

    # 定义其他模型参数
    num_classes = 2  # 设置分类类别数为2（二分类任务）
    sampling_rate = 200  # 采样率200
    num_T = 64  # 时间特征维度
    out_graph = 32  # 动态图卷积输出维度
    dropout_rate = 0.5
    pool = 16  # Temporal learner 中的池化参数
    pool_step_rate = 0.25
    temporal_kernel = 20
    num_head = 8

    idx_graph = [2, 2, 2, 2, 1, 2, 2, 3, 4, 5, 2, 3, 1, 1]
    print("重新设计后的 idx_graph:", idx_graph)

    # 将模型与数据移动到 DEVICE 上（例如 'cuda' 或 'cpu'）
    model = ATDGNN(num_classes=num_classes, input_size=input_size, sampling_rate=sampling_rate,
                   num_T=num_T, out_graph=out_graph, dropout_rate=dropout_rate,
                   pool=pool, pool_step_rate=pool_step_rate, idx_graph=idx_graph, temporal_kernel=temporal_kernel,
                   layers_transformer=2, num_head = num_head, layers_len = 286)
    model.to(DEVICE)
    x = x.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量：{total_params}")#总参数量：5160980
                                    #总参数量：2694142
                                    #总参数量：5147188

    # 为便于调试，定义一个 hook 用于打印各模块的输入和输出 shape
    def print_hook(module, input, output):
        in_shapes = [inp.shape for inp in input if isinstance(inp, torch.Tensor)]
        if isinstance(output, (list, tuple)):
            out_shapes = [out.shape for out in output if isinstance(out, torch.Tensor)]
        elif isinstance(output, torch.Tensor):
            out_shapes = output.shape
        else:
            out_shapes = None
        print(f"{module.__class__.__name__} | input shape: {in_shapes} --> output shape: {out_shapes}")

    # 注册 hook 到部分关键模块
    hooks = []
    modules_to_hook = [
        model.Tception,
        model.feature_integrator,
        model.fc
    ]
    for mod in modules_to_hook:
        hooks.append(mod.register_forward_hook(print_hook))

    # 运行前向传播，并在 ATDGNN 内部打印关键节点输出
    output = model(x)
    print("最终输出 shape:", output.shape)

    # 取消 hook
    for hook in hooks:
        hook.remove()
