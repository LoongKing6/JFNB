import os
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from config.config import *
from einops.layers.torch import Rearrange
import torch.nn.init as init
from torch_geometric.nn import SGConv, global_add_pool
from torch_geometric.nn import HeteroConv, GATConv, GCNConv, SAGEConv
from torch_geometric.utils import trim_to_layer, dense_to_sparse
import scipy.signal
############################################################
from einops.layers.torch import Rearrange
from einops import rearrange
#########################################################
from torch.nn.utils import weight_norm
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math

_, os.environ['CUDA_VISIBLE_DEVICES'] = set_config()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# DeepConvNet
# [Deep learning with convolutional neural networks for EEG decoding and visualization]
# (https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
class DeepConvNet(nn.Module):
    def convBlock(self, inF, outF, droup_rate, kernalSize, *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=droup_rate), #丢弃层通过随机丢弃一部分神经元的输出，防止网络过度依赖特定的神经元，从而减少过拟合，提高模型的泛化能力，其中droup_rate是丢弃比例。
            Conv2dWithConstraint(inF, outF, kernalSize, bias=False, max_norm=2, *args, **kwargs), #卷积层，其中inF输入通道数，outF输出通道数，kernalSize卷积核大小，max_norm=2最大范数约束（范数约束的作用：控制网络权重的大小，从而提高模型的泛化能力，减少过拟合的风险）
            nn.BatchNorm2d(outF), #批量归一化层，用于加速训练，减少内部变量偏移。
            nn.ELU(), #激活函数，增加非线性
            nn.MaxPool2d((1, 3), stride=(1, 3)) #池化层，用于减少空间维度。其中，池化核大小为 (1, 3)，stride步长为（1，3）
        )

    def firstBlock(self, outF, droup_rate, kernalSize, channels, *args, **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(1, outF, kernalSize, padding=0, max_norm=2, *args, **kwargs), # 1 与 outF表示将输入的 1 个通道的数据通过卷积层的卷积变成 outF 个通道的数据
            Conv2dWithConstraint(25, 25, (channels, 1), padding=0, bias=False, max_norm=2), # 第二卷积层进一步提取特征，其中根据EEG通道的维度设置卷积核的大小为(channels, 1)
            nn.BatchNorm2d(outF), #后面的三行代码的就是常见的批量归一化、激活和池化层。
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=(1, 3))
        )

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(inF, outF, kernalSize, max_norm=0.5, *args, **kwargs)) #卷积层将输入的特征inF 映射到 最终的输出通道outF，从而得到最终结果。

    def calculateOutSize(self, model, channels, nTime):
        """
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        """
        data = torch.rand(1, 1, channels, nTime) #输入大小为(1, 1, channels, nTime)的数据
        model.eval() #后面三行根据数据大小计算特征图的大小，从而可以得到要准备的卷积层的大小（因为卷积层要处理特征图）
        out = model(data).shape
        return out[2:]

    def __init__(self, channels, nTime, n_classes=2, droup_rate=0.25, *args, **kwargs):
        super(DeepConvNet, self).__init__()

        kernalSize = (1, 5)  # Please note that the kernel size in the origianl paper is (1, 10), we found when the segment length is shorter than 4s (1s, 2s, 3s) larger kernel size will
        # cause network error. Besides using (1, 5) when EEG segment is 4s gives slightly higher ACC and F1 with a smaller model size.
        nFilt_FirstLayer = 25
        nFiltLaterLayer = [25, 50, 100, 200]

        firstLayer = self.firstBlock(nFilt_FirstLayer, droup_rate, kernalSize, channels) #构建第一个卷积块进行初步卷积的特征提取，得到低层次特征
        middleLayers = nn.Sequential(*[self.convBlock(inF, outF, droup_rate, kernalSize) # 通过多个模块进行特征抽象和维度压缩，即将低层次特征变成高层次特征，得到高层次特征，并且去除不重要和重复的信息，从而减少计算量。
                                       for inF, outF in zip(nFiltLaterLayer, nFiltLaterLayer[1:])])

        self.allButLastLayers = nn.Sequential(firstLayer, middleLayers)

        self.fSize = self.calculateOutSize(self.allButLastLayers, channels, nTime) #计算特征图大小
        self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], n_classes, (1, self.fSize[1])) #根据计算出特征图大小，用卷积块将特征映射到最终通道，从而得到最终结果。

    def forward(self, x):
        x = self.allButLastLayers(x) #先通过除了最后一个卷积层的所有卷积层
        x = self.lastLayer(x) #通过最后一个卷积层
        x = torch.squeeze(x, 3) #最后两行是移除输出中的多余维度
        x = torch.squeeze(x, 2)

        return x


# ShallowConvNet
# [Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization]
# (https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730)
class ShallowConvNet(nn.Module):
    def convBlock(self, inF, outF, droup_rate, kernalSize, *args, **kwargs):
        return nn.Sequential( #与DeepConvNet的convBlock函数类似
            nn.Dropout(p=droup_rate),
            Conv2dWithConstraint(inF, outF, kernalSize, bias=False, max_norm=2, *args, **kwargs),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=(1, 3))
        )

    def firstBlock(self, outF, droup_rate, kernalSize, channels, *args, **kwargs):
        return nn.Sequential( #与DeepConvNet的firstBlock函数类似
            Conv2dWithConstraint(1, outF, kernalSize, padding=0, max_norm=2, *args, **kwargs),
            Conv2dWithConstraint(40, 40, (channels, 1), padding=0, bias=False, max_norm=2),
            nn.BatchNorm2d(outF),
        )

    def calculateOutSize(self, channels, nTime):
        """
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        """
        data = torch.rand(1, 1, channels, nTime) #输入数据
        block_one = self.firstLayer #对数据进行卷积
        avg = self.avgpool #定义池化层
        dp = self.dp  #定义丢弃层
        out = torch.log(block_one(data).pow(2)) #后面都是为了求特征图大小，并进行池化和丢弃操作
        out = avg(out) #对数据进行池化
        out = dp(out) #对数据进行丢弃操作
        out = out.view(out.size()[0], -1)
        return out.size()

    def __init__(self, channels, nTime, n_classes=2, droup_rate=0.25, *args, **kwargs):
        super(ShallowConvNet, self).__init__()

        kernalSize = (1, 25)
        nFilt_FirstLayer = 40

        self.firstLayer = self.firstBlock(nFilt_FirstLayer, droup_rate, kernalSize, channels) #使用firstBlock构建第一个卷积块
        self.avgpool = nn.AvgPool2d((1, 75), stride=(1, 15)) #定义池化层，其中池化核大小为(1, 75)，步长为(1, 15)
        self.dp = nn.Dropout(p=droup_rate) #定义丢弃层，防止过拟合，丢弃率droup_rate
        self.fSize = self.calculateOutSize(channels, nTime) #计算特征图大小
        self.lastLayer = nn.Linear(self.fSize[-1], n_classes) #定义全连接层，将特征映射到最终类标签，得到结果

    def forward(self, x): #ShallowConvNet的结构中，数据从输入到输出的全过程
        x = self.firstLayer(x) #第一个卷积块提取特征
        x = torch.log(self.avgpool(x.pow(2))) #对特征进行平方后取对数增强对信号变化的敏感度，并通过池化降低维度
        x = self.dp(x) #丢弃层防止过拟合
        x = x.view(x.size()[0], -1) #输出特征展平，为全连接层做准备
        x = self.lastLayer(x) #全连接层生成分类标签结果

        return x


class EEGNetModule(nn.Module):
    def __init__(self, channels, F1, D, kernLength, dropout, input_size):
        super(EEGNetModule, self).__init__()

        #第一步：初始化EEGNetModule参数
        self.F1 = F1 #第一个卷积层的输出通道数
        self.D = D # F1的系数
        self.F2 = D * F1 #第二个卷积层的输出通道数
        self.T = input_size[2] #输入数据的时间维度（数据尺寸）
        self.kernLength = int(kernLength) #卷积核大小

        # 第二步：进入第一个卷积层和深度卷积提取初步特征
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(channels, self.kernLength), groups=1,
                               padding='same', bias=False) # 第一个卷积层，其中输入通道数为1，输出通道数为F1，卷积核大小为(channels：EEG信号通道数, self.kernLength：卷积核长度)，padding='same'保证输出特征图大小与输入相同
        self.batchnorm1 = nn.BatchNorm2d(num_features=F1) #归一化，加速训练，并稳定网络

        self.depthwiseConv = nn.Conv2d(in_channels=F1, out_channels=self.F2, kernel_size=(channels, 1), groups=F1,
                                       bias=False)# 深度卷积层，将输入的F1的每个通道进行卷积，从而输出F2通道，卷积核大小(channels, 1)
        self.batchnorm2 = nn.BatchNorm2d(num_features=self.F2) #归一化
        self.activation = nn.ELU() #激活函数
        self.avg_pool1 = nn.AvgPool2d((1, 4)) #池化层，池化核大小为(1, 4)
        self.dropout1 = nn.Dropout(dropout) #丢弃层，防止过拟合

        # 第三步：进入可分离卷积和第二个卷积层提取更高层次的特征
        self.separableConv = nn.Conv2d(in_channels=self.F2, out_channels=self.F2, kernel_size=(1, 16), padding='same',
                                       bias=False, groups=self.F2) #第一个卷积层，其中卷积核大小为(1, 16)，这一层使用了深度可分离卷积，每个通道单独进行卷积操作，从而减少计算量
        self.conv2 = nn.Conv2d(in_channels=self.F2, out_channels=self.F2, kernel_size=1, bias=False) #第二个卷积层，用于进一步提取特征
        self.batchnorm3 = nn.BatchNorm2d(num_features=self.F2)
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x): # EEGNetModule 的结构中，数据从输入到输出的全过程
        x = self.batchnorm1(self.conv1(x)) #输入数据进行第一个卷积层conv1(x)，并通过batchnorm1进行归一化
        x = self.activation(self.batchnorm2(self.depthwiseConv(x))) #进行depthwiseConv深度卷积和batchnorm2归一化后，使用activation激活函数引入非线性
        x = self.dropout1(self.avg_pool1(x)) #进入avg_pool1池化操作后，进入dropout1丢弃层防止过拟合
        x = self.separableConv(x) #进入可分离层
        x = self.conv2(x) #进入第二个卷积层
        x = self.activation(self.batchnorm3(x)) #归一化后进入激活函数
        x = self.dropout2(self.avg_pool2(x)) #池化后进入丢弃层
        return x


# EEGNet
# [EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces]
# (https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta?casa_token=lv6qPlB_YWgAAAAA:9c1FVN1Co6ae3vT6bjTh4VctC1sJLQPbv7uES2QtElX6JoAD2ICg4tndyvhaciMRSch51He_CszifyM0v1ZjBgp51WIW)
class EEGNet(nn.Module):
    def __init__(self, n_classes, input_size, sampling_rate, F1=8, kernLength=64, D=2, channels=32, dropout=0.25):
        super(EEGNet, self).__init__()

        #第一步：EEGNet初始化参数
        self.F1 = F1 #第一个卷积层的输出通道数
        self.D = D
        self.F2 = D * F1
        self.T = input_size[2]
        self.sampling_rate = sampling_rate #采样率，影响信号的时间分辨率
        self.kernLength = int(kernLength)

        #第二步： EEG数据处理模块，用于提取信号的低层次和高层次特征
        self.EEGNet_sep = EEGNetModule(channels=channels, F1=F1, D=D, kernLength=kernLength, dropout=dropout,
                                       input_size=input_size) #对上一个模块EEGNetModule进行实例化

        #第三步： 分类器
        self.flatten = nn.Flatten() #将多维特征展平为一维特征，为后面的全连接层做准备
        self.dense = nn.Linear(self.F2 * math.ceil(self.T / 32), n_classes)  # 全连接层，将展平后的数据输入到全连接层进行分类。这里T/32是因为有两次池化，每次都是T减少4倍。

    def forward(self, x):
        x = self.EEGNet_sep(x) #将数据传入EEGNetModule进行特征提取
        # 分类器
        x = self.flatten(x) #将提取到的多维数据展平为一维数据，准备进入全连接层进行分类
        x = self.dense(x) #进入全连接层进行分类，输出最终预测结果
        return F.log_softmax(x, dim=1) #对预测结果进行log-softmax操作，使结果符合概率分布


nonlinearity_dict = dict(relu=nn.ReLU(), elu=nn.ELU()) #这是一个字典，用于存储常用的激活函数


class CausalConv1d(nn.Conv1d):

    #初始化卷积层参数
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1,
                 groups=1, bias=True):
        super(CausalConv1d, self).__init__(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, bias=bias)  #stride卷积操作的步长。dilation膨胀卷积的膨胀率，通过增加卷积核内元素的间隔来扩大感受野的卷积操作，通过设置膨胀率来扩大间隔。groups分组卷积的分组数量。bias：是否使用偏置项（作用就像是一个“常数项”，用来调整模型的预测结果）
        #注：感受野：衡量特征提取范围的指标，描述了网络中某个特征在输入数据中能覆盖的区域。通过增加卷积层的深度、使用更大的卷积核或膨胀卷积等方式，可以有效增加感受野，从而让网络捕捉到更大的上下文信息。
        self.__padding = (kernel_size - 1) * dilation #计算因果卷积所需的填充量，填充的作用：保证卷积只使用当前的数据，不会使用未来的数据。


    # 进行因果卷积操作，在卷积的同时填充数据，保证卷积只使用当前的数据，不会使用未来的数据。
    def forward(self, x):
        return super(CausalConv1d, self).forward(F.pad(x, (self.__padding, 0))) #F.pad(x, (self.__padding, 0)是对其进行左填充。super(CausalConv1d, self).forward调用父类nn.Conv1d进行卷积操作，并且确保执行过程中进行填充操作，即只使用当前的数据，不会使用未来的数据。
    #注：nn.Conv1d 是 PyTorch 中的一个标准的一维卷积层（1D Convolution），用于处理一维序列数据，CausalConv1d(nn.Conv1d)表示CausalConv1d继承了父类nn.Conv1d

class Conv2dWithConstraint(nn.Conv2d): #Conv2dWithConstraint 是一个自定义的卷积层类，继承自 PyTorch 的 nn.Conv2d 类
    def __init__(self, *args, max_norm=None, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear): #LinearWithConstraint 是一个自定义的线性层类，继承自 PyTorch 的 nn.Linear 类
    def __init__(self, *args, max_norm=None, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm( #使用torch.renorm对卷积核的权重进行调整，p =2表示使用L2范数进行约束，dim=0表示对第一维进行约束，maxnorm=self.max_norm表示最大范数。
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class _TCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float, activation: str = "relu"):
        super(_TCNBlock, self).__init__()

        #定义第一个因果卷积层，批量归一化，激活函数，丢弃层
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size,
                                  dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels, momentum=0.01, eps=0.001)
        self.nonlinearity1 = nonlinearity_dict[activation]
        self.drop1 = nn.Dropout(dropout)

        #定义第二个因果卷积层，批量归一化，激活函数，丢弃层
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size,
                                  dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=0.01, eps=0.001)
        self.nonlinearity2 = nonlinearity_dict[activation]
        self.drop2 = nn.Dropout(dropout)

        #残差连接与通道匹配，这个操作是为了实现 残差连接（Residual Connection），允许信息从网络的前层直接传递到后续层
        if in_channels != out_channels:
            self.project_channels = nn.Conv1d(in_channels, out_channels, 1) #如果输入通道不等于输出通道，则使用Conv1d卷积层将输入通道映射到输出通道（使输入数据的维度与输出数据的维度匹配）
        else:
            self.project_channels = nn.Identity() #如果输入与输出通道相等，则使用nn.Identity，即不做任何修改

        #激活函数进行最终的非线性激活
        self.final_nonlinearity = nonlinearity_dict[activation]

    def forward(self, x):
        residual = self.project_channels(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nonlinearity1(out)
        out = self.drop1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nonlinearity2(out)
        out = self.drop2(out)
        return self.final_nonlinearity(out + residual)


class EEGTCNet(nn.Module):

    #第一步： 构造函数__init__函数，初始化EEGTCNet模型，定义模型的结构，包括卷积层、池化层、全连接层等。
    def __init__(self, n_classes: int, in_channels: int = 32, layers: int = 2, kernel_s: int = 4, filt: int = 12,
                 dropout: float = 0.3, activation: str = 'relu', F1: int = 8, D: int = 2, kernLength: int = 32,
                 dropout_eeg: float = 0.2
                 ): #n_classes分类的类别数量，layers卷积层的层数（多少个卷积核），filt滤波器数量，F1输出通道数，D为卷积系数（决定第二个通道数 F2 = F1 * D），kernLength卷积核的长度
        super(EEGTCNet, self).__init__()
        regRate = 0.25
        numFilters = F1
        F2 = numFilters * D

    #第二步：定义EEGNet 模块
        self.eegnet = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding="same", bias=False), #第一个卷积层提取低层次特征
            nn.BatchNorm2d(F1, momentum=0.01, eps=0.001), #批量归一化
            Conv2dWithConstraint(F1, F2, (in_channels, 1), bias=False, groups=F1, #定义了之前的 Conv2dWithConstraint模型，即带有最大范数约束的卷积层，避免过拟合
                                 max_norm=1),
            nn.BatchNorm2d(F2, momentum=0.01, eps=0.001), #批量归一化
            nn.ELU(), #激活函数
            nn.AvgPool2d((1, 8)), #池化层
            nn.Dropout(dropout_eeg),
            nn.Conv2d(F2, F2, (1, 16), padding="same", groups=F2, bias=False), #第二个卷积层为深度卷积层，用于在每个通道中将第一层的低层次特征转换成高层次特征，得到时间序列特征
            nn.Conv2d(F2, F2, 1, bias=False), #第三个卷积层，用于将深度卷积层中的每个通道内的特征融合在一起，从而实现通道之间的信息交互，最后得到一个时间序列特征
            nn.BatchNorm2d(F2, momentum=0.01, eps=0.001),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_eeg),
            Rearrange("b c 1 t -> b c t") #改变数据的形状与维度，目的：因为不需要 1 这个维度，所以移除，保留 b c t，从而调整数据形状与维度（从4维变成3维），方便后续作为数据进行输入。
        )

    #第三步：定义 tcn_blocks 模块
        in_channels = [F2] + (layers - 1) * [filt] #定义一个列表，用于存储每个卷积层的输入通道数，第一个卷积核的输入通道数为F2，后面的卷积核的输入通道数都为filt
        dilations = [2 ** i for i in range(layers)] #定义一个列表，用于存储每个卷积层的膨胀率，用于控制每层卷积的感受野。
        self.tcn_blocks = nn.ModuleList([
            _TCNBlock(in_ch, filt, kernel_size=kernel_s, dilation=dilation,
                      dropout=dropout, activation=activation) # 定义之前的_TCNBlock模块，对时间序列数据进行卷积和特征提取
            for in_ch, dilation in zip(in_channels, dilations)
        ])

        self.classifier = LinearWithConstraint(filt, n_classes, max_norm=regRate) #全连接层，将_TCNBlock模块提取的特征映射到分类空间
        # 初始化函数
        self.initialize_weights()

    #第四步：定义初始化函数
    def initialize_weights(self):
        for module in self.modules():
            if hasattr(module, "weight") and module.weight is not None:
                if "norm" not in module.__class__.__name__.lower():
                    init.xavier_uniform_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                init.constant_(module.bias, 0) #将偏置初始化为0

    def forward(self, x):
        x = self.eegnet(x) #输入数据，通过EEGNet模块提取空间和时间特征
        for blk in self.tcn_blocks: # 数据通过多个TCN层进行处理（x = blk(x)就是表示将当前数据x传入TCN模块进行处理，所以循环多次就是进行多层处理），每一层使用_TCNBlock模块进一步提取时序特征
            x = blk(x)
        x = self.classifier(x[:, :, -1]) #在全连接层，通过分类器LinearWithConstraint进行分类
        return x


# TCNet_Fusion
# [Electroencephalography-based motor imagery classification using temporal convolutional network fusion]
# (https://www.sciencedirect.com/science/article/abs/pii/S1746809421004237)
class TCNet_Fusion(nn.Module):
    def __init__(self, input_size, n_classes, channels, sampling_rate, kernel_s=3,
                 dropout=0.3, F1=24, D=2, dropout_eeg=0.3, layers=1, filt=12, activation='elu'):
        super(TCNet_Fusion, self).__init__()
        self.kernLength = int(0.25 * sampling_rate)
        F2 = F1 * D
        self.n_classes = n_classes

        self.EEGNet_sep = EEGNetModule(channels=channels, F1=F1, D=D, kernLength=self.kernLength, dropout=dropout_eeg,
                                       input_size=input_size)

        in_channels = [F2] + (layers - 1) * [filt]
        dilations = [2 ** i for i in range(layers)]
        self.tcn_blocks = nn.ModuleList([
            _TCNBlock(in_ch, filt, kernel_size=kernel_s, dilation=dilation,
                      dropout=dropout, activation=activation)
            for in_ch, dilation in zip(in_channels, dilations)
        ])
        size = self.get_size_temporal(input_size)
        self.dense = nn.Linear(size[-1], n_classes)
        self.softmax = nn.Softmax(dim=1)

    def get_size_temporal(self, input_size):
        data = torch.randn((1, input_size[0], input_size[1], input_size[2]))
        x = self.EEGNet_sep(data)
        eeg_output = torch.squeeze(x, 2)
        for blk in self.tcn_blocks:
            tcn_output = blk(eeg_output)
        con1_output = torch.cat((eeg_output, tcn_output), dim=1)  # 沿特征维度拼接
        fc1_output = torch.flatten(eeg_output, start_dim=1)
        fc2_output = torch.flatten(con1_output, start_dim=1)
        # 再次Concatenation
        con2_output = torch.cat((fc1_output, fc2_output), dim=1)
        size = con2_output.size()
        return size

    def forward(self, x):
        x = self.EEGNet_sep(x)
        eeg_output = torch.squeeze(x, 2)
        for blk in self.tcn_blocks:
            tcn_output = blk(eeg_output)
        con1_output = torch.cat((eeg_output, tcn_output), dim=1)  # 沿特征维度拼接
        fc1_output = torch.flatten(eeg_output, start_dim=1)
        fc2_output = torch.flatten(con1_output, start_dim=1)
        # 再次Concatenation
        con2_output = torch.cat((fc1_output, fc2_output), dim=1)
        # Dense and Softmax
        dense_output = self.dense(con2_output)
        output = self.softmax(dense_output)
        return output


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


# LGGNet
# [LGGNet: Learning from local-global-graph representations for brain–computer interface]
# (https://ieeexplore.ieee.org/abstract/document/10025569)
class LGGNet(nn.Module):
    def temporal_learner(self, in_chan, out_chan, kernel, pool, pool_step_rate):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
            PowerLayer(dim=-1, length=pool, step=int(pool_step_rate * pool))
        )

    def __init__(self, num_classes, input_size, sampling_rate, num_T,
                 out_graph, dropout_rate, pool, pool_step_rate, idx_graph):
        # input_size: EEG frequency x channel x datapoint
        super(LGGNet, self).__init__()
        self.idx = idx_graph
        self.window = [0.5, 0.25, 0.125]
        self.pool = pool
        self.channel = input_size[1]
        self.brain_area = len(self.idx)

        # by setting the convolutional kernel being (1,lenght) and the strids being 1, we can use conv2d to
        # achieve the 1d convolution operation.
        self.Tception1 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[0] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception2 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[1] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception3 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[2] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_T)
        self.OneXOneConv = nn.Sequential(
            nn.Conv2d(num_T, num_T, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, 2))
        )
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

        # trainable adj weight for global network
        # 表示全局邻接矩阵。它被定义为浮点型张量，并设置为需要梯度计算（requires_grad=True）
        self.global_adj = nn.Parameter(torch.FloatTensor(self.brain_area, self.brain_area), requires_grad=True)
        # 根据给定的张量的形状和分布进行参数初始化。用来对global_adj进行初始化，采用的是Xavier均匀分布初始化方法。
        nn.init.xavier_uniform_(self.global_adj)
        # to be used after local graph embedding
        self.bn = nn.BatchNorm1d(self.brain_area)
        self.bn_ = nn.BatchNorm1d(self.brain_area)
        # learn the global network of networks
        self.GCN = GraphConvolution(size[-1], out_graph)

        self.fc = nn.Sequential(  # 组合神经网络模块
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(self.brain_area * out_graph), num_classes)
        )

    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        out = self.OneXOneConv(out)
        out = self.BN_s(out)
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        out = self.local_filter_fun(out, self.local_filter_weight)
        out = self.aggregate.forward(out)
        adj = self.get_adj(out)
        out = self.bn(out)
        out = self.GCN(out, adj)
        out = self.bn_(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

    def get_size_temporal(self, input_size):
        # input_size: frequency x channel x data point
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        z = self.Tception1(data)
        out = z
        z = self.Tception2(data)
        out = torch.cat((out, z), dim=-1)
        z = self.Tception3(data)
        out = torch.cat((out, z), dim=-1)
        out = self.BN_t(out)
        out = self.OneXOneConv(out)
        out = self.BN_s(out)
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        size = out.size()
        return size

    # 定义局部滤波器的前向传播函数
    def local_filter_fun(self, x, w):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def get_adj(self, x, self_loop=True):
        """
        x：输入的特征矩阵，大小为(b, node, feature)，其中b为批次大小，node为节点数目，feature为每个节点的特征向量维度。
        self_loop：一个布尔值，表示是否在邻接矩阵中加入自环（自己到自己的连接）。
        """
        # x: b, node, feature
        # 利用模型中的self_similarity方法计算输入特征矩阵x的自相似度矩阵。结果为一个大小为(b, n, n)的张量，其中n为节点数目
        adj = self.self_similarity(x)  # b, n, n
        num_nodes = adj.shape[-1]
        # 将自相似度矩阵与全局邻接矩阵的和进行逐元素相乘，并经过ReLU激活函数处理。这一步可以用来控制邻接矩阵中的连接关系
        adj = F.relu(adj * (self.global_adj + self.global_adj.transpose(1, 0)))
        if self_loop:
            # 在邻接矩阵中添加自环，即在对角线上设置为1，表示每个节点与自己相连接
            adj = adj + torch.eye(num_nodes).to(DEVICE)
        # 计算邻接矩阵每一行的元素之和，得到一个大小为(b, n)的张量
        rowsum = torch.sum(adj, dim=-1)
        # 创建一个与rowsum大小相同的全零张量mask，并将rowsum中和为0的位置置为1。这一步是为了处理邻接矩阵中存在度为0的节点，避免除以0的错误
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        # 将mask添加到rowsum中，实现对邻接矩阵的修正。避免除以0的错误，并保证每个节点的度至少为1
        rowsum += mask
        # 计算度矩阵的逆平方根
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        # 将逆平方根得到的张量转换为对角矩阵
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        # 通过矩阵乘法和广播机制，将度矩阵的逆平方根与邻接矩阵相乘，得到归一化后的邻接矩阵
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj

    def self_similarity(self, x):
        # x: b, node, feature
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)
        return s

    def compute_l2_regularization(self):
        l2_reg = torch.tensor(0.0).to(DEVICE)
        for param in self.parameters():
            l2_reg += torch.norm(param, p=2)  # 计算每个可优化参数的L2范数
        return l2_reg

    def loss_fn(self, logits, labels):
        l2_reg = self.compute_l2_regularization()  # 计算L2正则化项
        loss = F.cross_entropy(logits, labels) + self.weight_decay * l2_reg  # 将L2正则化项添加到损失函数中
        return loss


"""TSceptionIJCNN"""

class TSception(nn.Module):
    """
    Y. Ding et al., "TSception:A Deep Learning Framework for Emotion Detection Using EEG,"
    2020 International Joint Conference on Neural Networks (IJCNN), Glasgow, UK, 2020, pp. 1-7,
    doi: 10.1109/IJCNN48605.2020.9206750.
    """
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                      kernel_size=kernel, stride=step, padding=0),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        # input_size: 1 x EEG channel x datapoint
        super(TSception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self.conv_block(1, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self.conv_block(1, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)

        self.Sception1 = self.conv_block(num_T, num_S, (int(input_size[-2]), 1), 1, int(self.pool*0.25))
        self.Sception2 = self.conv_block(num_T, num_S, (int(input_size[-2] * 0.5), 1), (int(input_size[-2] * 0.5), 1),
                                         int(self.pool*0.25))
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)

        size = self.get_size(input_size)
        self.fc = nn.Sequential(
            nn.Linear(size[1], hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

    def get_size(self, input_size):
        # here we use an array with the shape being
        # (1(mini-batch),1(convolutional channel),EEG channel,time data point)
        # to simulate the input data and get the output size
        data = torch.ones((1, 1, input_size[-2], int(input_size[-1])))
        y = self.Tception1(data)
        out = y
        y = self.Tception2(data)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(data)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_final = z
        z = self.Sception2(out)
        out_final = torch.cat((out_final, z), dim=2)
        out = self.BN_s(out_final)
        out = out.view(out.size()[0], -1)
        return out.size()


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


class GraphConvolution(nn.Module):
    """
    Simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
        if bias:
            self.bias = nn.Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj):
        output = torch.matmul(x, self.weight) - self.bias
        output = F.relu(torch.matmul(adj, output))
        return output


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, depth_multiplier, kernel_size, bias=False):
        super(DepthwiseConv2d, self).__init__()
        self.depth_multiplier = depth_multiplier
        # depth_multiplier决定了每个输入通道应该扩展到多少个输出通道
        self.depthwise = nn.Conv2d(
            in_channels, in_channels * depth_multiplier, kernel_size=kernel_size,
            groups=in_channels, bias=bias, padding=0  # 设置padding为0来减少高度
        )

    def forward(self, x):
        return self.depthwise(x)


class ATC_Conv(nn.Module):
    def __init__(self, n_channel, in_channels, F1, D, KC, P2, dropout=0.3):
        super(ATC_Conv, self).__init__()
        F2 = F1 * D  # Output dimension
        # 第一层常规卷积: 时间卷积层
        self.temporal_conv = nn.Conv2d(in_channels, F1, (1, KC), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.elu = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)
        # 第二层卷积: 深度卷积层
        self.depthwise_conv = DepthwiseConv2d(in_channels=F1, depth_multiplier=D, kernel_size=(n_channel, 1))
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.avgpool1 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)
        # 第三层卷积
        self.spatial_conv = nn.Conv2d(F1 * D, F2, (1, KC), padding='same', bias=False)  # 修改核的尺寸以适配 KC
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, P2))  # 池化尺寸由P2控制
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x):
        # 时间卷积
        x = self.temporal_conv(x)
        x = self.batchnorm1(x)
        x = self.elu(x)
        x = self.dropout1(x)
        # 深度卷积
        x = self.depthwise_conv(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout2(x)
        # 空间卷积
        x = self.spatial_conv(x)
        x = self.batchnorm3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout3(x)
        return x


# ATCNet
# [Physics-informed attention temporal convolutional network for EEG-based motor imagery classification]
# (https://ieeexplore.ieee.org/abstract/document/9852687/)
class ATCNet(nn.Module):
    def __init__(self, input_size, n_channel, n_classes, n_windows=8,
                 eegn_F1=24, eegn_D=2, eegn_kernelSize=50, eegn_poolSize=8, eegn_dropout=0.3, num_heads=2,
                 tcn_depth=2, tcn_kernelSize=4, tcn_filters=32, tcn_dropout=0.3, fuse='average',
                 activation='elu'):
        super(ATCNet, self).__init__()
        self.n_windows = n_windows
        self.conv_block = ATC_Conv(n_channel, 1, eegn_F1, eegn_D, eegn_kernelSize, eegn_poolSize, eegn_dropout)
        self.fuse = fuse

        in_channels = [eegn_F1 * eegn_D] + (tcn_depth - 1) * [tcn_filters]
        dilations = [2 ** i for i in range(tcn_depth)]

        self.attention_block = MultiHeadSelfAttention(eegn_F1 * eegn_D, num_heads)

        self.tcn_blocks = nn.ModuleList([
            _TCNBlock(in_ch, tcn_filters, kernel_size=tcn_kernelSize, dilation=dilation,
                      dropout=tcn_dropout, activation=activation)
            for in_ch, dilation in zip(in_channels, dilations)
        ])
        self.fuse_layer = nn.Linear(tcn_filters, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):  # (64,1,32,800)
        x = self.conv_block(x)  # (64,100,1,12)
        x = torch.flatten(x, start_dim=2)  # Flatten the channel and height dimensions (64,100,12)

        outputs = []
        for i in range(self.n_windows):
            windows_data = x[:, :, i:x.shape[2] - self.n_windows + i + 1]  # Sliding window
            # Attention block
            tcn_input = self.attention_block(windows_data)  # (batch_size, channels, T_w)
            for blk in self.tcn_blocks:
                tcn_output = blk(tcn_input)
                tcn_input = tcn_output
            # (64,32,5)
            tcn_output = tcn_output[:, :, -1]  # Last timestep
            outputs.append(tcn_output)
        # (64,32)
        if self.fuse == 'average':
            output = torch.mean(torch.stack(outputs, dim=1), dim=1)
        elif self.fuse == 'concat':
            output = torch.cat(outputs, dim=1)
        else:
            raise ValueError("Invalid fuse method")

        output = self.fuse_layer(output)
        output = self.softmax(output)
        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size, d, T_w = x.size()

        # Permute to (batch_size, T_w, d)
        x = x.permute(0, 2, 1)

        # Linear transformations
        Q = self.query_linear(x)  # (batch_size, T_w, d_model)
        K = self.key_linear(x)  # (batch_size, T_w, d_model)
        V = self.value_linear(x)  # (batch_size, T_w, d_model)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, T_w, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, T_w, d_k)
        K = K.view(batch_size, T_w, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, T_w, d_k)
        V = V.view(batch_size, T_w, self.num_heads, self.d_k).transpose(1, 2)  # (batch_size, num_heads, T_w, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # (batch_size, num_heads, T_w, T_w)
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, T_w, T_w)
        context = torch.matmul(attention_weights, V)  # (batch_size, num_heads, T_w, d_k)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, T_w, self.d_model)  # (batch_size, T_w, d_model)

        # Final linear transformation
        output = self.out_linear(context)  # (batch_size, T_w, d_model)
        output = self.layer_norm(output + x)  # Add & Norm

        # Permute back to (batch_size, d, T_w)
        output = output.permute(0, 2, 1)

        return output


class AdjacencyProcessor:
    def __init__(self, device=DEVICE):
        self.device = device

    def normalize_features(self, x):
        """
        对输入特征进行归一化处理
        x: (b, node, feature)
        返回: 归一化后的特征矩阵 (b, node, feature)
        """
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + 1e-6
        return (x - mean) / std

    def compute_similarity(self, x):
        """
        计算输入特征矩阵 x 的余弦相似度矩阵。
        x: (b, node, feature)
        返回: (b, node, node)
        """
        x = F.normalize(x, p=2, dim=2)  # 对每个特征向量进行L2归一化
        x_ = x.permute(0, 2, 1)  # 调整维度顺序为 (b, feature, node)
        s = torch.bmm(x, x_)  # 计算余弦相似度矩阵 (b, node, node)
        s = (s + 1) / 2  # 将相似度从 [-1, 1] 转换到 [0, 1]
        return s

    def normalize_adjacency_matrix(self, x):
        """
        归一化邻接矩阵。
        x：输入的特征矩阵，大小为(b, node, feature)。
        返回值：归一化后的邻接矩阵，大小为(b, node, node)。
        """
        # 计算自相似度矩阵
        adj = self.compute_similarity(x)  # (b, node, node)

        num_nodes = adj.shape[-1]
        # 加入自环
        adj = adj + torch.eye(num_nodes).to(DEVICE)
        # 计算度矩阵
        rowsum = torch.sum(adj, dim=-1)  # 计算度矩阵
        # 避免除以0
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        # 计算度矩阵的逆平方根
        d_inv_sqrt = torch.pow(rowsum, -0.5)  # (b, node)
        # 将逆平方根转换为对角矩阵
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)  # (b, node, node)
        # 归一化邻接矩阵
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)  # (b, node, node)
        return adj

    def process_batch(self, x_batch):
        """
        处理输入批次的特征矩阵 x_batch，返回相应的 edge_index 和 edge_weight。
        x_batch：大小为(b, node, feature)的输入特征矩阵。
        返回：
        - edge_index：大小为 (2, num_edges) 的张量。
        - edge_weight：大小为 (num_edges,) 的张量。
        """
        # 归一化邻接矩阵
        adj = self.normalize_adjacency_matrix(x_batch)  # (b, node, node)

        edge_index, edge_weight = dense_to_sparse(adj[0])

        return edge_index, edge_weight


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)


class Chebynet(nn.Module):
    def __init__(self, xdim, K, num_out):
        super(Chebynet, self).__init__()
        self.K = K
        self.gc1 = nn.ModuleList()  # https://zhuanlan.zhihu.com/p/75206669
        for i in range(K):
            self.gc1.append(GraphConvolution(xdim[2], num_out))

    def generate_cheby_adj(self, A, K, device):
        support = []
        for i in range(K):
            if i == 0:
                # support.append(torch.eye(A.shape[1]).cuda())  #torch.eye生成单位矩阵
                temp = torch.eye(A.shape[1])
                temp = temp.to(device)
                support.append(temp)
            elif i == 1:
                support.append(A)
            else:
                temp = torch.matmul(support[-1], A)
                support.append(temp)
        return support

    def forward(self, x, L):
        device = x.device
        adj = self.generate_cheby_adj(L, self.K, device)
        for i in range(len(self.gc1)):
            if i == 0:
                result = self.gc1[i](x, adj[i])
            else:
                result += self.gc1[i](x, adj[i])
        result = F.relu(result)
        return result


# DGCNN
# [EEG emotion recognition using dynamical graph convolutional neural networks]
# (https://ieeexplore.ieee.org/abstract/document/8320798)
class DGCNN(nn.Module):
    def __init__(self, input_size, batch_size, k_adj, num_out, nclass=2):
        super(DGCNN, self).__init__()
        self.batch_size = batch_size
        xdim = [batch_size] + [32, 5]
        self.K = k_adj
        self.layer1 = Chebynet(xdim, k_adj, num_out)
        self.BN1 = nn.BatchNorm1d(xdim[2])  # 对第二维（第一维为batch_size)进行标准化
        self.fc1 = Linear(xdim[1] * num_out, 32)
        self.fc2 = Linear(32, 8)
        self.fc3 = Linear(8, nclass)
        self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]).cuda())
        nn.init.xavier_normal_(self.A)

    def normalize_A(self, A, symmetry=False):
        A = F.relu(A)
        if symmetry:
            A = A + torch.transpose(A, 0, 1)  # A+ A的转置
            d = torch.sum(A, 1)  # 对A的第1维度求和
            d = 1 / torch.sqrt(d + 1e-10)  # d的-1/2次方
            D = torch.diag_embed(d)
            L = torch.matmul(torch.matmul(D, A), D)
        else:
            d = torch.sum(A, 1)
            d = 1 / torch.sqrt(d + 1e-10)
            D = torch.diag_embed(d)
            L = torch.matmul(torch.matmul(D, A), D)
        return L

    def forward(self, x):
        x = x.squeeze(1)
        x = self.BN1(x.transpose(1, 2)).transpose(1, 2)  # 因为第三维 才为特征维度
        L = self.normalize_A(self.A)  # A是自己设置的可训练参数及邻接矩阵
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = F.relu(self.fc1(result))
        result = F.relu(self.fc2(result))
        result = self.fc3(result)
        return result
######################################################################

# 全局超参数（可在 main 中覆盖）
POOLING_SIZE = 16   # MSCNet 中的池化大小
HEADS = 4           # Attention 中多头数量
DEPTH = 3           # Attention 堆叠深度

class CausalConv1d(nn.Conv1d):
    """1D 因果卷积，确保不使用未来信息"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = (self.kernel_size[0] - 1) * self.dilation[0]
        x = F.pad(x, (padding, 0))
        return super().forward(x)

class _TCNBlock(nn.Module):
    """Temporal Convolutional Network Block（带残差）"""
    def __init__(self, in_dim, depth, kernel_size, filters, drop_prob, activation=nn.ELU):
        super().__init__()
        self.activation = activation()
        self.downsample = (nn.Conv1d(in_dim, filters, 1, bias=False)
                           if in_dim != filters else None)

        self.layers = nn.ModuleList()
        for i in range(depth):
            dilation = 2 ** i
            convs = nn.Sequential(
                CausalConv1d(in_dim if i == 0 else filters, filters,
                             kernel_size, dilation=dilation, bias=False),
                nn.BatchNorm1d(filters),
                activation(),
                nn.Dropout(drop_prob),
                CausalConv1d(filters, filters,
                             kernel_size, dilation=dilation, bias=False),
                nn.BatchNorm1d(filters),
                activation(),
                nn.Dropout(drop_prob),
            )
            self.layers.append(convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, feature) -> (batch, feature, time)
        x = x.permute(0, 2, 1)
        res = x if self.downsample is None else self.downsample(x)
        out = x
        for layer in self.layers:
            y = layer(out)
            y = self.activation(y + res)
            res, out = y, y
        # back to (batch, time, feature)
        return out.permute(0, 2, 1)

class MSCNet(nn.Module):
    """Multi-Scale Convolutional Module（论文 MSCM）"""
    def __init__(self, f1=16, pooling_size=POOLING_SIZE,
                 dropout_rate=0.5, number_channel=22):
        super().__init__()
        # 三个并行分支，kernel = 125/62/31
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 125), padding='same'),
            nn.Conv2d(f1, f1, (number_channel, 1), groups=f1),
            nn.BatchNorm2d(f1),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size)),
            nn.Dropout(dropout_rate),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 62), padding='same'),
            nn.Conv2d(f1, f1, (number_channel, 1), groups=f1),
            nn.BatchNorm2d(f1),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size)),
            nn.Dropout(dropout_rate),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 31), padding='same'),
            nn.Conv2d(f1, f1, (number_channel, 1), groups=f1),
            nn.BatchNorm2d(f1),
            nn.ELU(),
            nn.AvgPool2d((1, pooling_size)),
            nn.Dropout(dropout_rate),
        )
        # (b, 3*f1, 1, TP) -> (b, TP, 3*f1)
        self.projection = Rearrange('b e 1 w -> b w e')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (b, 1, chans, time)
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)
        x3 = self.cnn3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return self.projection(x)

class MultiHeadAttention(nn.Module):
    """标准 MHSA"""
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.scale = emb_size ** 0.5
        self.to_q = nn.Linear(emb_size, emb_size)
        self.to_k = nn.Linear(emb_size, emb_size)
        self.to_v = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        # x: (b, seq_len, emb)
        b, n, e = x.shape
        h = self.num_heads
        d = e // h
        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=h)
        k = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h=h)
        v = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h=h)
        scores = torch.einsum('b h i d, b h j d -> b h i j', q, k) / self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.att_drop(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)

class ResidualAdd(nn.Module):
    """残差+LayerNorm"""
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(emb_size)
        self.drop = nn.Dropout(drop_p)

    def forward(self, x, **kwargs):
        return self.norm(x + self.drop(self.fn(x, **kwargs)))

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads, drop_p):
        super().__init__(
            ResidualAdd(
                MultiHeadAttention(emb_size, num_heads, drop_p),
                emb_size, drop_p
            )
        )

class TransformerEncoder(nn.Sequential):
    """堆叠多层 Transformer Encoder Block"""
    def __init__(self, heads=HEADS, depth=DEPTH, emb_size=None):
        layers = [TransformerEncoderBlock(emb_size, heads, drop_p=0.1)
                  for _ in range(depth)]
        super().__init__(*layers)

class TCANet(nn.Module):
    """完整的 TCANet 模型"""
    def __init__(self,
                 input_size,
                 out_features,
                 f1=16,
                 pooling_size=POOLING_SIZE,
                 drop_prob=0.5,
                 tcn_depth=2,
                 tcn_kernel=4,
                 tcn_filters=16,
                 attn_heads=HEADS,
                 attn_depth=DEPTH):
        super().__init__()
        n_chans = input_size[1]
        n_times = input_size[2]
        self.mseegnet = MSCNet(f1, pooling_size, drop_prob, number_channel=n_chans)
        # MSC 输出维度 = 3*f1
        self.tcn = _TCNBlock(
            in_dim=3 * f1,
            depth=tcn_depth,
            kernel_size=tcn_kernel,
            filters=tcn_filters,
            drop_prob=drop_prob
        )
        self.sa = TransformerEncoder(
            heads=attn_heads,
            depth=attn_depth,
            emb_size=tcn_filters
        )
        self.drop = nn.Dropout(drop_prob)
        # Flatten 后长度 = seq_len * tcn_filters
        seq_len = n_times // pooling_size
        self.classifier = nn.Linear(seq_len * tcn_filters, out_features)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, 1, n_chans, n_times)
        返回 logits: (batch, out_features)，features: (batch, seq_len, tcn_filters)
        """
        x = self.mseegnet(x)          # -> (b, seq_len, 3*f1)
        x = self.tcn(x)               # -> (b, seq_len, tcn_filters)
        res = x
        x = self.sa(x)                # -> (b, seq_len, tcn_filters)
        x = self.drop(x + res)
        b, l, f = x.shape
        feats = x                     # 保存作为中间特征
        x = x.reshape(b, l * f)       # flatten
        logits = self.classifier(x)   # -> (b, out_features)
        return logits

###########SFSWTS################################################################################

class SpatialMapper(nn.Module):
    def __init__(self, channel_map, H, W):
        super().__init__()
        self.channel_map = channel_map
        self.H, self.W = H, W

    def forward(self, x):
        # x: [B*T, C, F] -> mapped to [B*T, F, H, W]
        B, C, F = x.shape
        device = x.device
        grid = torch.zeros(B, F, self.H, self.W, device=device)
        for idx, (h, w) in enumerate(self.channel_map):
            if idx < C:
                grid[:, :, h, w] = x[:, idx, :]
        return grid

# ==== Swin Backbone ====
class SwinBackbone(nn.Module):
    def __init__(self, in_chans=1, embed_dim=96):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        feat = self.model(x)  # [B*T, D, 1, 1]
        return feat.view(feat.size(0), -1)  # [B*T, D]

# ==== Temporal Attention ====
class TemporalAttention1(nn.Module):
    def __init__(self, embed_dim, num_heads=6, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.drop(ffn_out))

class SFSWTS(nn.Module):
    def __init__(self, channel_map, H, W, seq_len, num_classes):
        super().__init__()
        self.channel_map = channel_map
        self.H, self.W = H, W
        self.seq_len = seq_len
        self.freq_bands = 1

        self.mapper = SpatialMapper(channel_map, H, W)
        self.swin = SwinBackbone(in_chans=self.freq_bands, embed_dim=64)
        self.temporal = TemporalAttention1(embed_dim=64, num_heads=4)
        self.classifier = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [B, 1, C, T] -> [B, C, F=1, T]
        x = x.permute(0, 2, 1, 3)
        B, C, F, T = x.shape

        # 并行展开所有时间点
        x_reshaped = x.permute(0, 3, 1, 2).reshape(B * T, C, F)  # [B*T, C, F]
        gi = self.mapper(x_reshaped)                            # [B*T, F, H, W]
        feat = self.swin(gi)                                    # [B*T, D]
        feat = feat.view(B, T, -1)                              # [B, T, D]

        out = self.temporal(feat)                               # [B, T, D]
        out = out.mean(dim=1)                                   # [B, D]
        return self.classifier(out)                             # [B, num_classes]

#################CCMTL###################################################################

class Channel(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16):
        super(Channel, self).__init__()
        self.gate_c = nn.Sequential(
            nn.Linear(gate_channel, gate_channel // reduction_ratio),
            nn.BatchNorm1d(gate_channel // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channel // reduction_ratio, gate_channel)
        )

    def forward(self, in_tensor):
        avg_pool = F.avg_pool1d(in_tensor, in_tensor.size(2)).squeeze(2)
        gate_c = self.gate_c(avg_pool)
        return gate_c.unsqueeze(2).expand_as(in_tensor)

class Modulator(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16):
        super(Modulator, self).__init__()
        self.channel_att = Channel(gate_channel, reduction_ratio)

    def forward(self, in_tensor):
        att = torch.sigmoid(self.channel_att(in_tensor))
        return att * in_tensor

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CCMTL(nn.Module):
    def __init__(self, args):
        super(CCMTL, self).__init__()
        self.args = args
        # CNN 模块：输入通道数为 EEG 通道数
        in_ch = args.eeg_channels
        out_ch = args.cnn_out_channels
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2),
            nn.Conv1d(out_ch, out_ch, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(out_ch),
            nn.Dropout(0.2),
        )
        # 计算 Flatten 后维度
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, args.seq_len)
            conv_out = self.cnn(dummy)
            flat_dim = conv_out.shape[1] * conv_out.shape[2]
        # 通道重加权
        self.modulator = Modulator(out_ch, args.reduction_ratio)
        # Flatten + FC
        self.fc_layer = nn.Sequential(
            Flatten(),
            nn.Linear(flat_dim, args.n_units)
        )
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.n_units, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # 双向 LSTM
        rnn_hidden = args.lstm_hidden_size
        self.eeg_rnn1 = nn.LSTM(args.n_units, rnn_hidden, bidirectional=True)
        self.eeg_rnn2 = nn.LSTM(rnn_hidden * 2, rnn_hidden, bidirectional=True)
        # 分类层
        fc_in = 2 * rnn_hidden if args.use_lstm else args.n_units
        self.fc = nn.Linear(fc_in, args.n_classes)

    def forward(self, x):
        # 支持输入 (batch, 1, channels, time) 或 (batch, channels, time)
        if x.dim() == 4:
            b, _, ch, t = x.size()
            x = x.view(b, ch, t)
        # x: [batch, channels, seq_len]
        o = self.cnn(x)
        if self.args.use_modulator:
            o = self.modulator(o)
        o = self.fc_layer(o)           # [batch, n_units]
        o = o.unsqueeze(0)             # [1, batch, n_units]
        o = self.transformer_encoder(o)
        if self.args.use_lstm:
            out1, _ = self.eeg_rnn1(o)
            out2, (h2, _) = self.eeg_rnn2(out1)
            # 拼接最后一层正反向隐藏态
            h_fwd, h_bwd = h2[-2], h2[-1]
            final = torch.cat((h_fwd, h_bwd), dim=1)
            return self.fc(final)
        else:
            o = o.squeeze(0)
            return self.fc(o)

############## CFBM ###################################################################

class CFBM(nn.Module):
    def __init__(self, input_size, num_class):
        super(CFBM, self).__init__()
        # input_size: (channels, H, W)
        in_ch, H, W = input_size

        # 第一层卷积：in_ch -> 16
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=5, padding='same')
        self.in1   = nn.InstanceNorm2d(16)
        self.relu  = nn.ReLU()

        # 第二层卷积：16 -> 1
        self.conv8 = nn.Conv2d(16, 1, kernel_size=1, padding='same')

        # 下采样：H,W -> H/2,W/2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 计算展平后输入全连接的维度
        flattened_dim = 1 * (H // 2) * (W // 2)

        # 全连接层
        self.fla    = nn.Flatten()
        self.dense1 = nn.Linear(flattened_dim, 64)
        self.dense2 = nn.Linear(64, 32)
        self.x_layer= nn.Linear(32, num_class)
        self.softmax= nn.Softmax(dim=1)

    def forward(self, x):
        # x: (batch, in_ch, H, W)
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)

        x = self.conv8(x)
        x = self.relu(x)

        x = self.pool1(x)
        x = self.fla(x)

        x = self.dense1(x)
        x = self.relu(x)

        x = self.dense2(x)
        x = self.relu(x)

        x = self.x_layer(x)
        x = self.softmax(x)
        return x

########### EmT ########################################################
class GraphConvolution(Module):
    """LGG-specific GCN layer."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            torch.empty(in_features, out_features)
        )
        nn.init.xavier_uniform_(self.weight, gain=1.414)

        if bias:
            self.bias = Parameter(
                torch.zeros(1, 1, out_features)
            )
        else:
            self.register_parameter("bias", None)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        output = torch.matmul(x, self.weight)

        if self.bias is not None:
            output = output + self.bias

        output = torch.matmul(adj, output)
        return F.relu(output)


class GCN(Module):
    """Simple GCN layer."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            torch.empty(in_features, out_features)
        )

        if bias:
            self.bias = Parameter(
                torch.empty(out_features)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, data):
        graph, adj = data
        adj = self.norm_adj(adj)

        support = torch.matmul(graph, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            output = output + self.bias

        return F.relu(output), adj

    @staticmethod
    def norm_adj(adj):
        rowsum = torch.sum(adj, dim=-1)
        rowsum = rowsum.masked_fill(rowsum == 0, 1.0)

        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)

        adj = torch.matmul(
            torch.matmul(d_mat_inv_sqrt, adj),
            d_mat_inv_sqrt,
        )
        return adj


class ChebyNet(Module):
    def __init__(self, K, in_feature, out_feature):
        super().__init__()

        if K < 1:
            raise ValueError("K必须大于或等于1。")

        self.K = K

        self.filter_weight, self.filter_bias = self.init_filter(
            K,
            in_feature,
            out_feature,
        )

    @staticmethod
    def init_filter(K, feature, out, bias=True):
        weight = nn.Parameter(
            torch.empty(K, 1, feature, out),
            requires_grad=True,
        )
        nn.init.normal_(weight, mean=0.0, std=0.1)

        bias_parameter = None

        if bias:
            bias_parameter = nn.Parameter(
                torch.zeros(1, 1, out),
                requires_grad=True,
            )
            nn.init.normal_(
                bias_parameter,
                mean=0.0,
                std=0.1,
            )

        return weight, bias_parameter

    @staticmethod
    def get_L(adj):
        degree = torch.sum(adj, dim=-1)
        degree_norm = torch.rsqrt(
            degree.clamp_min(1.0e-5)
        )
        degree_matrix = torch.diag_embed(degree_norm)

        # 近似 lambda_max = 2
        L = -torch.matmul(
            torch.matmul(degree_matrix, adj),
            degree_matrix,
        )

        return L

    @staticmethod
    def rescale_L(L):
        largest_eigval = torch.linalg.eigvalsh(L).amax()
        largest_eigval = largest_eigval.clamp_min(1.0e-5)

        identity = torch.eye(
            L.size(-1),
            device=L.device,
            dtype=L.dtype,
        )

        return (2.0 / largest_eigval) * L - identity

    def chebyshev(self, x, L):
        # X_0 = X
        cheby_terms = [x]

        # X_1 = L * X
        if self.K > 1:
            cheby_terms.append(
                torch.matmul(L, x)
            )

        # X_k = 2 * L * X_(k-1) - X_(k-2)
        for _ in range(2, self.K):
            x_current = (
                2.0 * torch.matmul(L, cheby_terms[-1])
                - cheby_terms[-2]
            )
            cheby_terms.append(x_current)

        # (K, B, C, Fin)
        x_cheby = torch.stack(
            cheby_terms,
            dim=0,
        )

        # filter_weight: (K, 1, Fin, Fout)
        output = torch.matmul(
            x_cheby,
            self.filter_weight,
        )
        output = torch.sum(output, dim=0)

        if self.filter_bias is not None:
            output = output + self.filter_bias

        return F.relu(output)

    def forward(self, data):
        x, adj = data

        L = self.get_L(adj)
        output = self.chebyshev(x, L)

        return output, adj


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(
            self.norm(x),
            **kwargs,
        )


class GraphEncoder1(nn.Module):
    def __init__(
        self,
        num_layers,
        num_node,
        in_features,
        out_features,
        K,
        graph2token="Linear",
        encoder_type="GCN",
    ):
        super().__init__()

        self.graph2token = graph2token
        self.K = K

        supported_tokenizers = [
            "Linear",
            "AvgPool",
            "MaxPool",
            "Flatten",
        ]

        if graph2token not in supported_tokenizers:
            raise ValueError(
                "不支持当前graph2token类型。"
            )

        if graph2token == "Linear":
            self.tokenizer = nn.Linear(
                num_node * out_features,
                out_features,
            )
        else:
            self.tokenizer = None

        layers = []

        for i in range(num_layers):
            if i == 0:
                current_in_features = in_features
            else:
                current_in_features = out_features

            layer = self.get_layer(
                encoder_type,
                current_in_features,
                out_features,
            )
            layers.append(layer)

        self.encoder = nn.Sequential(*layers)

    def get_layer(
        self,
        encoder_type,
        in_features,
        out_features,
    ):
        if encoder_type == "GCN":
            return GCN(
                in_features,
                out_features,
            )

        if encoder_type == "Cheby":
            return ChebyNet(
                self.K,
                in_features,
                out_features,
            )

        raise ValueError(
            "encoder_type必须为'GCN'或'Cheby'。"
        )

    def forward(self, x, adj):
        # x: (batch, channel, feature)
        x, _ = self.encoder((x, adj))

        if self.tokenizer is not None:
            x = x.reshape(x.size(0), -1)
            return self.tokenizer(x)

        if self.graph2token == "AvgPool":
            return torch.mean(x, dim=-1)

        if self.graph2token == "MaxPool":
            return torch.max(x, dim=-1).values

        return x.reshape(x.size(0), -1)


class Attention_emt(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        anchor=3,
        dropout=0.0,
        alpha=0.25,
    ):
        super().__init__()

        inner_dim = dim_head * heads
        project_out = not (
            heads == 1 and dim_head == dim
        )

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(
            dim,
            inner_dim * 3,
            bias=False,
        )

        self.STA = nn.Sequential(
            nn.Dropout(alpha * dropout),
            weight_norm(
                nn.Conv2d(
                    heads,
                    heads,
                    kernel_size=(anchor, 1),
                    stride=1,
                    padding=self.get_padding(anchor),
                )
            ),
        )

        if project_out:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout),
            )
        else:
            self.to_out = nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(
            3,
            dim=-1,
        )

        q, k, v = map(
            lambda tensor: rearrange(
                tensor,
                "b n (h d) -> b h n d",
                h=self.heads,
            ),
            qkv,
        )

        dots = torch.matmul(
            q,
            k.transpose(-1, -2),
        )
        dots = dots * self.scale

        attention = self.attend(dots)

        output = torch.matmul(
            attention,
            v,
        )

        output = self.STA(output)

        output = rearrange(
            output,
            "b h n d -> b n (h d)",
        )

        return self.to_out(output)

    @staticmethod
    def get_padding(kernel):
        return int(0.5 * (kernel - 1)), 0


class TTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        alpha=0.25,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention_emt(
                                dim,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                alpha=alpha,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(
                                dim,
                                mlp_dim,
                                dropout,
                            ),
                        ),
                    ]
                )
            )

    def forward(self, x):
        for attention, feed_forward in self.layers:
            x = attention(x) + x
            x = feed_forward(x) + x

        return x


class EmT(nn.Module):
    def __init__(
        self,
        layers_graph=(1, 2),
        layers_transformer=1,
        num_adj=2,
        num_chan=32,
        num_feature=20,
        hidden_graph=32,
        K=2,
        num_head=8,
        dim_head=16,
        dropout=0.25,
        num_class=2,
        alpha=0.25,
        graph2token="Linear",
        encoder_type="GCN",
    ):
        super().__init__()

        if num_adj < 2:
            raise ValueError(
                "num_adj必须大于或等于2。"
            )

        self.num_chan = num_chan
        self.num_feature = num_feature
        self.graph_encoder_type = encoder_type

        self.GE1 = GraphEncoder1(
            num_layers=layers_graph[0],
            num_node=num_chan,
            in_features=num_feature,
            out_features=hidden_graph,
            K=K,
            graph2token=graph2token,
            encoder_type=encoder_type,
        )

        self.GE2 = GraphEncoder1(
            num_layers=layers_graph[1],
            num_node=num_chan,
            in_features=num_feature,
            out_features=hidden_graph,
            K=K,
            graph2token=graph2token,
            encoder_type=encoder_type,
        )

        self.adjs = nn.Parameter(
            torch.empty(
                num_adj,
                num_chan,
                num_chan,
            ),
            requires_grad=True,
        )
        nn.init.xavier_uniform_(self.adjs)

        transformer_dim = hidden_graph

        if graph2token in ["AvgPool", "MaxPool"]:
            transformer_dim = num_chan

        elif graph2token == "Flatten":
            transformer_dim = (
                num_chan * hidden_graph
            )

        self.transformer = TTransformer(
            dim=transformer_dim,
            depth=layers_transformer,
            heads=num_head,
            dim_head=dim_head,
            mlp_dim=dim_head,
            dropout=dropout,
            alpha=alpha,
        )

        self.to_GNN_out = nn.Linear(
            num_chan * num_feature,
            transformer_dim,
            bias=False,
        )

        self.MLP = nn.Sequential(
            nn.Linear(
                transformer_dim,
                num_class,
            )
        )

    def prepare_input(self, x):
        """
        将原始脑电数据转换成原版EmT需要的(B, S, C, F)。

        支持输入：
            (B, C, T)
            (B, 1, C, T)
            (B, S, C, F)

        当前示例：
            (20, 1, 32, 800)
                ->
            (20, 40, 32, 20)

        其中：
            B = 20
            S = 40个时间窗口
            C = 32个通道
            F = 每个窗口20个时间点
        """

        # (B, C, T) -> (B, 1, C, T)
        if x.ndim == 3:
            x = x.unsqueeze(1)

        if x.ndim != 4:
            raise ValueError(
                "输入必须是(B,C,T)、(B,1,C,T)"
                "或(B,S,C,F)，"
                f"当前输入为{tuple(x.shape)}。"
            )

        _, _, channels, last_dimension = x.shape

        if channels != self.num_chan:
            raise ValueError(
                f"模型要求{self.num_chan}个通道，"
                f"但输入包含{channels}个通道。"
            )

        # 数据已经是EmT要求的(B, S, C, F)
        if last_dimension == self.num_feature:
            return x

        # 最后一维是原始时间点T，将其划分为多个窗口
        remainder = (
            last_dimension % self.num_feature
        )

        # 当时间长度不能整除窗口长度时，在末尾补零
        if remainder != 0:
            padding_length = (
                self.num_feature - remainder
            )
            x = F.pad(
                x,
                (0, padding_length),
            )

        windows_per_group = (
            x.size(-1) // self.num_feature
        )

        # (B, 原序列数, C, T)
        # ->
        # (B, 新序列数, C, F)
        x = rearrange(
            x,
            "b s c (w f) -> b (s w) c f",
            w=windows_per_group,
            f=self.num_feature,
        )

        return x

    def forward(self, x):
        # 将(B, 1, 32, 800)转换为(B, 40, 32, 20)
        x = self.prepare_input(x)

        batch_size, sequence_length, channels, features = x.shape

        # 每个时间窗口分别进行图卷积
        x = rearrange(
            x,
            "b s c f -> (b s) c f",
        )

        if self.graph_encoder_type == "Cheby":
            adjs = self.get_adj(
                self_loop=False
            )
        else:
            adjs = self.get_adj(
                self_loop=True
            )

        # Multi-view pyramid residual GNN block
        residual = x.reshape(
            x.size(0),
            -1,
        )
        residual = self.to_GNN_out(residual)

        x1 = self.GE1(
            x,
            adjs[0],
        )

        x2 = self.GE2(
            x,
            adjs[1],
        )

        x = torch.stack(
            (residual, x1, x2),
            dim=1,
        )
        x = torch.mean(
            x,
            dim=1,
        )

        # 恢复时间序列：
        # (B*S, H) -> (B, S, H)
        x = rearrange(
            x,
            "(b s) h -> b s h",
            b=batch_size,
            s=sequence_length,
        )

        # 时间上下文Transformer
        x = self.transformer(x)

        # 对所有时间窗口进行平均
        x = torch.mean(
            x,
            dim=1,
        )

        # 分类
        x = self.MLP(x)

        return x

    def get_adj(self, self_loop=True):
        num_nodes = self.adjs.shape[-1]

        # 构造对称且非负的邻接矩阵
        adj = F.relu(
            self.adjs
            + self.adjs.transpose(-1, -2)
        )

        if self_loop:
            identity = torch.eye(
                num_nodes,
                device=self.adjs.device,
                dtype=self.adjs.dtype,
            )
            adj = adj + identity.unsqueeze(0)

        return adj


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

####################### EEGDeformer ###########################################


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def cnn_block(self, in_chan, kernel_size, dp):
        return nn.Sequential(
            nn.Dropout(p=dp),
            nn.Conv1d(in_channels=in_chan, out_channels=in_chan,
                      kernel_size=kernel_size, padding=self.get_padding_1D(kernel=kernel_size)),
            nn.BatchNorm1d(in_chan),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, in_chan, fine_grained_kernel=11, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            dim = int(dim * 0.5)
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
                self.cnn_block(in_chan=in_chan, kernel_size=fine_grained_kernel, dp=dropout)
            ]))
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        dense_feature = []
        for attn, ff, cnn in self.layers:
            x_cg = self.pool(x)
            x_cg = attn(x_cg) + x_cg
            x_fg = cnn(x)
            x_info = self.get_info(x_fg)  # (b, in_chan)
            dense_feature.append(x_info)
            x = ff(x_cg) + x_fg
        x_dense = torch.cat(dense_feature, dim=-1)  # b, in_chan*depth
        x = x.view(x.size(0), -1)   # b, in_chan*d_hidden_last_layer
        emd = torch.cat((x, x_dense), dim=-1)  # b, in_chan*(depth + d_hidden_last_layer)
        return emd

    def get_info(self, x):
        # x: b, k, l
        x = torch.log(torch.mean(x.pow(2), dim=-1))
        return x

    def get_padding_1D(self, kernel):
        return int(0.5 * (kernel - 1))


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class Deformer(nn.Module):
    def cnn_block(self, out_chan, kernel_size, num_chan):
        return nn.Sequential(
            Conv2dWithConstraint(1, out_chan, kernel_size, padding=self.get_padding(kernel_size[-1]), max_norm=2),
            Conv2dWithConstraint(out_chan, out_chan, (num_chan, 1), padding=0, max_norm=2),
            nn.BatchNorm2d(out_chan),
            nn.ELU(),
            nn.MaxPool2d((1, 2), stride=(1, 2))
        )

    def __init__(self, *, num_chan, num_time, temporal_kernel, num_kernel=64,
                 num_classes, depth=4, heads=16,
                 mlp_dim=16, dim_head=16, dropout=0.):
        super().__init__()

        self.cnn_encoder = self.cnn_block(out_chan=num_kernel, kernel_size=(1, temporal_kernel), num_chan=num_chan)

        dim = int(0.5*num_time)  # embedding size after the first cnn encoder

        self.to_patch_embedding = Rearrange('b k c f -> b k (c f)')

        self.pos_embedding = nn.Parameter(torch.randn(1, num_kernel, dim))

        self.transformer = Transformer(
            dim=dim, depth=depth, heads=heads, dim_head=dim_head,
            mlp_dim=mlp_dim, dropout=dropout,
            in_chan=num_kernel, fine_grained_kernel=temporal_kernel,
        )

        L = self.get_hidden_size(input_size=dim, num_layer=depth)

        out_size = int(num_kernel * L[-1]) + int(num_kernel * depth)

        self.mlp_head = nn.Sequential(
            nn.Linear(out_size, num_classes)
        )

    def forward(self, eeg):
        # eeg: (b, chan, time)
        # eeg = torch.unsqueeze(eeg, dim=1)  # (b, 1, chan, time)
        x = self.cnn_encoder(eeg)  # (b, num_kernel, 1, 0.5*num_time)

        x = self.to_patch_embedding(x)

        b, n, _ = x.shape
        x += self.pos_embedding
        x = self.transformer(x)
        return self.mlp_head(x)

    def get_padding(self, kernel):
        return (0, int(0.5 * (kernel - 1)))

    def get_hidden_size(self, input_size, num_layer):
        return [int(input_size * (0.5 ** i)) for i in range(num_layer + 1)]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

######## DBGCN #################################################################

def compute_DE_from_raw_torch(raw, fs=200, seg_len_s=1.0, step_s=0.5, bands=None, eps=1e-12, device=None):
    """
    raw: torch.Tensor, shape (B, N, T) or (N, T) or (B, T)  (channels last dim = time)
    returns: de_seq: (B, N, n_frames, n_bands)  (torch.float32 on same device)
    Implementation:
      - frame the signal with seg_len = seg_len_s*fs, step = step_s*fs (use unfold)
      - per frame compute rFFT -> power spectrum -> integrate power in each band
      - DE = 0.5 * log(2*pi*e * band_power)
    """
    if bands is None:
        bands = [(0.5,4),(4,8),(8,12),(12,30)]
    if device is None:
        device = raw.device if isinstance(raw, torch.Tensor) else torch.device('cpu')

    # ensure tensor
    x = raw if isinstance(raw, torch.Tensor) else torch.tensor(raw, dtype=torch.float32, device=device)
    if x.ndim == 2:
        # (N, T) -> add batch dim
        x = x.unsqueeze(0)  # (1, N, T)
    if x.ndim == 1:
        # (T,) -> treat as (1,1,T)
        x = x.unsqueeze(0).unsqueeze(0)

    # now x: (B, N, T)
    B, N, T = x.shape
    seg_len = max(1, int(seg_len_s * fs))
    step = max(1, int(step_s * fs))
    if seg_len > T:
        seg_len = T
        step = seg_len

    # unfold frames along time
    # PyTorch unfold works on 3D only for dimension; use tensor.unfold
    frames = x.unfold(dimension=2, size=seg_len, step=step)  # (B, N, n_frames, seg_len)
    n_frames = frames.size(2)
    if n_frames == 0:
        # fallback: single frame = full length
        seg_len = T
        step = seg_len
        frames = x.unsqueeze(2)  # (B,N,1,T)
        n_frames = 1

    # reshape to (M, seg_len) where M = B * N * n_frames
    M = B * N * n_frames
    frames_2d = frames.contiguous().view(M, seg_len).to(device).float()  # (M, seg_len)

    # apply window
    window = torch.hann_window(seg_len, device=device, dtype=frames_2d.dtype)
    frames_2d = frames_2d * window

    # rfft
    # n_freq = seg_len//2 + 1
    spec = torch.fft.rfft(frames_2d, n=seg_len)  # (M, n_freq), complex
    power = (spec.real ** 2 + spec.imag ** 2)  # (M, n_freq)

    # frequency bins
    n_freq = power.shape[-1]
    freqs = torch.linspace(0.0, fs / 2.0, steps=n_freq, device=device)

    # accumulate band power
    n_bands = len(bands)
    band_power = torch.zeros(M, n_bands, device=device, dtype=power.dtype)  # (M, n_bands)
    for i, (low, high) in enumerate(bands):
        # create mask
        mask = (freqs >= low) & (freqs <= high)
        if mask.sum() == 0:
            # find nearest bin
            # choose the bin with frequency closest to mid-band
            mid = (low + high) / 2.0
            idx = torch.argmin(torch.abs(freqs - mid))
            band_power[:, i] = power[:, idx]
        else:
            band_power[:, i] = power[:, mask].sum(dim=1)

    # DE
    const = 2.0 * math.pi * math.e
    de = 0.5 * torch.log(const * band_power + eps)  # (M, n_bands)

    # reshape back to (B, N, n_frames, n_bands)
    de_seq = de.view(B, N, n_frames, n_bands)  # (B, N, n_frames, n_bands)
    return de_seq.float()

# -----------------------------
# From de_seq -> adjacency (torch)
# -----------------------------
def compute_adjacency_from_de_seq_torch(de_seq, eps=1e-12):
    """
    de_seq: torch.Tensor, shape (B, N, T, F)
    returns: A_pos (B,N,N) in [0,1]
    """
    B, N, T, F = de_seq.shape
    L = T * F
    X = de_seq.reshape(B, N, L).float()
    Xm = X - X.mean(dim=2, keepdim=True)
    numerator = torch.bmm(Xm, Xm.transpose(1, 2))
    sum_sq = (Xm.pow(2).sum(dim=2))
    denom = torch.sqrt(sum_sq.unsqueeze(2) * sum_sq.unsqueeze(1))
    corr = numerator / (denom + eps)
    corr = torch.clamp(corr, -1.0, 1.0)
    A_pos = (corr + 1.0) / 2.0
    return A_pos.float()

# -----------------------------
# BiLSTMEncoder（unchanged interface）
# -----------------------------
class BiLSTMEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers=1, dropout=0.5):
        super().__init__()
        self.in_dim = in_dim
        self.bilstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim,
                              num_layers=n_layers, batch_first=True, bidirectional=True,
                              dropout=dropout if n_layers > 1 else 0.)
        self.fc = nn.Linear(hidden_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, n_ch, T, F) where F == self.in_dim
        B, n_ch, T, n_bands = x.shape
        assert n_bands == self.in_dim, f"Bilstm expected input_size={self.in_dim}, got {n_bands}"
        x2 = x.reshape(B * n_ch, T, n_bands)
        out, _ = self.bilstm(x2)
        last = out[:, -1, :]
        last = self.dropout(last)
        feat = self.fc(last)
        feat = feat.reshape(B, n_ch, -1)
        return feat

# -----------------------------
# GraphConv & spectral pool (GPU power method)
# -----------------------------
class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, H, A):
        I = torch.eye(A.size(-1), device=A.device).unsqueeze(0).expand(A.size(0), -1, -1)
        A_hat = A + I
        D = torch.sum(A_hat, dim=-1)
        D_inv_sqrt = torch.pow(D + 1e-12, -0.5)
        D_inv_sqrt = D_inv_sqrt.unsqueeze(-1)
        A_norm = D_inv_sqrt * A_hat * D_inv_sqrt.transpose(1, 2)
        H_new = torch.bmm(A_norm, H)
        H_new = torch.matmul(H_new, self.weight)
        if self.bias is not None:
            H_new = H_new + self.bias
        return H_new

def batch_spectral_pool_torch(H, A, keep_ratio=0.5, power_iters=20):
    B, N, D = H.shape
    k = max(1, int(math.ceil(N * keep_ratio)))
    device = A.device
    v = torch.randn(B, N, 1, device=device)
    for _ in range(power_iters):
        v = torch.bmm(A, v)
        norms = v.norm(dim=1, keepdim=True) + 1e-12
        v = v / norms
    principal = v.squeeze(-1)
    scores = principal.abs()
    topk_idx = torch.topk(scores, k=k, dim=1)[1]
    Hp_list, Ap_list = [], []
    for b in range(B):
        idx = topk_idx[b]
        Hp_list.append(H[b][idx])
        Ap_list.append(A[b][idx][:, idx])
    Hp = torch.stack(Hp_list, dim=0)
    Ap = torch.stack(Ap_list, dim=0)
    return Hp, Ap

# -----------------------------
# DBGCN (forward accepts raw or de_seq)
# -----------------------------
class DBGCN(nn.Module):
    def __init__(self, in_bands, bilstm_hidden=64, node_feat_dim=128, gcn_hidden=128, n_classes=2, dropout=0.5,
                 fs=200, seg_len_s=1.0, step_s=0.5, bands=None):
        """
        in_bands: number of DE bands (BiLSTM input_size)
        fs / seg_len_s / step_s / bands: params for raw->DE conversion if needed
        """
        super().__init__()
        self.in_bands = in_bands
        self.encoder = BiLSTMEncoder(in_dim=in_bands, hidden_dim=bilstm_hidden, out_dim=node_feat_dim, n_layers=1, dropout=dropout)
        self.gconv1 = GraphConv(node_feat_dim, gcn_hidden)
        self.gconv2 = GraphConv(gcn_hidden, gcn_hidden)
        self.readout_fc = nn.Linear(gcn_hidden, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.brelu_alpha = nn.Parameter(torch.tensor(0.5))
        # raw->DE params
        self.fs = fs
        self.seg_len_s = seg_len_s
        self.step_s = step_s
        self.bands = bands if bands is not None else [(0.5,4),(4,8),(8,12),(12,30)]

    def brelu(self, x):
        return x * torch.sigmoid(self.brelu_alpha * x)

    def forward(self, x, A=None):
        """
        x: either
           - de_seq: (B, N, T, F) where F == in_bands  (preferred)
           - raw: (B, N, T_raw)   (or possibly (N, T_raw) if batchless)
           - raw with extra dims: (B, 1, N, T) etc. We try to detect automatically
        A: optional adjacency (B,N,N). If None, compute from de_seq internally.
        """
        # Normalize input to tensor on device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        device = x.device

        # If input has shape (B, N, T, F) and F matches expected in_bands -> treat as de_seq
        if x.dim() == 4 and x.shape[-1] == self.in_bands:
            de_seq = x.float()
        else:
            # Try to massage input into (B, N, T)
            x_proc = x
            if x_proc.dim() == 5:
                # e.g., (B, regions, freq, channels, time) or similar -> try to squeeze freq dim if 1 and pick first region if multiple
                # We'll squeeze dims of size 1
                # Prefer shape (B, ???, channels, time) -> try to reduce to (B, channels, time)
                s = list(x_proc.shape)
                # If there's a dim equal to 1, squeeze it
                for dim_idx in [1,2]:
                    if s[dim_idx] == 1:
                        x_proc = x_proc.squeeze(dim_idx)
                # after squeezing possibly become 4D -> handle below
            if x_proc.dim() == 4:
                # possible shapes:
                # (B, 1, channels, time) or (B, channels, time, 1) etc.
                # Try to find time dim (largest dimension)
                sizes = list(x_proc.shape)
                # If last dim is very large (time), and second dim small -> assume (B, channels, time) but with extra dim
                # We'll try several heuristics:
                # - if x_proc.shape[1] == 1 and x_proc.shape[3] > 50: squeeze dim1 => (B, channels, time)
                if x_proc.shape[1] == 1 and x_proc.shape[3] > 50:
                    x_proc = x_proc.squeeze(1)  # -> (B, channels, time)
                # - if x_proc.shape[2] == 1 and x_proc.shape[3] > 50: squeeze dim2
                if x_proc.dim() == 4 and x_proc.shape[2] == 1 and x_proc.shape[3] > 50:
                    x_proc = x_proc.squeeze(2)
                # - if it is now (B, channels, time) good
            if x_proc.dim() == 3:
                # now expect (B, N, T)
                raw = x_proc.float().to(device)
            elif x_proc.dim() == 2:
                # (N, T) or (B,T)
                # promote to (1, N, T) if missing batch
                raw = x_proc.unsqueeze(0).float().to(device)
            else:
                # as fallback, try flattening trailing dims to time
                raw = x.view(x.shape[0], x.shape[1], -1).float().to(device)

            # Heuristic: if last dim looks small (<= in_bands) we may already have de_seq with wrong axes.
            if raw.shape[-1] <= self.in_bands:
                # This likely is already de_seq but with swapped axes. Try to permute if possible
                # If shape (B, N, F) and F==in_bands, convert to (B, N, 1, F)
                if raw.dim() == 3 and raw.shape[-1] == self.in_bands:
                    de_seq = raw.unsqueeze(2)  # single time frame
                else:
                    # fallback: raise helpful error
                    raise RuntimeError(f"Input tensor has ambiguous shape {x.shape}; cannot infer raw vs de_seq. "
                                       "If input is raw signal, pass shape (B, N, T) or (B, 1, N, T) etc. "
                                       "If input is de_seq, ensure last dim == in_bands.")
            else:
                # compute DE from raw using GPU-friendly routine
                de_seq = compute_DE_from_raw_torch(raw, fs=self.fs, seg_len_s=self.seg_len_s, step_s=self.step_s,
                                                   bands=self.bands, device=device)  # (B,N,n_frames,n_bands)

        # ensure de_seq is (B,N,T,F) and last dim equals in_bands
        if de_seq.dim() != 4:
            raise RuntimeError(f"de_seq should be 4D (B,N,T,F). Got shape {de_seq.shape}")
        if de_seq.shape[-1] != self.in_bands:
            raise RuntimeError(f"de_seq last dim {de_seq.shape[-1]} != in_bands {self.in_bands}")

        B, N, T_frames, F_bands = de_seq.shape
        H = self.encoder(de_seq)  # (B, N, node_feat_dim)
        H = F.relu(H)

        # adjacency
        if A is None:
            A = compute_adjacency_from_de_seq_torch(de_seq)
        else:
            if not isinstance(A, torch.Tensor):
                A = torch.tensor(A, device=H.device, dtype=torch.float32)
            else:
                A = A.to(H.device).float()

        # GCN1
        H1 = self.gconv1(H, A)
        H1 = self.brelu(H1)

        # spectral pooling (reduce nodes)
        Hp, Ap = batch_spectral_pool_torch(H1, A, keep_ratio=0.5, power_iters=20)

        # GCN2
        H2 = self.gconv2(Hp, Ap)
        H2 = self.brelu(H2)

        # readout
        g = H2.mean(dim=1)
        g = self.dropout(g)
        logits = self.readout_fc(g)
        return logits

### ACCNet ##############################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from torch_geometric.nn import GCNConv, global_mean_pool, GATv2Conv
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data  # 放在文件头部已 import 的位置

criterion = torch.nn.CrossEntropyLoss()
import numpy as np
import argparse
import torch.fft as fft

old_repr = torch.Tensor.__repr__


def tensor_info(tensor):
    return repr(tensor.shape)[6:] + ' ' + repr(tensor.dtype)[6:] + '@' + str(tensor.device) + '\n' + old_repr(tensor)


torch.Tensor.__repr__ = tensor_info

devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Alembic_Layer(nn.Module):
    def __init__(self, args=None, num_channels=32, signal_length=500, fs=128, numtaps=101):
        super(Alembic_Layer, self).__init__()
        torch.manual_seed(8989)
        self.fs = fs
        self.numtaps = numtaps
        self.num_channels = num_channels
        self.signal_length = signal_length

    def forward(self, x, filter_params_batch):
        # x can be either:
        #  - (N_nodes, L)  where N_nodes = B * C (old behavior)
        #  - (B, C, L)  (new FATIG-style)
        if x.dim() == 2:
            # old style (N_nodes, L)
            N_nodes, L = x.shape
            # try to infer batch:
            # assume filter_params_batch.shape == (B, 3, 2)
            batch_size = filter_params_batch.shape[0]
            num_channels = N_nodes // batch_size
            x = x.view(batch_size, num_channels, L)
        elif x.dim() == 3:
            batch_size, num_channels, L = x.shape
        else:
            raise ValueError("Unsupported x shape in Alembic_Layer.forward: {}".format(x.shape))

        # make sure internal params match
        self.num_channels = num_channels
        self.signal_length = L

        # filter_params_batch expected shape: (B, 3, 2)
        lowcuts, highcuts = filter_params_batch[:, :, 0], filter_params_batch[:, :, 1]

        # generate kernels
        filter_kernels = self.batch_bandpass_filter_kernel_vectorized(lowcuts, highcuts, self.fs, self.numtaps)

        # x: (B, C, L) -> dynamic_conv expects (B, C, L)
        x = self.dynamic_conv(x, filter_kernels, batch_size)

        # output shape expected: (B, num_filters, C, L)
        x = x.view(batch_size, filter_kernels.size(1), num_channels, L)
        return x

    def dynamic_conv(self, x, filter_kernels, batch_size):
        # x: (B, C, L)
        in_channels, signal_length = x.size(1), x.size(2)
        num_filters = filter_kernels.size(1)
        kernel_size = filter_kernels.size(2)

        # filter_kernels: (B, num_filters, K) -> expand to group conv
        filter_kernels = filter_kernels.view(batch_size * num_filters, 1, kernel_size)
        filter_kernels = filter_kernels.repeat(1, in_channels, 1)
        filter_kernels = filter_kernels.view(batch_size * in_channels * num_filters, 1, kernel_size)

        # flatten x to (1, B*C, L)
        x = x.view(1, batch_size * in_channels, signal_length)
        x = nn.functional.conv1d(x, filter_kernels, groups=batch_size * in_channels, padding='same')

        # restore to (B, C * num_filters, L)
        x = x.view(batch_size, in_channels * num_filters, signal_length)
        return x

    def batch_bandpass_filter_kernel_vectorized(self, lowcuts, highcuts, fs, numtaps):
        # lowcuts / highcuts shape: (B, num_filters)
        batch_size = lowcuts.size(0)
        num_filters = lowcuts.size(1)
        nyquist = 0.5 * fs
        lows = lowcuts / nyquist
        highs = highcuts / nyquist
        n = torch.arange(numtaps, device=devices) - (numtaps - 1) / 2
        n = n.view(1, 1, -1).repeat(batch_size, num_filters, 1)
        # avoid division by zero at center
        pi = torch.tensor(np.pi, device=devices, dtype=torch.float32)
        # compute taps using sinc difference
        taps = (torch.sin(pi * n * highs.unsqueeze(2)) - torch.sin(pi * n * lows.unsqueeze(2))) / (pi * n)
        taps[:, :, (numtaps - 1) // 2] = highs - lows
        window = torch.hann_window(numtaps).view(1, 1, -1).repeat(batch_size, num_filters, 1).to(devices)
        taps *= window
        return taps.view(batch_size, num_filters, numtaps)


class Dynamic_Anchor_Layer(nn.Module):
    def __init__(self, kernel_size=3, sigma=2, n_peaks=3, min_peak_distance=3,
                 num_channels=32, signal_length=500, fs=128):
        super(Dynamic_Anchor_Layer, self).__init__()
        torch.manual_seed(8989)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.sample_rate = fs
        self.signal_length = signal_length
        self.num_channels = num_channels
        self.min_peak_distance = min_peak_distance
        self.n_peaks = n_peaks

    def forward(self, x):
        # Accept x of shape (N_nodes, L) or (B, C, L)
        if x.dim() == 2:
            # (N_nodes, L) -> try to infer B and C not available here
            # We assume caller knows to pass correct x; treat each row as a channel
            signal = x
            batch_size = 1
        elif x.dim() == 3:
            signal = x
            batch_size = x.shape[0]
            self.num_channels = x.shape[1]
            self.signal_length = x.shape[2]
        else:
            raise ValueError("Unsupported x shape in Dynamic_Anchor_Layer.forward: {}".format(x.shape))

        peak_freqs, f_d = self.analyze_signal_frequency(signal)
        # peak_freqs shape: (B, n_peaks)
        # derive cut lines (low/mid/high) per sample
        freqs_sorted, _ = torch.sort(peak_freqs, dim=-1)
        Cut_Line_1stStage = (freqs_sorted[:, 1] - freqs_sorted[:, 0]) / 2 + freqs_sorted[:, 0]
        Cut_Line_2rdStage = (freqs_sorted[:, 2] - freqs_sorted[:, 1]) / 2 + freqs_sorted[:, 1]

        LOW_lowcut = (torch.ones(size=[peak_freqs.shape[0]]) * 0.5).unsqueeze(1).unsqueeze(1).to(devices)
        LOW_highcut = Cut_Line_1stStage.unsqueeze(1).unsqueeze(1)
        MID_lowcut = Cut_Line_1stStage.unsqueeze(1).unsqueeze(1)
        MID_highcut = Cut_Line_2rdStage.unsqueeze(1).unsqueeze(1)
        HIG_lowcut = Cut_Line_2rdStage.unsqueeze(1).unsqueeze(1)
        HIG_highcut = (torch.ones(size=[peak_freqs.shape[0]]) * 45).unsqueeze(1).unsqueeze(1).to(devices)

        cuts = (torch.cat((LOW_lowcut, LOW_highcut,
                           MID_lowcut, MID_highcut,
                           HIG_lowcut, HIG_highcut), dim=1)).view(peak_freqs.shape[0], 3, 2)
        return cuts.to(devices), f_d.to(devices)

    def gaussian_kernel(self):
        k = torch.arange(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1, dtype=torch.float32, device=devices)
        kernel = torch.exp(-k ** 2 / (2 * self.sigma ** 2))
        return kernel / kernel.sum()

    def analyze_signal_frequency(self, signal):
        # signal can be (B, C, L) or (N_nodes, L)
        if signal.dim() == 3:
            batch_size, channels, signal_length = signal.shape
            # rfft along time
            fft_result = fft.rfft(signal, dim=-1)
            frequencies = fft.rfftfreq(signal_length, 1 / self.sample_rate).to(devices)
            f_d = torch.abs(fft_result) ** 2  # (B, C, Nfreq)
            # average across channels -> sample-level PSD
            f_ = f_d.mean(1)  # (B, Nfreq)
        elif signal.dim() == 2:
            # treat each row as a channel for a single sample
            N, signal_length = signal.shape
            fft_result = fft.rfft(signal, dim=-1)  # (N, Nfreq)
            frequencies = fft.rfftfreq(signal_length, 1 / self.sample_rate).to(devices)
            f_d = torch.abs(fft_result) ** 2
            # average across channels (rows)
            f_ = f_d.mean(0, keepdim=True)  # (1, Nfreq)
        else:
            raise ValueError("Unsupported signal dim in analyze_signal_frequency")

        # drop DC component to avoid selecting 0 Hz as peak
        if f_.shape[-1] > 2:
            f_no_dc = f_[:, 1:]
            freqs_no_dc = frequencies[1:]
        else:
            f_no_dc = f_
            freqs_no_dc = frequencies

        # smooth PSD
        kernel = self.gaussian_kernel().to(devices)
        smoothed = F.conv1d(f_no_dc.unsqueeze(1), kernel.view(1, 1, -1), padding=self.kernel_size // 2).squeeze(1)

        batch_size = smoothed.shape[0]
        nfreqs = smoothed.shape[1]
        peaks = (torch.diff(smoothed, dim=1)[:, :-1] > 0) & (torch.diff(smoothed, dim=1)[:, 1:] < 0)
        peaks = F.pad(peaks, (1, 1), "constant", 0)
        peak_frequencies = torch.zeros((batch_size, self.n_peaks), dtype=torch.float32, device=devices)
        peak_magnitudes = torch.zeros((batch_size, self.n_peaks), dtype=torch.float32, device=devices)

        # select top peaks with spacing constraint
        _, sorted_idx = torch.sort(smoothed, dim=1, descending=True)
        sorted_peaks = torch.gather(peaks, 1, sorted_idx)
        sorted_freqs = torch.gather(freqs_no_dc.unsqueeze(0).expand(batch_size, -1), 1, sorted_idx)

        mask = torch.ones_like(sorted_peaks, dtype=torch.bool)
        for i in range(1, sorted_peaks.size(1)):
            mask[:, i] = torch.all(
                torch.abs(sorted_freqs[:, i].unsqueeze(1) - sorted_freqs[:, :i]) > self.min_peak_distance,
                dim=1)

        valid_peaks = sorted_peaks & mask
        cumsum = torch.cumsum(valid_peaks.float(), dim=1)
        top_n = (cumsum <= self.n_peaks) & valid_peaks

        for i in range(batch_size):
            sel = sorted_idx[i][top_n[i]]
            num_sel = sel.size(0)
            if num_sel > 0:
                peak_frequencies[i, :num_sel] = freqs_no_dc[sel]
                peak_magnitudes[i, :num_sel] = smoothed[i, sel]
            if num_sel < self.n_peaks:
                rem = self.n_peaks - num_sel
                rand_idx = torch.randint(0, nfreqs, (rem,), device=devices)
                peak_frequencies[i, num_sel:] = freqs_no_dc[rand_idx]
                peak_magnitudes[i, num_sel:] = smoothed[i, rand_idx]

        return peak_frequencies, f_d


class EdgeWeightingModel(nn.Module):
    def __init__(self, num_edges, channels):
        super(EdgeWeightingModel, self).__init__()
        self.ch = channels
        # dynamic weight matrix size
        self.weight_matrix = nn.Parameter(torch.ones(channels, channels))
        self.alpha = nn.Parameter(torch.tensor(0.5))
        nn.init.kaiming_uniform_(self.weight_matrix, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, edges_1, edges_2, edges_3):
        # edges_i expected flattened; reshape to (B, ch, ch)
        edges_1 = edges_1.view(-1, self.ch, self.ch)
        edges_2 = edges_2.view(-1, self.ch, self.ch)
        edges_3 = edges_3.view(-1, self.ch, self.ch)

        normalized_weight = nn.functional.softmax(self.weight_matrix.view(-1), dim=0).view(self.ch, self.ch)

        weighted_edges_1 = edges_1 * normalized_weight.unsqueeze(0)

        edges_2_weighted = self.alpha * torch.bmm(weighted_edges_1, edges_2.transpose(1, 2)) + (
                    1 - self.alpha) * edges_2
        edges_3_weighted = self.alpha * torch.bmm(weighted_edges_1, edges_3.transpose(1, 2)) + (
                    1 - self.alpha) * edges_3

        edges_2_weighted = edges_2_weighted.view(-1)
        edges_3_weighted = edges_3_weighted.view(-1)

        return edges_1.view(-1), edges_2_weighted, edges_3_weighted


class Output_encoder(nn.Module):

    def __init__(self):
        super(Output_encoder, self).__init__()
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1
        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 6
        self.dropout = 0.2

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(self.input_channels, self.mid_channels, kernel_size=self.kernel_size,
                      stride=self.stride, bias=False, padding=(self.kernel_size // 2)),
            nn.BatchNorm1d(self.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(self.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(self.mid_channels, self.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(self.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(self.mid_channels * 2, self.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(self.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.features_len)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)

        x_flat = x.reshape(x.shape[0], -1)
        return x_flat


def hilbert_torch(x):
    N = x.shape[-1]
    Xf = fft.fft(x, dim=-1)
    h = torch.zeros(N, dtype=torch.complex64, device=x.device)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    return fft.ifft(Xf * h, dim=-1)


def calculate_phase_difference_matrix(signals):
    analytic_signals = hilbert_torch(signals)
    phases = torch.angle(analytic_signals)
    phase_diff = phases.unsqueeze(2) - phases.unsqueeze(1)
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    phase_diff_matrix = phase_diff.mean(dim=-1)
    return phase_diff_matrix


def sigmoid_normalize_phase_diff_matrix(phase_diff_matrix):
    return torch.sigmoid(phase_diff_matrix)


def get_edge_attr(type, signal_patch, batch):
    # signal_patch expected shape either (B*C, L) or (B, C, L) flattened
    if type.upper() in ('COS', 'CORR'):
        # compute cosine similarity matrix per-sample
        signal_patch = signal_patch.reshape(batch, signal_patch.shape[0] // batch, signal_patch.shape[-1])
        norms = torch.norm(signal_patch, dim=2, keepdim=True)
        signal_patch_normalized = signal_patch / (norms + 1e-8)
        sim_matrices = torch.bmm(signal_patch, signal_patch_normalized.transpose(1, 2))
        edge_attr = sim_matrices.view(-1)
        edge_attr_2 = (edge_attr - edge_attr.min()) / (edge_attr.max() - edge_attr.min() + 1e-10)
        edge_attr = edge_attr_2
    elif type.upper() in ('PLI',):
        signal_patch = signal_patch.reshape(batch, signal_patch.shape[0] // batch, signal_patch.shape[-1])
        phase_diff_matrices = calculate_phase_difference_matrix(signal_patch)
        normalized_matrices = sigmoid_normalize_phase_diff_matrix(phase_diff_matrices)
        edge_attr = normalized_matrices.view(-1)
    else:
        # fallback: ones
        edge_attr = torch.ones(signal_patch.numel(), device=signal_patch.device)
    return edge_attr


class ACCNet(nn.Module):

    def __init__(self, args, in_channels, hidden_channels, out_channels, num_channels, signal_length, fs=128):
        super(ACCNet, self).__init__()
        torch.manual_seed(9898)
        # set layers with dynamic channel/length
        self.DAL = Dynamic_Anchor_Layer(fs=fs, num_channels=num_channels, signal_length=signal_length)
        self.AL = Alembic_Layer(args, num_channels=num_channels, signal_length=signal_length, fs=fs)

        self.edge_compute = getattr(args, 'edge_compute', 'COS')
        self.batch_size = getattr(args, 'batch_size', 1)
        hidden_channels = hidden_channels
        heads = getattr(args, 'GNN_inheads', 4)

        # in_channels: temporal length for branch features
        in_channels = signal_length
        in_channels_psd = signal_length // 2 + 1

        self.GAT_raw = GATv2Conv(in_channels_psd, hidden_channels, edge_dim=1, heads=heads, concat=True)
        self.GAT1 = GATv2Conv(in_channels, hidden_channels, edge_dim=1, heads=heads, concat=True)
        self.GAT2 = GATv2Conv(in_channels, hidden_channels, edge_dim=1, heads=heads, concat=True)
        self.GAT3 = GATv2Conv(in_channels, hidden_channels, edge_dim=1, heads=heads, concat=True)

        self.encoder = nn.Sequential(
            nn.BatchNorm1d(hidden_channels * heads),
            nn.Linear(hidden_channels * heads, out_channels)
        )

        self.Cross = EdgeWeightingModel(num_edges=1024, channels=num_channels)
        self.Beta = getattr(args, 'T', 0.5)

    def forward(self, data):
        """
        兼容两类输入：
          - data: torch_geometric.data.Data（含 .x, .edge_index, .batch，可选 .y）
          - data: Tensor，形状 (B, C, L)
        """
        data = torch.squeeze(data, dim=1)
        # 如果直接传入 Tensor，则构造 Data（complete graph）
        if isinstance(data, torch.Tensor):
            x_in = data  # (B, C, L)
            B, C, L = x_in.shape
            device = x_in.device

            rows = []
            cols = []
            for b in range(B):
                start = b * C
                idx = torch.arange(start, start + C, device=device)
                rr = idx.repeat_interleave(C)
                cc = idx.repeat(C)
                rows.append(rr)
                cols.append(cc)
            edge_index = torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)  # (2, B*C*C)
            batch_vec = torch.repeat_interleave(torch.arange(B, device=device), repeats=C)
            data = Data(x=x_in.to(device), edge_index=edge_index.to(device), batch=batch_vec.to(device))
        else:
            # 保证 data 是 Data-like
            if not hasattr(data, 'x'):
                raise ValueError("forward expects Data or Tensor with shape (B,C,L).")

        x_in = data.x
        device = x_in.device

        # Adaptive Parameter Learning Unit (APU)
        filter_params, f_d = self.DAL(x_in)

        # Dynamic Filtering Unit (DFU)
        x = self.AL(x_in, filter_params)  # -> (B, 3, C, L)
        bz = x.shape[0]
        wave = x.shape[1]
        chans = x.shape[2]
        fea = x.shape[3]

        # Graph init
        f_d = f_d.view(f_d.shape[0] * f_d.shape[1], f_d.shape[2]).float()  # (B*C, Nfreq)

        x_branch1 = x[:, 0, :, :].reshape(x.shape[0] * x.shape[2], -1).float()
        x_branch2 = x[:, 1, :, :].reshape(x.shape[0] * x.shape[2], -1).float()
        x_branch3 = x[:, 2, :, :].reshape(x.shape[0] * x.shape[2], -1).float()

        # --------- 这里用稳健方式推断 batch_len（修复 None 情况） ----------
        if hasattr(data, 'y') and getattr(data, 'y') is not None:
            # data.y 存在且不为 None
            if isinstance(data.y, torch.Tensor):
                batch_len = int(data.y.shape[0])
            else:
                try:
                    batch_len = int(len(data.y))
                except Exception:
                    batch_len = None
        else:
            batch_len = None

        if batch_len is None:
            # 尝试从 data.batch 推断
            if hasattr(data, 'batch') and getattr(data, 'batch') is not None:
                try:
                    batch_len = int(data.batch.max().item()) + 1
                except Exception:
                    batch_len = None

        if batch_len is None:
            # 如果 x_in 是 (B, C, L) ，直接用 B
            if x_in.dim() == 3:
                batch_len = int(x_in.shape[0])
            else:
                # x_in 可能是 (N_nodes, L) -> 尝试根据 DAL.num_channels 推断 B = N_nodes / C
                num_ch = getattr(self.DAL, 'num_channels', None)
                if num_ch and x_in.shape[0] % num_ch == 0:
                    batch_len = int(x_in.shape[0] // num_ch)
                else:
                    # 最保守的 fallback
                    batch_len = 1
        # -----------------------------------------------------------------

        # build edge weights（注意 get_edge_attr 内部会用 batch 参数）
        edge_weight1 = get_edge_attr(type=self.edge_compute,
                                     signal_patch=(x[:, 0, :, :]).reshape(x.shape[0] * x.shape[2], -1).float(),
                                     batch=batch_len).to(device)
        edge_weight2 = get_edge_attr(type=self.edge_compute,
                                     signal_patch=(x[:, 1, :, :]).reshape(x.shape[0] * x.shape[2], -1).float(),
                                     batch=batch_len).to(device)
        edge_weight3 = get_edge_attr(type=self.edge_compute,
                                     signal_patch=(x[:, 2, :, :]).reshape(x.shape[0] * x.shape[2], -1).float(),
                                     batch=batch_len).to(device)

        f_weight = get_edge_attr(type=self.edge_compute, signal_patch=f_d, batch=batch_len).to(device)

        # Cross-frequency coupling
        _, edge_weight2, edge_weight3 = self.Cross(edge_weight1, edge_weight2, edge_weight3)

        # GATs（edge_attr 需与 data.edge_index 列对齐，形式为 (E,1)）
        x_psd = self.GAT_raw(f_d.to(device), data.edge_index.to(device), f_weight.unsqueeze(1).to(device))
        x_psd = global_mean_pool(x_psd, data.batch.to(device))

        x_branch1 = self.GAT1(x_branch1.to(device), data.edge_index.to(device), edge_weight1.unsqueeze(1).to(device))
        x_branch1 = global_mean_pool(x_branch1, data.batch.to(device))

        x_branch2 = self.GAT2(x_branch2.to(device), data.edge_index.to(device), edge_weight2.unsqueeze(1).to(device))
        x_branch2 = global_mean_pool(x_branch2, data.batch.to(device))

        x_branch3 = self.GAT3(x_branch3.to(device), data.edge_index.to(device), edge_weight3.unsqueeze(1).to(device))
        x_branch3 = global_mean_pool(x_branch3, data.batch.to(device))

        combined_branch = x_psd + x_branch1 + self.Beta * (x_branch2 + x_branch3)

        x_out = self.encoder(combined_branch)

        return x_out

######### FG_HANet_Simple #######################################################

import torch
import torch.nn as nn
import numpy as np

# ---------------------------
# Utility: design FIR filters
# ---------------------------
def design_bandpass_fir_fs(fs=200, low_hz=0.1, high_hz=76.0, bw=2.0, F2=38, L=41):
    """
    生成 F2 个带通 FIR（sinc 差分），核长 L，返回 (F2, L) 的 torch.Tensor
    采用 L2 归一化以便初始化 Conv 权重数值稳定。
    """
    bands = []
    start = low_hz
    for i in range(F2):
        f1 = start + i * bw
        f2 = f1 + bw
        if f2 > high_hz:
            f2 = high_hz
        bands.append((f1, f2))
    n = np.arange(L)
    mid = (L - 1) / 2.0
    filters = np.zeros((F2, L), dtype=np.float32)
    for i, (fl, fh) in enumerate(bands):
        wl = 2 * np.pi * fl / fs
        wh = 2 * np.pi * fh / fs
        h = np.zeros(L, dtype=np.float32)
        for idx in range(L):
            t = idx - mid
            if abs(t) < 1e-8:
                h[idx] = (wh - wl) / np.pi
            else:
                h[idx] = (np.sin(wh * t) - np.sin(wl * t)) / (np.pi * t)
        # L2 normalize
        norm = np.linalg.norm(h) + 1e-12
        h = h.astype(np.float32) / norm
        filters[i] = h
    return torch.from_numpy(filters)  # (F2, L)


# ---------------------------
# MirrorFlip Block
# ---------------------------
class MirrorFlipBlock(nn.Module):
    """
    工程化实现：将通道顺序翻转（torch.flip），适用于未提供电极映射的情况。
    若你提供具体电极对映射（例如10-20索引），我可替换为严格的“保留中线、交换左右对”的实现。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (B, E, T)
        return torch.flip(x, dims=[1])


# ---------------------------
# VSConv (空间卷积)
# ---------------------------
class VSConv(nn.Module):
    """
    把 (B, E, T) 看作 (B, 1, E, T)，对高度 E 做 kernel-size=E 的 Conv2d -> produce F1 spatial components.
    输出 (B, F1, T)
    """
    def __init__(self, in_channels_E, F1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=F1, kernel_size=(in_channels_E, 1), bias=False)

    def forward(self, x):
        # x: (B, E, T)
        # x2 = x.unsqueeze(1)  # (B,1,E,T)
        out = self.conv(x)  # (B,F1,1,T)
        out = out.squeeze(2)  # (B,F1,T)
        return out


# ---------------------------
# VTConv with FIR init
# ---------------------------
class VTConv_FIR_Init(nn.Module):
    """
    对每个空间通道应用 F2 个 FIR 滤波器（groups=F1）。
    输入 (B, F1, T) -> Conv1d grouped -> 输出 reshape 成 (B, F2, F1, T_out)
    """
    def __init__(self, F1, F2=38, L=41, fs=200):
        super().__init__()
        self.F1 = F1
        self.F2 = F2
        self.L = L
        self.fs = fs
        self.conv = nn.Conv1d(in_channels=F1, out_channels=F1 * F2, kernel_size=L, groups=F1, bias=False)
        self._init_fir_filters()

    def _init_fir_filters(self):
        filters = design_bandpass_fir_fs(fs=self.fs, F2=self.F2, L=self.L)  # (F2, L)
        weight = torch.zeros((self.F1 * self.F2, 1, self.L), dtype=torch.float32)
        for j in range(self.F1):
            for f in range(self.F2):
                out_idx = j * self.F2 + f
                weight[out_idx, 0, :] = filters[f]
        self.conv.weight.data = weight

    def forward(self, x):
        # x: (B, F1, T)
        out = self.conv(x)  # (B, F1*F2, T_out)
        B, C, T_out = out.shape
        out = out.view(B, self.F1, self.F2, T_out).permute(0, 2, 1, 3).contiguous()
        # (B, F2, F1, T_out)
        return out


# ---------------------------
# Power Calculation (PC)
# ---------------------------
class PowerCalculationModule(nn.Module):
    """
    BN -> square -> time-global-avg-pool.
    Input V: (B, F2, F1, T) -> output p: (B, F2, F1)
    """
    def __init__(self, F2, F1):
        super().__init__()
        self.bn = nn.BatchNorm1d(F2 * F1, eps=1e-5, momentum=0.1)
        self.F2 = F2
        self.F1 = F1

    def forward(self, V):
        B, F2, F1, T = V.shape
        v_flat = V.view(B, F2 * F1, T)
        v_bn = self.bn(v_flat)
        v_sq = v_bn.pow(2)
        v_gap = v_sq.mean(dim=2)  # (B, F2*F1)
        p = v_gap.view(B, F2, F1)
        return p


# ---------------------------
# HAD (Hemispheric Asymmetry Discriminator)
# ---------------------------
class HAD_Discriminator(nn.Module):
    """
    论文中说明每个 band 有 ω_g ∈ R^{F1}，我们实现为对每个 (F2, F1) 的可学习缩放系数 omega。
    d(p) = omega * p，z = p_tilde - d(p)
    """
    def __init__(self, F2, F1):
        super().__init__()
        self.omega = nn.Parameter(torch.zeros(1, F2, F1))
        nn.init.normal_(self.omega, mean=0.0, std=0.01)

    def forward(self, p, p_tilde):
        d_p = self.omega * p
        z = p_tilde - d_p
        return z


# ---------------------------
# Complete FG-HANet (simplified runnable reproduction)
# ---------------------------
class FG_HANet_Simple(nn.Module):
    """
    完整前向包含：Mirror -> VSConv(right/left) -> VTConv(right/left) -> PC -> HAD -> classifier
    """
    def __init__(self, E=32, F1=41, F2=38, L=41, fs=200, num_classes=2):
        super().__init__()
        self.mirror = MirrorFlipBlock()
        self.vsconv_right = VSConv(in_channels_E=E, F1=F1)
        self.vsconv_left = VSConv(in_channels_E=E, F1=F1)
        self.vtconv_right = VTConv_FIR_Init(F1=F1, F2=F2, L=L, fs=fs)
        self.vtconv_left = VTConv_FIR_Init(F1=F1, F2=F2, L=L, fs=fs)
        self.pc = PowerCalculationModule(F2=F2, F1=F1)
        self.had = HAD_Discriminator(F2=F2, F1=F1)
        self.classifier = nn.Linear(F2 * F1, num_classes)

    def forward(self, x):
        # x: (B, E, T)
        x_tilde = self.mirror(x)
        v_s_right = self.vsconv_right(x)        # (B, F1, T)
        v_t_right = self.vtconv_right(v_s_right) # (B, F2, F1, T_out)
        p_right = self.pc(v_t_right)            # (B, F2, F1)

        v_s_left = self.vsconv_left(x_tilde)
        v_t_left = self.vtconv_left(v_s_left)
        p_left = self.pc(v_t_left)

        z = self.had(p_right, p_left)  # (B, F2, F1)
        B = z.shape[0]
        feat = z.view(B, -1)
        logits = self.classifier(feat)
        return logits


# ---------------------------
# Helper: count parameters
# ---------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------
# Main block: run on data=(16,32,1000)
# ---------------------------
if __name__ == "__main__":
    # match the style of你的参考 main：构建模型 -> 打印 -> 运行 forward
    data = torch.ones((16, 32, 1000), dtype=torch.float32)  # (B,C,T)

    # you can adjust num_classes to 2 per your example
    model = FG_HANet_Simple(E=32, F1=41, F2=38, L=41, fs=200, num_classes=2)
    print(model)
    print("Parameter count:", count_parameters(model))

    out_logits= model(data)
    print("out_logits.shape =", out_logits.shape)


############### TFDEEG ######################################################################################

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
import numpy as np
from config.config import *
from torch_geometric.utils import to_dense_batch
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
import numpy as np

from torch.nn.utils import weight_norm
from einops import rearrange
from einops.layers.torch import Rearrange

_, os.environ['CUDA_VISIBLE_DEVICES'] = set_config()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



#滑动因果频率 + 树卷积

class TFDEEG(nn.Module):
#temporal_learner 方法定义了一个时间卷积块，用于提取时间特征。包含一个二维卷积层和一个功率层，用于对卷积后的特征进行幂变换。
    def temporal_learner(
            self, in_chan, out_chan, kernel, pool, pool_step_rate):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
            PowerLayer(dim=-1, length=pool, step=int(pool_step_rate * pool))
        )
#__init__ 方法初始化了 ATDGNN 模型的各种参数和层
    def __init__(self, num_heads, num_classes, input_size, sampling_rate, num_T,
                 out_graph, dropout_rate, pool, pool_step_rate, idx_graph):
        super(TFDEEG, self).__init__()

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
        self.num_heads = num_heads
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
        self.Tception1 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[0] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception2 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[1] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception3 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[2] * sampling_rate)),
                                               self.pool, pool_step_rate)
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
        self.sliding_window_processor = SlidingWindowProcessor(
            model_dim=self.channel,
            num_heads=self.num_heads,
            window_size=self.window_size,
            stride=self.stride,
            dropout=0.0  # 根据需要设置
        )
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

    def get_size_temporal(self, input_size):
        # input_size: frequency x channel x data point
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        z = self.Tception1(data)
        out = z
        z = self.Tception2(data)
        out = torch.cat((out, z), dim=-1)
        z = self.Tception3(data)
        out = torch.cat((out, z), dim=-1) #out=(1,64,32,547)
        #######################################
        out = self.feature_integrator(out)  # 特征整合和降维out=(1,32,547)
        out = self.sliding_window_processor(out)  # 滑动窗口处理
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
        # Temporal convolution
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        ##############################
        out = self.feature_integrator(out)  # 特征整合和降维
        out = self.sliding_window_processor(out)  # 滑动窗口处理(20,32,2300)
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
# —— 左右半球节点简单版 ——
##############################
class GraphConvolution_tree(nn.Module):
    """树形结构的简单GCN层（1层卷积+ReLU）。"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.zeros((1, 1, out_features)))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj):
        # x: [B, N, in_features], adj: [N, N]
        h = torch.matmul(x, self.weight)  # [B, N, out_features]
        if self.bias is not None:
            h = h + self.bias
        h = torch.matmul(adj, h)         # 聚合邻居节点特征
        return F.relu(h)

'''高性能左右(最后一个维度拼接)+全局的三层（最后一个维度，先半后全）'''
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
        self.gc_leaf_proj = GraphConvolution_tree(in_f, out_f)

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

class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))
#sample_rate=200
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

'''轻量版'''
class FeatureIntegrator(nn.Module):
    def __init__(self,sr, in_channels, out_channels, kernel_size=64, stride=64):
        super(FeatureIntegrator, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bandPass =  BandPassSimpleMA(sr, res_scale=0.3)

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

#######滑动窗口技术###########################################################

#####因果多头注意力模块（大队长3.2CT_MSA）######
class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=2, window_size=1, qkv_bias=False, qk_scale=None, dropout=0., causal=True,
                 device=None):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.num_heads = heads
        self.causal = causal
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

        # 注册mask为buffer，自动跟随模型设备迁移
        self.register_buffer("mask", torch.tril(torch.ones(window_size, window_size)))

    def forward(self, x):
        # (b*n, t, c)
        B_prev, T_prev, C_prev = x.shape

        if self.window_size > 0:
            x = x.reshape(-1, self.window_size, C_prev)

        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.causal:
            # 这里使用buffer中的mask，无需额外调用to(x.device)
            attn = attn.masked_fill_(self.mask == 0, float("-inf"))

        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.window_size > 0:
            x = x.reshape(B_prev, T_prev, C_prev)
        return x

# Pre Normalization in Transformer
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# FFN in Transformer
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class CT_MSA(nn.Module):
    # Causal Temporal MSA
    def __init__(self,
                 dim,  # hidden dim
                 depth,  # the number of MSA in CT-MSA
                 heads,  # the number of heads
                 window_size,  # the size of local window
                 mlp_dim,  # mlp dimension
                 num_time,  # the number of time slot
                 dropout=0.,  # dropout rate
                 device=None):  # device, e.g., cuda
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_time, dim))
        self.layers = nn.ModuleList([])
        # 设置1层temporal attention即可
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                TemporalAttention(dim=dim,
                                  heads=heads,
                                  window_size=window_size,
                                  dropout=dropout,
                                  device=device),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        # x: (b, c, n, t)
        b, c, n, t = x.shape
        # 对输入x进行reshape,转换成便于计算的shape: (b, c, n, t) --> (b,n,t,c) --> (b*n,t,c)
        x = x.permute(0, 2, 3, 1).reshape(b*n, t, c)
        # 为输入x添加位置编码: (b*n,t,c) + (1,t,c) = (b*n,t,c)
        x = x + self.pos_embedding

        # 执行注意力机制 和 前馈神经网络层
        for attn, ff in self.layers:
            x = attn(x) + x   # 执行注意力机制并添加残差连接: (b*n,t,c) + (b*n,t,c) = (b*n,t,c)
            x = ff(x) + x   # 执行前馈神经网络并添加残差连接: (b*n,t,c) + (b*n,t,c) = (b*n,t,c)
        x = x.reshape(b, n, t, c).permute(0, 3, 1, 2) #  (b*n,t,c)--reshape-->(b,n,t,c)--permute-->(b,c,n,t)
        return x

###简化版本频域提取########################
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
###########################################

####频谱细节提取##########################
'''RGB版本四通道'''

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
        # 第一层降维并提取局部特征
        self.conv1 = nn.Conv2d(in_channels=4*channels, out_channels=channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        # 第二层空洞卷积扩大感受野
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels,
                               kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.act   = nn.GELU()

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

        # 6) CNN 提取特征
        y = self.conv1(rgb)    # (B, C, F, T_f)
        y = self.bn1(y)
        y = self.act(y)

        y = self.conv2(y)      # (B, C, F, T_f)
        y = self.bn2(y)
        y = self.act(y)

        # 7) 沿频率维度池化 → (B, C, T_f)
        y = y.mean(dim=2)

        # 8) 插值对齐到原始窗口长度 T
        if T_f != T:
            y = F.interpolate(y, size=T, mode='linear', align_corners=False)

        return y  # (B, C, T)

######################################

'''实验6：在实验5的基础上，将B1与B2串联，并且去除OneXOneConv（第18个epoch才让acc=1.00）'''
###########################################
# 修改后的 SlidingWindowProcessor
# 外层依然使用滑动窗口，每个窗口分为两个支路：
#   支路 A：标准多头注意力 + TCN_block（输出形状 (B, 32, window_size)）
#   支路 B：串联：简化版频域特征提取 -> 因果多头注意力模块（CT_MSA）
# 最后将支路 A 与支路 B 的输出在通道维度上拼接后融合
###########################################
'''串联'''
class SlidingWindowProcessor(nn.Module):
    def __init__(self, model_dim, num_heads, window_size, stride, dropout=0.):
        super(SlidingWindowProcessor, self).__init__()
        self.window_size = window_size
        self.stride = stride
        # 每个窗口归一化，调整为 (B, window_size, model_dim)
        self.layer_norm = nn.LayerNorm([window_size, model_dim])

        #### 支路 A：标准多头注意力 + TCN_block
        self.standard_attn = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm_std = nn.LayerNorm(model_dim)
        self.tcn_block = TemporalConvBlock(in_channels=model_dim, out_channels=model_dim)  # 输出通道 32

        #### 支路 B：串联简化版频域特征提取 -> 因果多头注意力模块
        self.freq_branch = FrequencyBranchSimp(n_fft=64, hop_length=32, win_length=64)
        self.ct_msa = CT_MSA(
            dim=model_dim,
            depth=1,
            heads=num_heads,
            window_size=window_size,
            mlp_dim=2 * model_dim,
            num_time=window_size,
            dropout=dropout
        )
        self.highpass    = FrequencyDetailBranch(channels=model_dim)

        self.layer_norm_ct = nn.LayerNorm(model_dim)

        #### 融合层：融合支路 A 与支路 B
        # 支路 A 输出 (B, 32, window_size)，支路 B 输出 (B, model_dim, window_size)
        self.fuse_linear = nn.Linear(2*model_dim, model_dim)
        self.final_norm = nn.LayerNorm(model_dim)
        # 最后融合所有窗口输出的卷积层（保持原有设计），输出通道 32
        self.fusion_conv = nn.Conv1d(model_dim, model_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        输入 x: (B, model_dim, L)
        输出: 融合后的特征，形状 (B, 32, fused_length)
        """
        batch_size, model_dim, length = x.shape
        window_outputs = []

        for window_start in range(0, length - self.window_size + 1, self.stride):
            window_end = window_start + self.window_size
            # 分割窗口，形状 (B, model_dim, window_size)
            window = x[:, :, window_start:window_end]
            # 调整为 (B, window_size, model_dim) 便于归一化
            window_norm = window.permute(0, 2, 1)
            window_norm = self.layer_norm(window_norm)

            #### 支路 A：标准多头注意力 + TCN_block
            std_attn_out, _ = self.standard_attn(window_norm, window_norm, window_norm)
            std_attn_out = self.layer_norm_std(std_attn_out + window_norm)  # (B, window_size, model_dim)
            branchA = self.tcn_block(std_attn_out.permute(0, 2, 1))  # (B, 32, window_size)

            #### 支路 B：串联简化版频域特征提取 -> 因果多头注意力模块
            # 统一输入形状 (B, model_dim, window_size)
            window_for_B = window  # (B, model_dim, window_size)
            branchB_inter_f = self.freq_branch(window_for_B)  # (B, model_dim, window_size)
            branchB_inter_h = self.highpass(branchB_inter_f)
            branchB_inter_f = self.layer_norm_ct(branchB_inter_f.permute(0, 2, 1)).permute(0, 2, 1)
            branchB_inter_h = self.layer_norm_ct(branchB_inter_h.permute(0, 2, 1)).permute(0, 2, 1)
            branchB_inter = branchB_inter_h * 0.03 + branchB_inter_f
            branchB_inter = self.layer_norm_ct(branchB_inter.permute(0, 2, 1)).permute(0, 2, 1)
            # branchB_inter = F.gelu(branchB_inter)
            # 调整形状供 CT_MSA 使用：CT_MSA 需要 (B, model_dim, 1, window_size)
            branchB_inter = branchB_inter.unsqueeze(2)  # (B, model_dim, 1, window_size)
            ct_out = self.ct_msa(branchB_inter)  # (B, model_dim, 1, window_size)
            ct_out = ct_out.squeeze(2).permute(0, 2, 1)  # (B, window_size, model_dim)
            ct_out = self.layer_norm_ct(ct_out + window_norm)  # 残差连接
            branchB_out = ct_out.permute(0, 2, 1)  # (B, model_dim, window_size)

            #### 融合支路 A 与支路 B
            fused = torch.cat([branchA, branchB_out], dim=1)  # (B, 32 + model_dim, window_size)
            fused = fused.permute(0, 2, 1)  # (B, window_size, 32 + model_dim)
            fused = self.fuse_linear(fused)  # (B, window_size, model_dim)
            window_outputs.append(fused)

        # 堆叠所有窗口，得到 (B, window_size, num_windows, model_dim)
        stacked_outputs = torch.stack(window_outputs, dim=2) #(1,100,23,32)
        # 调整形状以适应最终融合卷积层
        # 假设支路A（TCN_block）输出通道为 32，reshape 为 (B, 32, -1)
        stacked_outputs = stacked_outputs.permute(0, 1, 3, 2).reshape(batch_size, model_dim, -1) #(1,32,2300)
        fused_output = self.fusion_conv(stacked_outputs)
        return fused_output

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

    idx_graph = [2, 2, 2, 2, 1, 2, 2, 3, 4, 5, 2, 3, 1, 1]
    print("重新设计后的 idx_graph:", idx_graph)

    # 将模型与数据移动到 DEVICE 上（例如 'cuda' 或 'cpu'）
    model = TFDEEG(num_classes=num_classes, input_size=input_size, sampling_rate=sampling_rate,
                   num_T=num_T, out_graph=out_graph, dropout_rate=dropout_rate,
                   pool=pool, pool_step_rate=pool_step_rate, idx_graph=idx_graph)
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
        model.Tception1, model.Tception2, model.Tception3,
        model.feature_integrator, model.sliding_window_processor,
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

####### MoCE ##################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LorentzManifold(nn.Module):
    """
    Lorentz model

    x=(x0,x1,...,xd)

    -x0^2+x1^2+...+xd^2=-K

    K >0 represents |curvature|
    """

    def __init__(
            self,
            dim,
            init_curvature=1.0,
            learnable=True
    ):
        super().__init__()

        self.dim = dim

        if learnable:

            self.log_k = nn.Parameter(
                torch.log(
                    torch.tensor(
                        init_curvature
                    )
                )
            )

        else:

            self.register_buffer(
                "log_k",
                torch.log(
                    torch.tensor(
                        init_curvature
                    )
                )
            )

    @property
    def k(self):

        return torch.exp(self.log_k)

    def proj(self, x):

        """
        Project Euclidean vector
        to Lorentz manifold

        input:
            (...,d)

        output:
            (...,d+1)

        """

        k = self.k

        x_norm = torch.sum(
            x * x,
            dim=-1,
            keepdim=True
        )

        time = torch.sqrt(
            x_norm + k
        )

        return torch.cat(
            [
                time,
                x
            ],
            dim=-1
        )

    def logmap0(self, x):

        """
        logarithmic map
        Lorentz -> tangent
        """

        spatial = x[..., 1:]

        return spatial

    def expmap0(self, v):

        """
        tangent -> Lorentz

        """

        return self.proj(v)

    def inner(self, x, y):

        """
        Lorentz inner product

        """

        time = -x[..., 0] * y[..., 0]

        space = torch.sum(
            x[..., 1:] * y[..., 1:],
            dim=-1
        )

        return time + space

    def distance(self, x, y):

        """
        Hyperbolic distance

        """

        z = -self.inner(x, y) / self.k

        z = torch.clamp(
            z,
            min=1 + 1e-5
        )

        return torch.acosh(z)

    def frechet_mean(self, x, weight=None, iterations=5):
        if weight is None:
            weight = torch.ones(x.shape[1], device=x.device)

        weight = weight / weight.sum()

        mean = torch.sum(x * weight.view(1, -1, 1), dim=1)

        time = torch.sqrt(
            torch.sum(mean[..., 1:] ** 2, dim=-1, keepdim=True) + self.k
        )

        mean = torch.cat([time, mean[..., 1:]], dim=-1)
        return mean

class EEGEncoder(nn.Module):
    """
    Input:

        (B,1,32,800)

    Output:

        (B,feature_dim)

    """

    def __init__(
            self,
            channels=32,
            feature_dim=32
    ):
        super().__init__()

        self.conv = torch.nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=(1, 25),
                padding=(0, 12)
            ),

            nn.BatchNorm2d(16),

            nn.ELU(),

            nn.Conv2d(
                16,
                channels,
                kernel_size=(channels, 1)
            ),

            nn.BatchNorm2d(channels),

            nn.ELU(),

            nn.AvgPool2d(
                kernel_size=(1, 8)
            )

        )

        self.fc = nn.Sequential(

            nn.Linear(
                2250,
                128
            ),

            nn.ELU(),

            nn.Dropout(0.25),

            nn.Linear(
                128,
                feature_dim
            )

        )

    def forward(self, x):
        """
        x:

        B,1,32,800

        """

        x = self.conv(x)

        x = x.flatten(
            start_dim=1
        )

        x = self.fc(x)

        return x

class CurvatureAwareAttention(nn.Module):
    """
    MoCE core module

    Each brain region attends to
    other regions only.

    """

    def __init__(
            self,
            dim,
            num_regions=14,
            heads=4
    ):
        super().__init__()

        self.num_regions = num_regions
        self.dim = dim

        self.heads = heads

        self.q = nn.Linear(
            dim,
            dim
        )

        self.k = nn.Linear(
            dim,
            dim
        )

        self.v = nn.Linear(
            dim,
            dim
        )

        self.temperature = nn.Parameter(
            torch.tensor(1.0)
        )

    def forward(
            self,
            x,
            manifold
    ):
        """

        x:

        (B,N,D+1)

        """

        B, N, _ = x.shape

        # remove Lorentz time coordinate

        spatial = x[..., 1:]

        Q = self.q(
            spatial
        )

        K = self.k(
            spatial
        )

        V = self.v(
            spatial
        )

        score = torch.matmul(
            Q,
            K.transpose(-1, -2)
        )

        score = score / self.temperature

        # ==================================================
        # remove self attention
        # modality attends others only
        # ==================================================

        mask = torch.eye(
            N,
            device=x.device
        )

        score = score - mask * 1e9

        attn = torch.softmax(
            score,
            dim=-1
        )

        out = torch.matmul(
            attn,
            V
        )

        # restore Lorentz coordinate

        out = manifold.expmap0(
            out
        )

        return out, attn

class MoCEFusion(nn.Module):
    """
    Mixture of Curvature Experts fusion

    """

    def __init__(
            self,
            num_regions=14,
            hyper_dim=32
    ):
        super().__init__()

        self.num_regions = num_regions

        self.manifold = LorentzManifold(
            dim=hyper_dim
        )

        self.attention = CurvatureAwareAttention(
            dim=hyper_dim,
            num_regions=num_regions
        )

    def forward(self, x):
        """

        x:

        (B,14,33)

        """

        x, attn = self.attention(
            x,
            self.manifold
        )

        # Frechet mean fusion

        fused = self.manifold.frechet_mean(
            x
        )

        return fused, attn


class LorentzClassifier(nn.Module):

    def __init__(
            self,
            dim,
            num_classes
    ):
        super().__init__()

        # dim这里传入的是Lorentz维度
        # 去掉time coordinate后:
        # dim-1个feature

        self.fc = nn.Linear(
            dim - 1,
            num_classes
        )

    def forward(self, x):
        # Lorentz:
        # x=(time, spatial)

        spatial = x[..., 1:]

        return self.fc(spatial)


class HyperbolicExpert(nn.Module):
    """
    One brain region expert

    Euclidean feature

            |
            v

    Lorentz embedding


    Each region owns one curvature
    """

    def __init__(
            self,
            input_dim,
            hyper_dim=32,
            curvature=1.0
    ):
        super().__init__()

        self.fc = nn.Linear(
            input_dim,
            hyper_dim
        )

        self.manifold = LorentzManifold(
            dim=hyper_dim,
            init_curvature=curvature,
            learnable=True
        )

    def forward(self, x):
        """

        x:

        (B,input_dim)


        return:

        (B,hyper_dim+1)

        """

        x = self.fc(x)

        x = torch.tanh(x)

        x = self.manifold.expmap0(
            x
        )

        return x


class RegionExpertBank(nn.Module):
    """
    14 brain regions

    each region:

        EEGEncoder

             +

        Hyperbolic expert


    Output:

        (B,14,D+1)

    """

    def __init__(
            self,
            num_regions=14,
            channels=32,
            feature_dim=32,
            hyper_dim=32
    ):
        super().__init__()

        self.num_regions = num_regions

        self.encoders = nn.ModuleList(
            [
                EEGEncoder(
                    channels,
                    feature_dim
                )

                for _ in range(num_regions)
            ]
        )

        self.experts = nn.ModuleList(
            [
                HyperbolicExpert(
                    feature_dim,
                    hyper_dim
                )

                for _ in range(num_regions)
            ]
        )

    def forward(self, x):
        """

        x:

        (B,14,1,32,800)

        """

        outputs = []

        for i in range(self.num_regions):
            region_x = x[:, i].unsqueeze(1)

            feat = self.encoders[i](
                region_x
            )

            hyp = self.experts[i](
                feat
            )

            outputs.append(
                hyp
            )

        outputs = torch.stack(
            outputs,
            dim=1
        )

        return outputs


class MoCE(nn.Module):

    def __init__(
            self,
            num_regions=14,
            channels=32,
            feature_dim=32,
            hyper_dim=32,
            num_classes=2
    ):
        super().__init__()

        self.experts = RegionExpertBank(
            num_regions=num_regions,
            channels=channels,
            feature_dim=feature_dim,
            hyper_dim=hyper_dim
        )

        self.fusion = MoCEFusion(
            num_regions=num_regions,
            hyper_dim=hyper_dim
        )

        self.classifier = LorentzClassifier(
            hyper_dim + 1,
            num_classes
        )

        self.last_aux = None

    def forward(self, x):
        """
        x:B,14,1,32,800
        """

        expert_features = self.experts(x)

        fused, attn = self.fusion(expert_features)

        logits = self.classifier(fused)

        # self.last_aux = {
        #     "expert_features":expert_features,
        #     "attention":attn,
        #     "fused":fused
        # }

        return logits


########### ATDGNN ############################################################


class ATDGNN(nn.Module):

    def temporal_learner(self, in_chan, out_chan, kernel, pool, pool_step_rate):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
            PowerLayer(dim=-1, length=pool, step=int(pool_step_rate * pool))
        )

    def __init__(self, num_classes, input_size, sampling_rate, num_T,
                 out_graph, dropout_rate, pool, pool_step_rate, idx_graph):
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
        self.num_heads = 31
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
        self.Tception1 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[0] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception2 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[1] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception3 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[2] * sampling_rate)),
                                               self.pool, pool_step_rate)
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
        self.feature_integrator = FeatureIntegrator1(in_channels=self.channel, out_channels=self.channel)
        self.sliding_window_processor = SlidingWindowProcessor1(model_dim=self.channel, num_heads=self.num_heads,
                                                               window_size=self.window_size, stride=self.stride)
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

        # Dynamic Graph Convolution Layers
        # self.dynamic_gcn = DynamicGraphConvolution(size[-1], out_graph)
        self.dynamic_gcn = StackedDynamicGraphConvolution(size[-1], hidden_features, out_graph, num_layers=3)
        # 表示全局邻接矩阵。它被定义为浮点型张量，并设置为需要梯度计算（requires_grad=True）
        self.global_adj = nn.Parameter(torch.FloatTensor(self.brain_area, self.brain_area), requires_grad=True)
        # 根据给定的张量的形状和分布进行参数初始化。用来对global_adj进行初始化，采用的是Xavier均匀分布初始化方法。
        nn.init.xavier_uniform_(self.global_adj)
        # to be used after local graph embedding
        self.bn = nn.BatchNorm1d(self.brain_area)
        self.bn_ = nn.BatchNorm1d(self.brain_area)

        # Fully connected layer for classification
        self.fc = nn.Sequential(  # 组合神经网络模块
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(self.brain_area * out_graph), num_classes)
        )

    def get_size_temporal(self, input_size):
        # input_size: frequency x channel x data point
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        z = self.Tception1(data)
        out = z
        z = self.Tception2(data)
        out = torch.cat((out, z), dim=-1)
        z = self.Tception3(data)
        out = torch.cat((out, z), dim=-1)
        #######################################
        out = self.feature_integrator(out)  # 特征整合和降维
        out = self.sliding_window_processor(out)  # 滑动窗口处理
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
        # Temporal convolution
        y = self.Tception1(x)
        out = y
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        ##############################
        out = self.feature_integrator(out)  # 特征整合和降维
        out = self.sliding_window_processor(out)  # 滑动窗口处理
        ##############################
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        out = self.local_filter_fun(out, self.local_filter_weight)
        out = self.aggregate.forward(out)
        out = self.bn(out)
        out = self.dynamic_gcn(out)
        out = self.bn_(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out


class DynamicGraphConvolution(GraphConvolution):
    """
    Dynamic Graph Convolution Layer.
    Extends the GraphConvolution layer with a dynamic adjacency matrix based on feature similarity.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(DynamicGraphConvolution, self).__init__(in_features, out_features, bias)

    def forward(self, x, adj=None):
        if adj is None:
            # Compute adjacency matrix dynamically based on feature similarity, for example:
            adj = self.normalize_adjacency_matrix(x)

        output = torch.matmul(x, self.weight)
        if self.bias is not None:
            output += self.bias
        output = F.relu(torch.matmul(adj, output))
        return output

    def compute_similarity(self, x):
        # x: b, node, feature
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)
        return s

    def normalize_adjacency_matrix(self, x):
        """
        x：输入的特征矩阵，大小为(b, node, feature)，其中b为批次大小，node为节点数目，feature为每个节点的特征向量维度。
        self_loop：一个布尔值，表示是否在邻接矩阵中加入自环（自己到自己的连接）。
        """
        # x: b, node, feature
        # 利用模型中的self_similarity方法计算输入特征矩阵x的自相似度矩阵。结果为一个大小为(b, n, n)的张量，其中n为节点数目
        adj = self.compute_similarity(x)  # b, n, n
        num_nodes = adj.shape[-1]
        adj = adj + torch.eye(num_nodes).to(DEVICE)
        rowsum = torch.sum(adj, dim=-1)
        # 创建一个与rowsum大小相同的全零张量mask，并将rowsum中和为0的位置置为1。这一步是为了处理邻接矩阵中存在度为0的节点，避免除以0的错误
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        # 将mask添加到rowsum中，实现对邻接矩阵的修正。避免除以0的错误，并保证每个节点的度至少为1
        rowsum += mask
        # 计算度矩阵的逆平方根
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        # 将逆平方根得到的张量转换为对角矩阵
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        # 通过矩阵乘法和广播机制，将度矩阵的逆平方根与邻接矩阵相乘，得到归一化后的邻接矩阵
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj


class StackedDynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=3, bias=True):
        super(StackedDynamicGraphConvolution, self).__init__()
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(DynamicGraphConvolution(in_features, hidden_features, bias=bias))
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(DynamicGraphConvolution(hidden_features, hidden_features, bias=bias))
        # Last layer
        self.layers.append(DynamicGraphConvolution(hidden_features, out_features, bias=bias))

    def forward(self, x, adj=None):
        for layer in self.layers:
            x = layer(x, adj)
        return x


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))


class FeatureIntegrator1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=64, stride=64):
        super(FeatureIntegrator1, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        # 假设输入x的形状为 (batch_size, feature_dim, channels, length)
        batch_size, feature_dim, channels, length = x.size()

        # 你想将feature和length维度相结合
        # 首先，将x变形为 (batch_size, channels, feature_dim * length)
        x = x.reshape(batch_size, channels, feature_dim * length)

        # 然后，应用1D卷积
        x = self.conv(x)  # 卷积后的形状为 (batch_size, out_channels, new_length)

        return x


class SlidingWindowProcessor1(nn.Module):
    def __init__(self, model_dim, num_heads, window_size, stride):
        super(SlidingWindowProcessor1, self).__init__()
        self.window_size = window_size
        self.stride = stride
        self.layer_norm1 = nn.LayerNorm([window_size, model_dim])
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=model_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.tcn_block = TemporalConvBlock(in_channels=model_dim, out_channels=model_dim)
        # 定义融合层，使用1D卷积以保留32通道的结构，卷积核大小和步长可以根据需要调整
        self.fusion_conv = nn.Conv1d(model_dim, model_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        batch_size, channel, length = x.shape
        # 使用列表收集所有窗口的输出
        window_outputs = []

        for window_start in range(0, length - self.window_size + 1, self.stride):
            window_end = window_start + self.window_size
            window = x[:, :, window_start:window_end]
            window = window.permute(0, 2, 1)

            window = self.layer_norm1(window)
            attn_output, _ = self.multi_head_attention(window, window, window)
            attn_output = self.layer_norm2(attn_output + window)
            tcn_input = attn_output.permute(0, 2, 1)
            tcn_output = self.tcn_block(tcn_input)

            window_outputs.append(tcn_output)

        # 将所有窗口的输出沿着时间维度堆叠起来，形成一个新的维度
        stacked_outputs = torch.stack(window_outputs, dim=2)
        # 重新排列维度以匹配卷积层的输入要求
        stacked_outputs = stacked_outputs.permute(0, 3, 1, 2).reshape(batch_size, channel, -1)
        # 通过融合层整合所有窗口的输出
        fused_output = self.fusion_conv(stacked_outputs)

        return fused_output



######## CNDNet ###############################################################################


import torch
from torch import nn
import torch.nn.functional as F


class WeightedSpatialAttention(nn.Module):
    """
    Weighted spatial attention from CNDNet.

    Input:
        x: [B, C, T]
    Output:
        x_attn: [B, C, T]
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.theta1 = nn.Parameter(torch.zeros(1))
        self.theta2 = nn.Parameter(torch.zeros(1))
        self.conv = nn.Conv1d(
            2, 1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x):
        f_avg = x.mean(dim=1, keepdim=True)
        f_max = x.amax(dim=1, keepdim=True)

        weights = torch.softmax(
            torch.cat([self.theta1, self.theta2]), dim=0
        ) * 5.0
        w1, w2 = weights[0], weights[1]

        attn_input = torch.cat([w1 * f_avg, w2 * f_max], dim=1)
        mask = torch.sigmoid(self.conv(attn_input))
        return x * mask


class BSplineKANLayer(nn.Module):
    """
    Lightweight self-contained cubic B-spline KAN layer.
    """
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        grid_min=-1.0,
        grid_max=1.0,
    ):
        super().__init__()

        base_grid = torch.linspace(
            grid_min, grid_max, grid_size + 1, dtype=torch.float32
        )
        knots = torch.cat([
            base_grid[0].repeat(spline_order),
            base_grid,
            base_grid[-1].repeat(spline_order),
        ])
        self.register_buffer("knots", knots)

        self.spline_order = spline_order
        self.num_basis = knots.numel() - spline_order - 1

        self.omega = nn.Parameter(torch.ones(out_features, in_features))
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, self.num_basis)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.02)
        nn.init.ones_(self.omega)

    def _bspline_basis(self, x):
        x = torch.clamp(x, self.knots[0], self.knots[-1])

        basis = (
            (x.unsqueeze(-1) >= self.knots[:-1])
            & (x.unsqueeze(-1) < self.knots[1:])
        ).to(x.dtype)

        # Include right boundary.
        basis[..., -1] = torch.where(
            x == self.knots[-1],
            torch.ones_like(basis[..., -1]),
            basis[..., -1],
        )

        for degree in range(1, self.spline_order + 1):
            left_den = self.knots[degree:-1] - self.knots[:-degree-1]
            right_den = self.knots[degree+1:] - self.knots[1:-degree]

            left = (
                (x.unsqueeze(-1) - self.knots[:-degree-1])
                / (left_den + 1e-8)
            ) * basis[..., :-1]

            right = (
                (self.knots[degree+1:] - x.unsqueeze(-1))
                / (right_den + 1e-8)
            ) * basis[..., 1:]

            basis = left + right

        return basis

    def forward(self, x):
        x_grid = torch.tanh(x)

        silu_edge = (
            F.silu(x_grid).unsqueeze(1)
            * self.base_weight.unsqueeze(0)
        )

        basis = self._bspline_basis(x_grid)
        spline_edge = torch.einsum(
            "bdk,odk->bod",
            basis,
            self.spline_weight,
        )

        return (
            self.omega.unsqueeze(0)
            * (silu_edge + spline_edge)
        ).sum(dim=-1)


class KAN(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.5):
        super().__init__()
        self.kan = BSplineKANLayer(
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.kan(x))


class FeatureExtractor(nn.Module):
    """
    CNDNet-inspired feature extractor for [B, 32, 1000] EEG input.
    """
    def __init__(self, in_channels=32, dropout=0.5, kernel_size=7):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.AvgPool1d(2),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.AvgPool1d(2),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True),
            nn.AvgPool1d(2),
        )

        self.dropout = nn.Dropout(dropout)
        self.attention = WeightedSpatialAttention(kernel_size)

        self.block4 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.AvgPool1d(2),
        )

        # Fixed feature size so the classifier is independent of the exact
        # temporal length after convolution/pooling.
        self.adaptive_pool = nn.AdaptiveAvgPool1d(4)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"Expected input [B, C, T], got shape {tuple(x.shape)}"
            )

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.dropout(x)
        x = self.attention(x)
        x = self.block4(x)
        x = self.adaptive_pool(x)
        return x  # [B, 64, 4]


class CategoryClassifier(nn.Module):
    """Single classification head."""
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()

        feature_dim = 64 * 4

        self.gn = nn.GroupNorm(8, 64)
        self.linear = nn.Linear(feature_dim, 16)

        self.kan1 = KAN(feature_dim, 64, dropout=dropout)
        self.kan2 = KAN(64, 16, dropout=dropout)

        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        x = self.gn(x)
        x = x.flatten(1)

        branch_linear = self.linear(x)
        branch_kan = self.kan2(self.kan1(x))

        x = torch.cat([branch_linear, branch_kan], dim=1)
        return self.fc(x)


class CNDNet(nn.Module):
    """
    Simplified CNDNet for Bath visual-imagery EEG classification.

    Input:
        x: [B, 32, 1000]

    Output:
        logits: [B, num_classes]

    Usage is intentionally the same style as EEG-Deformer:
        model = CNDNet(num_chan=32, num_time=1000, num_classes=3)
        logits = model(data)
    """
    def __init__(
        self,
        num_chan=32,
        num_time=1000,
        num_classes=2,
        dropout=0.5,
    ):
        super().__init__()
        self.num_chan = num_chan
        self.num_time = num_time
        self.num_classes = num_classes

        self.feature_extractor = FeatureExtractor(
            in_channels=num_chan,
            dropout=dropout,
            kernel_size=7,
        )
        self.classifier = CategoryClassifier(
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x):
        """Return class logits only."""
        x = torch.squeeze(x, dim = 1)
        x = self.feature_extractor(x)
        logits = self.classifier(x)
        return logits
