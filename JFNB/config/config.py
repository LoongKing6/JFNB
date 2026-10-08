import argparse

#'/home/data/eeg-data'
def set_config():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset', type=str, default='EMO', choices=['EMO', 'DEAP', 'EEGMAT', 'ISRUC']) # EMO is RESB
    # parser.add_argument('--data-path', type=str, default='/home/hutao/缝好的/ISRUC数据集/RawData')
    parser.add_argument('--data-path', type=str, default='/home/hutao/缝好的/RESB数据集')
    # parser.add_argument('--data-path', type=str, default='/home/hutao/缝好的/RESB_Traditional')
    # parser.add_argument('--data-path', type=str, default='/home/hutao/缝好的/EEGMat')
    # parser.add_argument('--data-path', type=str, default=r'/home/hutao/研究生论文/缝好的/DEAP')
    parser.add_argument('--subjects', type=int, default=1)
    parser.add_argument('--num-class', type=int, default=2, choices=[2, 5, 6, 7]) # S is 6; T is 7; A,V,T_V,D,L is 2
    # A is Arousal; V is Motivation; T_V is Valence; D is State; L is Cognition; S is 6-class; T is 7-class
    parser.add_argument('--label-type', type=str, default='A', choices=['A', 'V', 'D', 'L', 'S', 'T_V', 'T', 'haha', 'Axis'])
    parser.add_argument('--train_method', type=str, default="n_fold", choices=["n_fold", "loso"])
    parser.add_argument('--segment', type=int, default=4)  # segment length in seconds
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--sampling-rate', type=int, default=2000, choices=[1000, 128, 2000])
    parser.add_argument('--target-rate', type=int, default=500, choices=[200 , 128, 500, 250])
    parser.add_argument('--trial-duration', type=int, default=6, help='trial duration in seconds')
    parser.add_argument('--input-shape', type=str, default="1,32,800")  # 输入形状 (1, 32, 512)
    parser.add_argument('--data-format', type=str, default='eeg')
    parser.add_argument('--bandpass', type=tuple, default=(1, 50))
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--denoise', type=bool, default=True, help='Enable regression denoising using resting state (EMO only)')
    parser.add_argument('--denoise-components', type=int, default=5, help='Number of PCA components for regression denoising')

    # Training Process
    parser.add_argument('--fold', type=int, default=10)
    parser.add_argument('--random-seed', type=int, default=3407)
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--patient', type=int, default=40)  # 早停 最开始为20
    parser.add_argument('--patient-cmb', type=int, default=10)  # 原始值为8
    parser.add_argument('--max-epoch-cmb', type=int, default=40)  # 最大迭代次数 原始值为20
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-3)  # 学习率 原始值为1e-3
    parser.add_argument('--training-rate', type=float, default=0.8)
    parser.add_argument('--weight-decay', type=float, default=0.001)  # 权重衰减
    parser.add_argument('--step-size', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)  # 原始0.5
    parser.add_argument('--LS', type=bool, default=True, help="Label smoothing")  # 原始值为True
    parser.add_argument('--LS-rate', type=float, default=0.1)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--balance', type=bool, default=False)

    #采集标签前后
    parser.add_argument('--pre_seconds', type=int, default=2)
    parser.add_argument('--post_seconds', type=int, default=2)

    # Deformer的参数
    parser.add_argument('--temporal_kernel', type=int, default=20)  # 要修改
    parser.add_argument('--AT', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=6)


    parser.add_argument('--save-path', default='./save/')
    parser.add_argument('--load-path', default='./save/max-acc.pth')
    parser.add_argument('--load-path-final', default='./save/final_model.pth')
    parser.add_argument('--save-model', type=bool, default=True)
    # Model Parameters
    parser.add_argument('--model', type=str, default='JFNB',
                        choices=['JFNB', 'LGGNet', 'EEGNet', 'DeepConvNet', 'ShallowConvNet', 'EEG-TCNet',
                                 'TSception', 'TCNet-Fusion', 'ATCNet', 'DGCNN', 'TCANet', 'SFSWTS', 'CCMTL',
                                 'CFBM', 'EmT', 'EEGDeformer', 'DBGCN', 'ACCNet', 'FG_HANet_Simple', 'TFDEEG',
                                 'MoCE', 'EMOD', 'CNDNet'])
    parser.add_argument('--pool', type=int, default=16)
    parser.add_argument('--pool-step-rate', type=float, default=0.25)
    parser.add_argument('--T', type=int, default=64)
    parser.add_argument('--graph-type', type=str, default='fro', choices=['fro', 'gen', 'hem','myD', 'BL'])# my is ER
    parser.add_argument('--hidden', type=int, default=32)  # 隐藏层

    # Reproduce the result using the saved model
    parser.add_argument('--reproduce', action='store_true', default=False)

    # t-SNE Visualization
    parser.add_argument('--tsne', action='store_true', default=False, help='Enable t-SNE visualization')
    parser.add_argument('--tsne-perplexity', type=float, default=30.0, help='t-SNE perplexity')
    parser.add_argument('--tsne-lr', type=float, default=200.0, help='t-SNE learning rate')
    parser.add_argument('--class-names', type=str, default=None, help='Comma-separated class names for t-SNE plot, e.g. "Negative,Positive"')

    args = parser.parse_args()
    gpu = args.gpu
    # Convert the input shape from string to tuple of integers
    args.input_shape = tuple(map(int, args.input_shape.split(',')))

    # Parse class names
    if args.class_names is not None:
        args.class_names = [s.strip() for s in args.class_names.split(',')]

    return args, gpu
