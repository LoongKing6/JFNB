import os

import numpy as np

from config.config import set_config
from train.distillation import train_global_expert
from train.era_geometry import generate_subject_geometries
from train.expert_data import parse_subject_list
from train.expert_training import train_individual_experts
from train.shared_geometry import build_shared_geometry
from utils.utils import seed_all

_, os.environ['CUDA_VISIBLE_DEVICES'] = set_config([])

if __name__ == '__main__':
    args, _ = set_config()
    sub_to_run = np.asarray(parse_subject_list(args), dtype=np.int64)
    seed_all(args.random_seed)

    if args.mode == 'individual':
        if args.prepare_data:
            from train.prepare_data import PrepareData
            PrepareData(args).run(sub_to_run, split=True, expand=True)
        train_individual_experts(args)
    elif args.mode == 'geometry':
        generate_subject_geometries(args)
        path, _ = build_shared_geometry(args)
        print('Shared ERA geometry saved:', path)
    elif args.mode == 'global':
        train_global_expert(args)
    else:
        from train.prepare_data import PrepareData
        from train.cross_validation import CrossValidation
        pd = PrepareData(args)
        pd.run(sub_to_run, split=True, expand=True)
        if args.train_method == "n_fold" and args.public_data and args.dataset == 'EMO':
            pd.build_public_preprocessed_data(sub_to_run, split=True, expand=True)
        cv = CrossValidation(args)
        if args.train_method == "n_fold" and args.public_data and args.dataset == 'EMO':
            cv.public_n_fold_CV(fold=args.fold, reproduce=args.reproduce)
        elif args.train_method == "n_fold":
            cv.n_fold_CV(subject=sub_to_run, fold=args.fold, reproduce=args.reproduce)
        if args.train_method == "loso":
            cv.loso_CV(subjects=sub_to_run, reproduce=args.reproduce)
        if args.train_method == "ERA":
            cv.era_inference(subjects=sub_to_run)
