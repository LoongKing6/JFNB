from config.config import *
from train.cross_validation import *
from train.prepare_data import *
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

if __name__ == '__main__':
    args, _ = set_config()
    sub_to_run = np.arange(args.subjects)
    pd = PrepareData(args)
    pd.run(sub_to_run, split=True, expand=True)
    cv = CrossValidation(args)
    seed_all(args.random_seed)

    if args.train_method == "n_fold":
        cv.n_fold_CV(subject=sub_to_run, fold=args.fold, reproduce=args.reproduce)
    if args.train_method == "loso":
        cv.loso_CV(subjects=sub_to_run, reproduce=args.reproduce)
