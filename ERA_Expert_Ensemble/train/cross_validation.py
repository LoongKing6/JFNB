
"""全加混淆矩阵"""
import copy
import datetime

import numpy as np
import torch

from config.config import *
from sklearn.model_selection import KFold
from train.train_model import *
from utils.utils import *
from train.expert_data import apply_normalization, normalization_from_checkpoint


ROOT = os.getcwd()
_, os.environ['CUDA_VISIBLE_DEVICES'] = set_config([])
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class CrossValidation:
    def __init__(self, args):
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        # Log the results per subject
        result_path = os.path.join(args.save_path, 'result')
        ensure_path(result_path)
        if args.train_method == 'ERA':
            self.text_file = os.path.join(result_path, "ERAresults_{}.txt".format(args.dataset))
        elif args.public_data and args.dataset == 'EMO':
            self.text_file = os.path.join(result_path, "public_results_{}.txt".format(args.dataset))
        else:
            self.text_file = os.path.join(result_path, "results_{}.txt".format(args.dataset))
        file = open(self.text_file, 'a')
        file.write("\n" + str(datetime.datetime.now()) +
                   "\nTrain:Parameter setting for " + str(args.model) + ' on ' + str(args.dataset) +
                   "\n1)number_class:" + str(args.num_class) +
                   "\n2)random_seed:" + str(args.random_seed) +
                   "\n3)learning_rate:" + str(args.learning_rate) +
                   "\n4)training_rate:" + str(args.training_rate) +
                   "\n5)pool:" + str(args.pool) +
                   "\n6)num_epochs:" + str(args.max_epoch) +
                   "\n7)batch_size:" + str(args.batch_size) +
                   "\n8)dropout:" + str(args.dropout) +
                   "\n9)hidden_node:" + str(args.hidden) +
                   "\n10)input_shape:" + str(args.input_shape) +
                   "\n11)class:" + str(args.label_type) +
                   "\n12)T:" + str(args.T) +
                   "\n13)graph-type:" + str(args.graph_type) +
                   "\n14)patient:" + str(args.patient) +
                   "\n15)patient-cmb:" + str(args.patient_cmb) +
                   "\n16)max-epoch-cmb:" + str(args.max_epoch_cmb) +
                   "\n17)fold:" + str(args.fold) +
                   "\n18)model:" + str(args.model) +
                   "\n19)data-path:" + str(args.data_path) +
                   "\n20)balance:" + str(args.balance) +
                   "\n21)bandpass:" + str(args.bandpass) +
                   "\n22)dataset:" + str(args.dataset) +
                    "\n23)overlap:" + str(args.overlap) +
                   "\n24)data_fuse:" + str(args.data_fuse) +
                   "\n25)public_data:" + str(args.public_data) +
                   '\n')
        file.close()

    def _normalize_single(self, data):
        data = data.copy()
        for channel in range(data.shape[2]):
            mean = np.mean(data[:, :, channel, :])
            std = np.std(data[:, :, channel, :])
            if std < 1e-8:
                std = 1.0
            data[:, :, channel, :] = (data[:, :, channel, :] - mean) / std
        return data

    def _format_era_center_lines(self, prefix, summary):
        lines = []
        for class_name, center in summary['class_centers'].items():
            lines.append('{} {} center -> E:{} R:{} count:{}'.format(
                prefix, class_name, center['mean_E'], center['mean_R'], center['count']))
        return lines

    def era_inference(self, subjects):
        """
        External ERA inference mode:
        no training, no validation, directly load final_model.pth and run forward inference.
        """
        if not os.path.exists(self.args.load_path_final):
            raise FileNotFoundError('final_model.pth not found: {}'.format(self.args.load_path_final))

        tta = []
        ttf = []
        ttk = []
        all_preds = []
        all_labels = []
        all_e_scores = []
        all_r_scores = []

        header = 'ERA inference on dataset {} using checkpoint {}'.format(
            self.args.dataset, self.args.load_path_final)
        print(header)
        self.log2txt(header)

        checkpoint = torch.load(self.args.load_path_final, map_location='cpu')
        checkpoint_normalization = None
        if isinstance(checkpoint, dict) and 'normalization_mean' in checkpoint:
            checkpoint_normalization = normalization_from_checkpoint(checkpoint)

        for sub in subjects:
            data_raw, label_raw = self.load_per_subject(sub)
            data_flat = self._flatten_segments(data_raw)
            label_flat = self._flatten_labels(label_raw)
            if checkpoint_normalization is not None:
                data_flat = apply_normalization(data_flat, *checkpoint_normalization)
            else:
                print('[WARN] Legacy checkpoint has no normalization statistics; using subject-wise normalization.')
                data_flat = self._normalize_single(data_flat)

            data_test = torch.from_numpy(data_flat).float()
            label_test = torch.from_numpy(label_flat).long()

            acc_test, pred, act, test_payload, test_era_summary = test(
                args=self.args, data=data_test, label=label_test,
                reproduce=False, subject=sub, fold=0)

            acc, f1, _, kappa = get_metrics(y_pred=pred, y_true=act)
            tta.append(acc)
            ttf.append(f1)
            ttk.append(kappa)
            all_preds.extend(pred)
            all_labels.extend(act)
            all_e_scores.extend(test_payload['E_score'])
            all_r_scores.extend(test_payload['R_score'])

            subject_result = 'sub {}: total test accuracy {}, f1: {}, kappa: {}, meanE: {}, meanR: {}'.format(
                sub, acc, f1, kappa, test_era_summary['mean_E'], test_era_summary['mean_R'])
            print(subject_result)
            self.log2txt(subject_result)
            for line in self._format_era_center_lines('sub {}:'.format(sub), test_era_summary):
                print(line)
                self.log2txt(line)

        tta = np.array(tta)
        ttf = np.array(ttf)
        ttk = np.array(ttk)
        mACC = np.mean(tta) if tta.size > 0 else 0.0
        mF1 = np.mean(ttf) if ttf.size > 0 else 0.0
        mKappa = np.mean(ttk) if ttk.size > 0 else 0.0
        stdACC = np.std(tta) if tta.size > 0 else 0.0
        stdKappa = np.std(ttk) if ttk.size > 0 else 0.0

        final_era_summary = summarize_era_distribution(all_e_scores, all_r_scores, all_labels, self.args.num_class)
        final_lines = [
            'Final: test mean ACC:{} std:{}'.format(mACC, stdACC),
            'Final: test mean F1:{}'.format(mF1),
            'Final: test mean Kappa:{} std:{}'.format(mKappa, stdKappa),
            'Final: test mean E:{}'.format(final_era_summary['mean_E']),
            'Final: test mean R:{}'.format(final_era_summary['mean_R'])
        ]
        final_lines.extend(self._format_era_center_lines('Final:', final_era_summary))

        for line in final_lines:
            print(line)
            self.log2txt(line)

        if len(all_preds) > 0:
            plot_confusion_matrix(args=self.args, y_true=all_labels, y_pred=all_preds)

    def load_per_subject(self, sub):
        """
        load data for sub
        param sub: which subject's data to load
        return: data and label
        """
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        sub_code = 'sub' + str(sub) + '.hdf'
        path = os.path.join(save_path, data_type, sub_code)
        dataset = h5py.File(path, 'r')
        data = np.array(dataset['data'])
        label = np.array(dataset['label'])
        print('>>> Data:{} Label:{}'.format(data.shape, label.shape))
        return data, label

    def load_public_dataset(self):
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        candidate_names = ['public_RESB_processed.hdf', 'public_RESB.hdf']
        for name in candidate_names:
            path = os.path.join(save_path, data_type, name)
            if os.path.exists(path):
                with h5py.File(path, 'r') as dataset:
                    data = np.array(dataset['data'])
                    label = np.array(dataset['label'])
                print('>>> Public Data:{} Label:{} Path:{}'.format(data.shape, label.shape, path))
                return data, label
        raise FileNotFoundError('Public HDF not found under {}'.format(os.path.join(save_path, data_type)))

    def prepare_data(self, idx_train, idx_test, data, label):
        """
        1. get training and testing data according to the index
        2. numpy.array-->torch.tensor
        param idx_train: index of training data
        param idx_test: index of testing data
        param data: (segments, 1, channel, data)
        param label: (segments,)
        return: data and label
        """
        data_train = data[idx_train]
        label_train = label[idx_train]
        data_test = data[idx_test]
        label_test = label[idx_test]

        # For DEAP we want to do trial-wise 10-fold, so the idx_train/idx_test is for trials.
        # data: (trial, segment, 1, chan, datapoint)
        # To use the normalization function, we should change the dimension from
        # (trial, segment, 1, chan, datapoint) to (trial*segments, 1, chan, datapoint)
        if data_train.ndim == 5:
            data_train = np.concatenate(data_train, axis=0)
            label_train = np.concatenate(label_train, axis=0)
        if data_test.ndim == 5:
            # When leave one trial out is conducted, the test data will be (segments, 1, chan, datapoint),
            # hence, no need to concatenate the first dimension to get trial*segments.
            data_test = np.concatenate(data_test, axis=0)
            label_test = np.concatenate(label_test, axis=0)

        data_train, data_test = self.normalize(train=data_train, test=data_test)
        # Prepare the data format for training the model using PyTorch
        data_train = torch.from_numpy(data_train).float()
        label_train = torch.from_numpy(label_train).long()

        data_test = torch.from_numpy(data_test).float()
        label_test = torch.from_numpy(label_test).long()
        return data_train, label_train, data_test, label_test

    def normalize(self, train, test):
        """
        this function do standard normalization for EEG channel by channel
        :param train: training data (sample, 1, chan, datapoint)
        :param test: testing data (sample, 1, chan, datapoint)
        :return: normalized training and testing data
        """
        # data: sample x 1 x channel x data
        for channel in range(train.shape[2]):
            mean = np.mean(train[:, :, channel, :])
            std = np.std(train[:, :, channel, :])
            train[:, :, channel, :] = (train[:, :, channel, :] - mean) / std
            test[:, :, channel, :] = (test[:, :, channel, :] - mean) / std
        return train, test

    def split_balance_class(self, data, label, train_rate, random):
        """
        Get the validation set using the same percentage of the two classe samples
        param data: training data (segment, 1, channel, data)
        param label: (segments,)
        param train_rate: the percentage of trianing data
        param random: bool, whether to shuffle the training data before get the validation data
        return: data_trian, label_train, and data_val, label_val
        """
        # Data dimension: segment x 1 x channel x data
        # Label dimension: segment x 1
        np.random.seed(0)
        # data : segments x 1 x channel x data
        # label : segments

        index_0 = np.where(label == 0)[0]
        index_1 = np.where(label == 1)[0]

        # for class 0
        index_random_0 = copy.deepcopy(index_0)

        # for class 1
        index_random_1 = copy.deepcopy(index_1)

        if random:
            np.random.shuffle(index_random_0)
            np.random.shuffle(index_random_1)

        index_train = np.concatenate((index_random_0[:int(len(index_random_0) * train_rate)],
                                      index_random_1[:int(len(index_random_1) * train_rate)]),
                                     axis=0)
        index_val = np.concatenate((index_random_0[int(len(index_random_0) * train_rate):],
                                    index_random_1[int(len(index_random_1) * train_rate):]),
                                   axis=0)

        # get validation
        val = data[index_val]
        val_label = label[index_val]

        train = data[index_train]
        train_label = label[index_train]

        return train, train_label, val, val_label

    def n_fold_CV(self, subject, fold, reproduce):
        """
        this function achieves n-fold cross-validation
        param subject: how many subjects to load
        param fold: how many fold.
        """
        # Train and evaluate the model subject by subject
        tta = []  # total test accuracy
        tva = []  # total validation accuracy
        ttf = []  # total test f1
        tvf = []  # total validation f1
        ttk = []  # test kappa
        tvk = []  # val kappa
        all_tsne_features = []  # 收集所有受试者的特征
        all_tsne_labels = []    # 收集所有受试者的标签

        all_preds = []
        all_labels = []
        all_e_scores = []
        all_r_scores = []

        for sub in subject:
            data, label = self.load_per_subject(sub)
            # 验证标签范围与分类数一致
            unique_labels = np.unique(label)
            assert len(unique_labels) <= self.args.num_class, \
                'Label mismatch: found {} unique labels {} but num_class={}'.format(
                    len(unique_labels), unique_labels, self.args.num_class)
            assert unique_labels.min() >= 0 and unique_labels.max() < self.args.num_class, \
                'Label values {} out of range [0, {})'.format(unique_labels, self.args.num_class)
            va_val = Averager()
            vf_val = Averager()
            vk_val = Averager()
            preds, acts = [], []
            sub_e_scores, sub_r_scores = [], []
            kf = KFold(n_splits=fold, shuffle=True)
            for idx_fold, (idx_train, idx_test) in enumerate(kf.split(data)):
                print('Outer loop: {}-fold-CV Fold:{}'.format(fold, idx_fold))
                data_train, label_train, data_test, label_test = self.prepare_data(
                    idx_train=idx_train, idx_test=idx_test, data=data, label=label)
                if self.args.balance:
                    data_train, label_train, data_val, label_val = self.split_balance_class(
                        data=data_train, label=label_train, train_rate=self.args.training_rate, random=True)

                if reproduce:
                    # to reproduce the reported ACC
                    acc_test, pred, act, test_payload, test_era_summary = test(
                        args=self.args, data=data_test, label=label_test,
                        reproduce=reproduce,
                        subject=sub, fold=idx_fold)
                    acc_val = 0
                    f1_val = 0
                    kappa_val = 0
                else:
                    # to train new models
                    print('Training:', data_train.size(), label_train.size())
                    print('Test:', data_test.size(), label_test.size())
                    acc_val, f1_val, kappa_val = self.first_stage(data=data_train, label=label_train,
                                                       subject=sub, fold=idx_fold)

                    combine_train(args=self.args,
                                  data_train=data_train, label_train=label_train,
                                  subject=sub, fold=idx_fold, target_acc=1)

                    acc_test, pred, act, test_payload, test_era_summary = test(
                        args=self.args, data=data_test, label=label_test,
                        reproduce=reproduce,
                        subject=sub, fold=idx_fold)
                # t-SNE 可视化
                if self.args.tsne:
                    test_loader = get_dataloader(data_test, label_test, self.args.batch_size)
                    # 加载该折训练好的模型
                    tsne_model = get_model(self.args)
                    if CUDA:
                        tsne_model = tsne_model.cuda()
                    if reproduce:
                        model_name_reproduce = 'sub' + str(sub) + '_fold' + str(idx_fold) + '.pth'
                        data_type = 'model_{}_{}'.format(self.args.data_format, self.args.label_type)
                        experiment_setting = 'T_{}_pool_{}'.format(self.args.T, self.args.pool)
                        load_path = os.path.join(self.args.save_path, experiment_setting, data_type, model_name_reproduce)
                    else:
                        load_path = self.args.load_path_final
                    load_model_checkpoint(tsne_model, load_path)
                    feat, lbl = extract_features(test_loader, tsne_model)
                    all_tsne_features.append(feat)
                    all_tsne_labels.append(lbl)
                va_val.add(acc_val)
                vf_val.add(f1_val)
                vk_val.add(kappa_val)
                preds.extend(pred)
                acts.extend(act)
                sub_e_scores.extend(test_payload['E_score'])
                sub_r_scores.extend(test_payload['R_score'])
                all_e_scores.extend(test_payload['E_score'])
                all_r_scores.extend(test_payload['R_score'])

            tva.append(va_val.item())
            tvf.append(vf_val.item())
            tvk.append(vk_val.item())
            acc, f1, _, kappa= get_metrics(y_pred=preds, y_true=acts)
            tta.append(acc)
            ttf.append(f1)
            ttk.append(kappa)
            all_preds.extend(preds)
            all_labels.extend(acts)
            sub_era_summary = summarize_era_distribution(sub_e_scores, sub_r_scores, acts, self.args.num_class)
            result = (
                'sub {}: total test accuracy {}, f1: {}, kappa: {}, meanE: {}, meanR: {}'
            ).format(sub, tta[-1], f1, kappa, sub_era_summary['mean_E'], sub_era_summary['mean_R'])
            print('sub {}: total test accuracy {}, f1: {}, kappa: {}, meanE: {}, meanR: {}'.format(
                sub, tta[-1], f1, kappa, sub_era_summary['mean_E'], sub_era_summary['mean_R']))
            print_era_class_centers('sub {}:'.format(sub), sub_era_summary)
            self.log2txt(result)
            self.log2txt('sub {} ERA centers: {}'.format(sub, sub_era_summary['class_centers']))

        # prepare final report
        tta = np.array(tta)
        ttf = np.array(ttf)
        tva = np.array(tva)
        tvf = np.array(tvf)
        ttk = np.array(ttk)
        tvk = np.array(tvk)
        mACC = np.mean(tta)
        mF1 = np.mean(ttf)
        mKappa = np.mean(ttk)
        std = np.std(tta)
        stdF1 = np.std(ttf)
        stdKappa = np.std(ttk)
        mACC_val = np.mean(tva)
        std_val = np.std(tva)
        mF1_val = np.mean(tvf)
        mKappa_val = np.mean(tvk)

        print('Final: test mean ACC:{} std:{}'.format(mACC, std))
        print('Final: test mean F1:{}'.format(mF1))
        print('Final: test mean Kappa:{} std:{}'.format(mKappa, stdKappa))
        print('Final: val mean ACC:{} std:{}'.format(mACC_val, std_val))
        print('Final: val mean F1:{}'.format(mF1_val))
        print('Final: val mean Kappa:{}'.format(mKappa_val))
        final_era_summary = summarize_era_distribution(all_e_scores, all_r_scores, all_labels, self.args.num_class)
        print('Final: test mean E:{}'.format(final_era_summary['mean_E']))
        print('Final: test mean R:{}'.format(final_era_summary['mean_R']))
        for class_name, center in final_era_summary['class_centers'].items():
            print('Final: {} center -> E:{} R:{} count:{}'.format(
                class_name, center['mean_E'], center['mean_R'], center['count']))
        results = ('test mAcc={} std:{} mF1={} std={} mKappa={} stdKappa={}\n'
                   'val mAcc={} F1={} mKappa={}\n'
                   'ERA meanE={} meanR={} centers={}').format(
            mACC, std, mF1, stdF1, mKappa, stdKappa,
            mACC_val, mF1_val, mKappa_val,
            final_era_summary['mean_E'], final_era_summary['mean_R'],
            final_era_summary['class_centers'])
        self.log2txt(results)

        # 所有受试者特征收集完毕后，统一进行 t-SNE 可视化
        if self.args.tsne and len(all_tsne_features) > 0:
            tsne_visualize(args=self.args, all_features=all_tsne_features, all_labels=all_tsne_labels)

        # 混淆矩阵可视化
        if len(all_preds) > 0:
            plot_confusion_matrix(args=self.args, y_true=all_labels, y_pred=all_preds)

    def public_n_fold_CV(self, fold, reproduce):
        """
        n-fold cross-validation for the public/common expert dataset.
        """
        data, label = self.load_public_dataset()
        unique_labels = np.unique(label)
        assert len(unique_labels) <= self.args.num_class, \
            'Label mismatch: found {} unique labels {} but num_class={}'.format(
                len(unique_labels), unique_labels, self.args.num_class)
        assert unique_labels.min() >= 0 and unique_labels.max() < self.args.num_class, \
            'Label values {} out of range [0, {})'.format(unique_labels, self.args.num_class)

        va_val = Averager()
        vf_val = Averager()
        vk_val = Averager()
        tva = []
        tvf = []
        tvk = []
        preds, acts = [], []
        sub_e_scores, sub_r_scores = [], []
        tta = []
        ttf = []
        ttk = []
        kf = KFold(n_splits=fold, shuffle=True)

        for idx_fold, (idx_train, idx_test) in enumerate(kf.split(data)):
            print('Public expert: {}-fold-CV Fold:{}'.format(fold, idx_fold))
            data_train, label_train, data_test, label_test = self.prepare_data(
                idx_train=idx_train, idx_test=idx_test, data=data, label=label)

            if reproduce:
                acc_test, pred, act, test_payload, test_era_summary = test(
                    args=self.args, data=data_test, label=label_test,
                    reproduce=reproduce, subject='public', fold=idx_fold)
                acc_val = 0
                f1_val = 0
                kappa_val = 0
            else:
                print('Public Training:', data_train.size(), label_train.size())
                print('Public Test:', data_test.size(), label_test.size())
                acc_val, f1_val, kappa_val = self.first_stage(
                    data=data_train, label=label_train, subject='public', fold=idx_fold)

                combine_train(args=self.args,
                              data_train=data_train, label_train=label_train,
                              subject='public', fold=idx_fold, target_acc=1)

                acc_test, pred, act, test_payload, test_era_summary = test(
                    args=self.args, data=data_test, label=label_test,
                    reproduce=reproduce, subject='public', fold=idx_fold)

            va_val.add(acc_val)
            vf_val.add(f1_val)
            vk_val.add(kappa_val)
            tva.append(acc_val)
            tvf.append(f1_val)
            tvk.append(kappa_val)
            preds.extend(pred)
            acts.extend(act)
            sub_e_scores.extend(test_payload['E_score'])
            sub_r_scores.extend(test_payload['R_score'])

            acc, f1, _, kappa = get_metrics(y_pred=pred, y_true=act)
            tta.append(acc)
            ttf.append(f1)
            ttk.append(kappa)

            fold_result = 'public fold {}: test accuracy {}, f1: {}, kappa: {}, meanE: {}, meanR: {}'.format(
                idx_fold, acc, f1, kappa, test_era_summary['mean_E'], test_era_summary['mean_R'])
            print(fold_result)
            self.log2txt(fold_result)
            for line in self._format_era_center_lines('public fold {}:'.format(idx_fold), test_era_summary):
                print(line)
                self.log2txt(line)

        final_acc, final_f1, _, final_kappa = get_metrics(y_pred=preds, y_true=acts)
        final_era_summary = summarize_era_distribution(sub_e_scores, sub_r_scores, acts, self.args.num_class)
        tta = np.array(tta)
        ttk = np.array(ttk)
        final_lines = [
            'Public Final: test mean ACC:{} std:{}'.format(np.mean(tta), np.std(tta)),
            'Public Final: test mean F1:{}'.format(np.mean(np.array(ttf))),
            'Public Final: test mean Kappa:{} std:{}'.format(np.mean(ttk), np.std(ttk)),
            'Public Final: aggregated ACC:{}'.format(final_acc),
            'Public Final: aggregated F1:{}'.format(final_f1),
            'Public Final: aggregated Kappa:{}'.format(final_kappa),
            'Public Final: val mean ACC:{} std:{}'.format(np.mean(np.array(tva)), np.std(np.array(tva))),
            'Public Final: val mean F1:{}'.format(np.mean(np.array(tvf))),
            'Public Final: val mean Kappa:{}'.format(np.mean(np.array(tvk))),
            'Public Final: test mean E:{}'.format(final_era_summary['mean_E']),
            'Public Final: test mean R:{}'.format(final_era_summary['mean_R'])
        ]
        final_lines.extend(self._format_era_center_lines('Public Final:', final_era_summary))

        for line in final_lines:
            print(line)
            self.log2txt(line)

        if len(preds) > 0:
            plot_confusion_matrix(args=self.args, y_true=acts, y_pred=preds)

    def first_stage(self, data, label, subject, fold):
        """
        this function achieves n-fold-CV to:
        1. select hyper-parameters on training data
        2. get the model for evaluation on testing data
        param data: (segments, 1, channel, data)
        param label: (segments,)
        param subject: which subject the data belongs to
        param fold: which fold the data belongs to
        return: mean validation accuracy
        """
        # use n-fold-CV to select hyper-parameters on training data
        # save the best performance model and the corresponding acc for the second stage
        # data: trial x 1 x channel x time
        kf = KFold(n_splits=3, shuffle=True)
        va = Averager()
        vf = Averager()
        vk = Averager()
        va_item = []
        maxAcc = 0.0
        for i, (idx_train, idx_val) in enumerate(kf.split(data)):
            print('Inner 3-fold-CV Fold:{}'.format(i))
            data_train, label_train = data[idx_train], label[idx_train]
            data_val, label_val = data[idx_val], label[idx_val]
            acc_val, F1_val, kappa_val = train(args=self.args,
                                    data_train=data_train,
                                    label_train=label_train,
                                    data_val=data_val,
                                    label_val=label_val,
                                    subject=subject,
                                    fold=fold)
            va.add(acc_val)
            vf.add(F1_val)
            vk.add(kappa_val)
            va_item.append(acc_val)
            if acc_val >= maxAcc:
                maxAcc = acc_val
                # choose the model with higher val acc as the model to second stage
                old_name = os.path.join(self.args.save_path, 'candidate.pth')
                new_name = os.path.join(self.args.save_path, 'max-acc.pth')
                if os.path.exists(new_name):
                    os.remove(new_name)
                os.rename(old_name, new_name)
                print('New max ACC model saved, with the val ACC being:{}'.format(acc_val))

        mAcc = va.item()
        mF1 = vf.item()
        mKappa = vk.item()
        return mAcc, mF1, mKappa

    def _flatten_segments(self, data):
        if len(data.shape) > 4:
            return np.concatenate(data, axis=0)
        else:
            return data

    def _flatten_labels(self, label):
        if len(label.shape) > 1:
            return np.concatenate(label, axis=0)
        else:
            return label

    """loso加tsne与混淆矩阵"""
    # def loso_CV(self, subjects, reproduce=False):
    #     """
    #     Leave-One-Subject-Out cross validation
    #     """
    #     tta = []
    #     ttf = []
    #     tva = []
    #     tvf = []
    #     ttk = []
    #     tvk = []
    #
    #     all_tsne_features = []  # 收集所有受试者的特征
    #     all_tsne_labels = []  # 收集所有受试者的标签
    #
    #     all_preds = []
    #     all_labels = []
    #
    #     for test_sub in subjects:
    #         print('LOSO - leaving out subject:', test_sub)
    #         test_data_raw, test_label_raw = self.load_per_subject(test_sub)
    #
    #         train_datas = []
    #         train_labels = []
    #         for sub in subjects:
    #             if sub == test_sub:
    #                 continue
    #             d, l = self.load_per_subject(sub)
    #             train_datas.append(d)
    #             train_labels.append(l)
    #
    #         train_data_flat = np.concatenate([self._flatten_segments(d) for d in train_datas], axis=0)
    #         train_label_flat = np.concatenate([self._flatten_labels(l) for l in train_labels], axis=0)
    #
    #         test_data_flat = self._flatten_segments(test_data_raw)
    #         test_label_flat = self._flatten_labels(test_label_raw)
    #
    #         print('Train shape (segments):', train_data_flat.shape, 'Train labels:', train_label_flat.shape)
    #         print('Test shape (segments):', test_data_flat.shape, 'Test labels:', test_label_flat.shape)
    #
    #         train_data_flat, test_data_flat = self.normalize(train=train_data_flat, test=test_data_flat)
    #
    #         data_train = torch.from_numpy(train_data_flat).float()
    #         label_train = torch.from_numpy(train_label_flat).long()
    #
    #         data_test = torch.from_numpy(test_data_flat).float()
    #         label_test = torch.from_numpy(test_label_flat).long()
    #
    #         va_val = Averager()
    #         vf_val = Averager()
    #         vk_val = Averager()
    #         preds, acts = [], []
    #
    #         if self.args.balance:
    #             # 保持行为一致：得到平衡切分（但不覆盖 data_train）
    #             _ = self.split_balance_class(data=train_data_flat, label=train_label_flat,
    #                                         train_rate=self.args.training_rate, random=True)
    #
    #         if reproduce:
    #             acc_test, pred, act = test(args=self.args, data=data_test, label=label_test,
    #                                        reproduce=reproduce,
    #                                        subject=test_sub, fold=0)
    #             acc_val = 0
    #             f1_val = 0
    #             kappa_val = 0
    #         else:
    #             acc_val, f1_val, kappa_val = self.first_stage_loso(data=train_data_flat, label=train_label_flat,
    #                                                subject=test_sub, fold=0)
    #
    #             combine_train(args=self.args,
    #                           data_train=data_train, label_train=label_train,
    #                           subject=test_sub, fold=0, target_acc=1)
    #
    #             acc_test, pred, act = test(args=self.args, data=data_test, label=label_test,
    #                                        reproduce=reproduce,
    #                                        subject=test_sub, fold=0)
    #
    #         # t-SNE 可视化
    #         if self.args.tsne:
    #             test_loader = get_dataloader(data_test, label_test, self.args.batch_size)
    #             # 加载该折训练好的模型
    #             tsne_model = get_model(self.args)
    #             if CUDA:
    #                 tsne_model = tsne_model.cuda()
    #             if reproduce:
    #                 model_name_reproduce = 'sub' + str(test_sub) + '_loso.pth'
    #                 data_type = 'model_{}_{}'.format(self.args.data_format, self.args.label_type)
    #                 experiment_setting = 'T_{}_pool_{}'.format(self.args.T, self.args.pool)
    #                 load_path = os.path.join(self.args.save_path, experiment_setting, data_type,
    #                                          model_name_reproduce)
    #             else:
    #                 load_path = self.args.load_path_final
    #             tsne_model.load_state_dict(torch.load(load_path))
    #             feat, lbl = extract_features(test_loader, tsne_model)
    #             all_tsne_features.append(feat)
    #             all_tsne_labels.append(lbl)
    #
    #         va_val.add(acc_val)
    #         vf_val.add(f1_val)
    #         vk_val.add(kappa_val)
    #         preds.extend(pred)
    #         acts.extend(act)
    #
    #         all_preds.extend(pred)
    #         all_labels.extend(act)
    #
    #         tva.append(va_val.item())
    #         tvf.append(vf_val.item())
    #         acc, f1, _, kappa = get_metrics(y_pred=preds, y_true=acts)
    #         tta.append(acc)
    #         ttf.append(f1)
    #         ttk.append(kappa)
    #         tvk.append(vk_val.item())
    #
    #         result = 'LOSO sub {}: total test accuracy {}, f1: {}, Kappa{}'.format(test_sub, tta[-1], f1, kappa)
    #         self.log2txt(result)
    #
    #     # final summary across all left-out subjects
    #     tta = np.array(tta)
    #     ttf = np.array(ttf)
    #     tva = np.array(tva)
    #     tvf = np.array(tvf)
    #     ttk = np.array(ttk)
    #     tvk = np.array(tvk)
    #     mACC = np.mean(tta)
    #     mF1 = np.mean(ttf)
    #     mKappa = np.mean(ttk)
    #     std = np.std(tta)
    #     stdF1 = np.std(ttf)
    #     stdKappa = np.std(ttk)
    #     mACC_val = np.mean(tva)
    #     std_val = np.std(tva)
    #     mF1_val = np.mean(tvf)
    #     mKappa_val = np.mean(tvk)
    #
    #     print('Final LOSO: test mean ACC:{} std:{}'.format(mACC, std))
    #     print('Final LOSO: test mean F1:{}'.format(mF1))
    #     print('Final LOSO: test mean Kappa:{} std:{}'.format(mKappa, stdKappa))
    #     print('Final LOSO: val mean ACC:{} std:{}'.format(mACC_val, std_val))
    #     print('Final LOSO: val mean F1:{}'.format(mF1_val))
    #     print('Final LOSO: val mean Kappa:{}'.format(mKappa_val))
    #     results = ('LOSO test mAcc={} std:{} mF1={} std:{} mKappa={} stdKappa={} \n'
    #                'val mAcc={} F1={} mKappa={}').format(mACC, std, mF1, stdF1,mKappa, stdKappa, mACC_val, mF1_val, mKappa_val)
    #     self.log2txt(results)
    #
    #     # 所有受试者特征收集完毕后，统一进行 t-SNE 可视化
    #     if self.args.tsne and len(all_tsne_features) > 0:
    #         tsne_visualize(args=self.args, all_features=all_tsne_features, all_labels=all_tsne_labels)
    #
    #     # 混淆矩阵可视化
    #     if len(all_preds) > 0:
    #         plot_confusion_matrix(args=self.args, y_true=all_labels, y_pred=all_preds)

    """原版loso"""
    def loso_CV(self, subjects, reproduce=False):
        """
        Leave-One-Subject-Out cross validation
        """
        tta = []
        ttf = []
        tva = []
        tvf = []
        ttk = []
        tvk = []

        for test_sub in subjects:
            print('LOSO - leaving out subject:', test_sub)
            test_data_raw, test_label_raw = self.load_per_subject(test_sub)

            train_datas = []
            train_labels = []
            for sub in subjects:
                if sub == test_sub:
                    continue
                d, l = self.load_per_subject(sub)
                train_datas.append(d)
                train_labels.append(l)

            train_data_flat = np.concatenate([self._flatten_segments(d) for d in train_datas], axis=0)
            train_label_flat = np.concatenate([self._flatten_labels(l) for l in train_labels], axis=0)

            test_data_flat = self._flatten_segments(test_data_raw)
            test_label_flat = self._flatten_labels(test_label_raw)

            print('Train shape (segments):', train_data_flat.shape, 'Train labels:', train_label_flat.shape)
            print('Test shape (segments):', test_data_flat.shape, 'Test labels:', test_label_flat.shape)

            train_data_flat, test_data_flat = self.normalize(train=train_data_flat, test=test_data_flat)

            data_train = torch.from_numpy(train_data_flat).float()
            label_train = torch.from_numpy(train_label_flat).long()

            data_test = torch.from_numpy(test_data_flat).float()
            label_test = torch.from_numpy(test_label_flat).long()

            va_val = Averager()
            vf_val = Averager()
            vk_val = Averager()
            preds, acts = [], []

            if self.args.balance:
                # 保持行为一致：得到平衡切分（但不覆盖 data_train）
                _ = self.split_balance_class(data=train_data_flat, label=train_label_flat,
                                            train_rate=self.args.training_rate, random=True)

            if reproduce:
                acc_test, pred, act, test_payload, test_era_summary = test(
                    args=self.args, data=data_test, label=label_test,
                    reproduce=reproduce,
                    subject=test_sub, fold=0)
                acc_val = 0
                f1_val = 0
                kappa_val = 0
            else:
                acc_val, f1_val, kappa_val = self.first_stage_loso(data=train_data_flat, label=train_label_flat,
                                                   subject=test_sub, fold=0)

                combine_train(args=self.args,
                              data_train=data_train, label_train=label_train,
                              subject=test_sub, fold=0, target_acc=1)

                acc_test, pred, act, test_payload, test_era_summary = test(
                    args=self.args, data=data_test, label=label_test,
                    reproduce=reproduce,
                    subject=test_sub, fold=0)

            va_val.add(acc_val)
            vf_val.add(f1_val)
            vk_val.add(kappa_val)
            preds.extend(pred)
            acts.extend(act)

            tva.append(va_val.item())
            tvf.append(vf_val.item())
            acc, f1, _, kappa = get_metrics(y_pred=preds, y_true=acts)
            tta.append(acc)
            ttf.append(f1)
            ttk.append(kappa)
            tvk.append(vk_val.item())

            result = 'LOSO sub {}: total test accuracy {}, f1: {}, Kappa{}'.format(test_sub, tta[-1], f1, kappa)
            sub_era_summary = summarize_era_distribution(
                test_payload['E_score'], test_payload['R_score'], acts, self.args.num_class)
            print('LOSO sub {}: total test accuracy {}, f1: {}, Kappa{}, meanE:{}, meanR:{}'.format(
                test_sub, tta[-1], f1, kappa, sub_era_summary['mean_E'], sub_era_summary['mean_R']))
            print_era_class_centers('LOSO sub {}:'.format(test_sub), sub_era_summary)
            self.log2txt(result)
            self.log2txt('LOSO sub {} ERA centers: {}'.format(test_sub, sub_era_summary['class_centers']))

        # final summary across all left-out subjects
        tta = np.array(tta)
        ttf = np.array(ttf)
        tva = np.array(tva)
        tvf = np.array(tvf)
        ttk = np.array(ttk)
        tvk = np.array(tvk)
        mACC = np.mean(tta)
        mF1 = np.mean(ttf)
        mKappa = np.mean(ttk)
        std = np.std(tta)
        stdF1 = np.std(ttf)
        stdKappa = np.std(ttk)
        mACC_val = np.mean(tva)
        std_val = np.std(tva)
        mF1_val = np.mean(tvf)
        mKappa_val = np.mean(tvk)

        print('Final LOSO: test mean ACC:{} std:{}'.format(mACC, std))
        print('Final LOSO: test mean F1:{}'.format(mF1))
        print('Final LOSO: test mean Kappa:{} std:{}'.format(mKappa, stdKappa))
        print('Final LOSO: val mean ACC:{} std:{}'.format(mACC_val, std_val))
        print('Final LOSO: val mean F1:{}'.format(mF1_val))
        print('Final LOSO: val mean Kappa:{}'.format(mKappa_val))
        all_e_scores = []
        all_r_scores = []
        all_eval_labels = []
        for test_sub in subjects:
            era_summary_path = os.path.join(self.args.save_path, 'era_results', f'sub{test_sub}_fold0_era_outputs.npz')
            if os.path.exists(era_summary_path):
                era_data = np.load(era_summary_path)
                all_e_scores.extend(era_data['E_score'].tolist())
                all_r_scores.extend(era_data['R_score'].tolist())
                all_eval_labels.extend(era_data['label'].tolist())
        final_era_summary = summarize_era_distribution(all_e_scores, all_r_scores, all_eval_labels, self.args.num_class)
        print('Final LOSO: test mean E:{}'.format(final_era_summary['mean_E']))
        print('Final LOSO: test mean R:{}'.format(final_era_summary['mean_R']))
        print_era_class_centers('Final LOSO:', final_era_summary)
        results = ('LOSO test mAcc={} std:{} mF1={} std:{} mKappa={} stdKappa={} \n'
                   'val mAcc={} F1={} mKappa={}\n'
                   'ERA meanE={} meanR={} centers={}').format(
            mACC, std, mF1, stdF1, mKappa, stdKappa,
            mACC_val, mF1_val, mKappa_val,
            final_era_summary['mean_E'], final_era_summary['mean_R'],
            final_era_summary['class_centers'])
        self.log2txt(results)

    def first_stage_loso(self, data, label, subject, fold):
        """
        inner 3-fold CV to select hyper-parameters on training data
        """
        kf = KFold(n_splits=3, shuffle=True)

        va = Averager()
        vf = Averager()
        vk = Averager()

        maxAcc = 0.0

        for i, (idx_train, idx_val) in enumerate(kf.split(data)):

            print('Inner 3-fold-CV Fold:{}'.format(i))

            data_train = data[idx_train]
            label_train = label[idx_train]

            data_val = data[idx_val]
            label_val = label[idx_val]

            # ==========================
            # 新增：保证输入都是 Tensor
            # ==========================
            if isinstance(data_train, np.ndarray):
                data_train = torch.from_numpy(data_train).float()

            if isinstance(label_train, np.ndarray):
                label_train = torch.from_numpy(label_train).long()

            if isinstance(data_val, np.ndarray):
                data_val = torch.from_numpy(data_val).float()

            if isinstance(label_val, np.ndarray):
                label_val = torch.from_numpy(label_val).long()
            # ==========================

            acc_val, F1_val, Kappa_val = train(
                args=self.args,
                data_train=data_train,
                label_train=label_train,
                data_val=data_val,
                label_val=label_val,
                subject=subject,
                fold=fold
            )

            va.add(acc_val)
            vf.add(F1_val)
            vk.add(Kappa_val)

            if acc_val >= maxAcc:

                maxAcc = acc_val

                old_name = os.path.join(self.args.save_path, 'candidate.pth')
                new_name = os.path.join(self.args.save_path, 'max-acc.pth')

                if os.path.exists(new_name):
                    os.remove(new_name)

                os.rename(old_name, new_name)

                print('New max ACC model saved, with the val ACC being:{}'.format(acc_val))

        return va.item(), vf.item(), vk.item()

    def log2txt(self, content):
        """
        This function log the content to results.txt
        param content: string, the content to log.
        """
        file = open(self.text_file, 'a')
        file.write(str(content) + '\n')
        file.close()
