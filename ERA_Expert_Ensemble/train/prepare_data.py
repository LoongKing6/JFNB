import pickle as cPickle

import mne
import pyedflib
from scipy import signal
from sklearn.decomposition import FastICA, PCA
from train.train_model import *
from scipy.signal import resample
import scipy.signal
from torch.utils.data import Dataset
import os
import json
import csv
import glob
import numpy as np
import h5py
import concurrent.futures

from config.config import *
_, os.environ['CUDA_VISIBLE_DEVICES'] = set_config([])
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

class eegDataset(Dataset):
    # x_tensor: (sample, channel, datapoint(feature)) type = torch.tensor
    # y_tensor: (sample,) type = torch.tensor

    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

        assert self.x.size(0) == self.y.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


class PrepareData:
    def __init__(self, args):
        # init all the parameters here
        # arg contains parameter settings
        self.args = args
        self.data = None
        self.label = None
        self.model = None
        self.data_path = args.data_path
        self.label_type = args.label_type
        self.dataset = args.dataset
        self.save_metadata = {}
        self.era_target_original_order = ['Cz', 'Fpz', 'Fz', 'FCz', 'Pz', 'Oz', 'T3', 'T4',
                                          'C3', 'C4', 'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8',
                                          'FC3', 'FC4', 'FT7', 'FT8', 'O1', 'O2', 'T5', 'T6',
                                          'P3', 'P4', 'CP3', 'CP4', 'TP7', 'TP8', 'A1', 'CPz']
        self.external_common_32_order = ['Fp1', 'Fp2', 'AF3', 'AF4', 'Fz', 'F3', 'F4', 'F7', 'F8',
                                         'FC1', 'FC2', 'FC5', 'FC6', 'Cz', 'C3', 'C4', 'T7', 'T8',
                                         'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'P7', 'P8',
                                         'PO3', 'PO4', 'Oz', 'O1', 'O2']
        # This mapping is only used for external ERA transfer inference. It keeps
        # the 32 original channels one-to-one without channel synthesis.
        self.external32_to_era32 = {
            'Fp1': 'Fp1',
            'Fp2': 'Fp2',
            'AF3': 'Fpz',
            'AF4': 'A1',
            'Fz': 'Fz',
            'F3': 'F3',
            'F4': 'F4',
            'F7': 'F7',
            'F8': 'F8',
            'FC1': 'FCz',
            'FC2': 'CPz',
            'FC5': 'FC3',
            'FC6': 'FC4',
            'Cz': 'Cz',
            'C3': 'C3',
            'C4': 'C4',
            'T7': 'T3',
            'T8': 'T4',
            'CP1': 'CP3',
            'CP2': 'CP4',
            'CP5': 'FT7',
            'CP6': 'FT8',
            'Pz': 'Pz',
            'P3': 'P3',
            'P4': 'P4',
            'P7': 'T5',
            'P8': 'T6',
            'PO3': 'TP7',
            'PO4': 'TP8',
            'Oz': 'Oz',
            'O1': 'O1',
            'O2': 'O2'
        }
        if self.dataset == 'MEEG':
            self.original_order = ['Fp1', 'Fp2', 'AF3', 'AF4', 'Fz', 'F3', 'F4', 'F7', 'F8',
                                   'FC1', 'FC2', 'FC5', 'FC6', 'Cz', 'C3', 'C4', 'T7', 'T8',
                                   'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'P7', 'P8',
                                   'PO3', 'PO4', 'Oz', 'O1', 'O2']
            self.args.sampling_rate = 1000
            self.args.target_rate = 500
            self.args.channels = 32
            self.args.segment = 4
            self.args.input_shape = (1, self.args.channels, int(self.args.segment * self.args.target_rate))
        elif self.dataset == 'DEAP':
            self.original_order = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
                                   'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8',
                                   'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
            self.args.sampling_rate = 128
            self.args.target_rate = 500
            self.args.channels = 32
            self.args.segment = 4
            self.args.input_shape = (1, self.args.channels, int(self.args.segment * self.args.target_rate))
        elif self.dataset == 'EMO':
            self.original_order = ['Cz', 'Fpz', 'Fz', 'FCz', 'Pz', 'Oz', 'T3', 'T4',
                                   'C3', 'C4', 'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8',
                                   'FC3', 'FC4', 'FT7', 'FT8', 'O1', 'O2', 'T5', 'T6',
                                   'P3', 'P4', 'CP3', 'CP4', 'TP7', 'TP8', 'A1', 'CPz']
            self.args.sampling_rate = 2000
            self.args.target_rate = self.args.target_rate
            self.args.channels = 32
            self.args.input_shape = (1, self.args.channels, int(self.args.segment * self.args.target_rate))
        elif self.dataset == 'EEGMAT':
            self.original_order = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4',
                                   'C3', 'C4', 'T5', 'T6', 'P3', 'P4', 'O1', 'O2',
                                   'Fz', 'Cz', 'Pz', 'A2-A1']
            self.args.sampling_rate = 500
            self.args.target_rate = 500
            self.args.channels = 32
            self.args.segment = 4
            self.args.input_shape = (1, self.args.channels, int(self.args.segment * self.args.target_rate))
        elif self.dataset == 'ISRUC':
            # ISRUC EEG channels - supports three naming conventions
            # Format 1: A1/A2 reference (early subjects) - e.g., F3-A2, C3-A2
            # Format 2: M1/M2 reference (later subjects) - e.g., F3-M2, C3-M2
            # Format 3: Monopolar (some subjects) - e.g., F3, C3, O1 (reference channels A1/A2 separate)
            self.original_order_a1a2 = ['F3-A2', 'C3-A2', 'O1-A2', 'F4-A1', 'C4-A1', 'O2-A1']
            self.original_order_m1m2 = ['F3-M2', 'C3-M2', 'O1-M2', 'F4-M1', 'C4-M1', 'O2-M1']
            self.original_order_mono = ['F3', 'C3', 'O1', 'F4', 'C4', 'O2']
            self.original_order = self.original_order_a1a2  # Default
            self.args.sampling_rate = 200
            self.args.target_rate = 100
            self.args.channels = 6
            self.args.segment = 30
        else:
            self.original_order = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3',
                                   'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8',
                                   'CP6',
                                   'CP2', 'P4', 'P8', 'PO4', 'O2']
        self.graph_fro_DEAP = [['Fp1', 'AF3'], ['Fp2', 'AF4'], ['F3', 'F7'], ['F4', 'F8'],
                               ['Fz'],
                               ['FC5', 'FC1'], ['FC6', 'FC2'], ['C3', 'Cz', 'C4'], ['CP5', 'CP1', 'CP2', 'CP6'],
                               ['P7', 'P3', 'Pz', 'P4', 'P8'], ['PO3', 'PO4'], ['O1', 'Oz', 'O2'],
                               ['T7'], ['T8']]
        self.graph_gen_DEAP = [['Fp1', 'Fp2'], ['AF3', 'AF4'], ['F3', 'F7', 'Fz', 'F4', 'F8'],
                               ['FC5', 'FC1', 'FC6', 'FC2'], ['C3', 'Cz', 'C4'], ['CP5', 'CP1', 'CP2', 'CP6'],
                               ['P7', 'P3', 'Pz', 'P4', 'P8'], ['PO3', 'PO4'], ['O1', 'Oz', 'O2'],
                               ['T7'], ['T8']]
        self.graph_hem_DEAP = [['Fp1', 'AF3'], ['Fp2', 'AF4'], ['F3', 'F7'], ['F4', 'F8'],
                               ['Fz', 'Cz', 'Pz', 'Oz'],
                               ['FC5', 'FC1'], ['FC6', 'FC2'], ['C3'], ['C4'], ['CP5', 'CP1'], ['CP2', 'CP6'],
                               ['P7', 'P3'], ['P4', 'P8'], ['PO3', 'O1'], ['PO4', 'O2'], ['T7'], ['T8']]

        self.graph_fro_EMO = [['Fp1'], ['Fp2'], ['F3', 'F7'], ['F4', 'F8'],
                              ['Fpz', 'Fz', 'FCz'],
                              ['FT7', 'FC3'], ['FT8', 'FC4'], ['C3', 'Cz', 'C4'], ['TP7', 'CP3', 'CPz', 'CP4', 'TP8'],
                              ['T5', 'P3', 'Pz', 'P4', 'T6'], ['O1', 'Oz', 'O2'],
                              ['T3'], ['T4'], ['A1']]

        self.graph_gen_EMO = [['Fp1', 'Fp2', 'Fpz'], ['F3', 'F7', 'Fz', 'F4', 'F8'],
                              ['FT7', 'FC3', 'FCz', 'FT8', 'FC4'], ['C3', 'Cz', 'C4'], ['TP7', 'CP3', 'CPz', 'CP4', 'TP8'],
                              ['T5', 'P3', 'Pz', 'P4', 'T6'], ['O1', 'Oz', 'O2'],
                              ['T3'], ['T4'], ['A1']]

        # self.graph_hem_EMO = [['Fp1'], ['Fp2'], ['F3', 'F7'], ['F4', 'F8'],
        #                       ['Fpz', 'Fz', 'FCz', 'Cz', 'Pz', 'Oz'],
        #                       ['FT7', 'FC3'], ['FT8', 'FC4'], ['C3'], ['C4'], ['TP7', 'CP3'], ['CP4', 'TP8'],
        #                       ['T5', 'P3'], ['P4', 'T6'], ['O1'], ['O2'], ['T3'], ['T4'], ['A1']]
        self.graph_hem_EMO = [['Fp1'], ['F3', 'F7'], ['FT7', 'FC3'], ['C3', 'T3'], ['TP7', 'CP3'], ['T5', 'P3'], ['O1'],
                              ['Fpz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'Oz'], ['A1'],
                              ['Fp2'], ['F4', 'F8'], ['FT8', 'FC4'], ['C4', 'T4'], ['CP4', 'TP8'], ['P4', 'T6'], ['O2'],
                              ]

        self.graph_10_EMO = [
            ['Fp1', 'Fp2', 'Fpz'],  # 前额叶顶部
            ['F7', 'F3', 'Fz', 'F4', 'F8'],  # 额叶（前带情绪网络）
            ['FT7', 'FC3', 'FC4', 'FT8'],  # 额-中央前区
            ['FCz', 'C3', 'Cz', 'C4'],  # 中央中线区（认知/情绪整合）
            ['T3', 'CP3', 'P3', 'TP7'],  # 左颞顶（语义/记忆）
            ['T4', 'CP4', 'P4', 'TP8'],  # 右颞顶（情感语境）
            ['Pz'],  # 后顶中线
            ['O1', 'Oz', 'O2'],  # 枕叶
            ['T5', 'T6'],  # 侧枕（外围）
            ['A1', 'A2']  # 耳
        ]

        self.graph_EMO = [[ch] for ch in self.original_order]

        self.graph_EEGMAT_fro = [['Fp1', 'Fp2'], ['F3', 'F7', 'Fz', 'F4', 'F8'],
                                 ['C3', 'Cz', 'C4', 'T3', 'T4'],
                                 ['P3', 'Pz', 'P4', 'T5', 'T6'],
                                 ['O1', 'O2'], ['A2-A1']]
        self.graph_EEGMAT_gen = [['Fp1', 'Fp2', 'Fz'], ['F3', 'F7', 'F4', 'F8'],
                                 ['C3', 'Cz', 'C4'], ['T3', 'T4', 'T5', 'T6'],
                                 ['P3', 'Pz', 'P4'], ['O1', 'O2', 'A2-A1']]
        self.graph_EEGMAT_hem = [['Fp1', 'F3', 'F7', 'C3', 'T3', 'T5', 'P3', 'O1'],
                                 ['Fp2', 'F4', 'F8', 'C4', 'T4', 'T6', 'P4', 'O2'],
                                 ['Fz', 'Cz', 'Pz', 'A2-A1']]
        self.graph_EEGMAT = [[ch] for ch in self.original_order]

        # ISRUC graph structures
        self.graph_ISRUC_fro = [['F3-A2'], ['C3-A2'], ['O1-A2'],
                                ['F4-A1'], ['C4-A1'], ['O2-A1']]
        self.graph_ISRUC_gen = [['F3-A2', 'F4-A1'],
                                ['C3-A2', 'C4-A1'],
                                ['O1-A2', 'O2-A1']]
        self.graph_ISRUC_hem = [['F3-A2', 'C3-A2', 'O1-A2'],
                                ['F4-A1', 'C4-A1', 'O2-A1']]
        self.graph_ISRUC = [[ch] for ch in self.original_order]

        self.graph_type = args.graph_type

    def is_external_era_dataset(self):
        return self.label_type == 'Axis' and self.dataset in ['DEAP', 'MEEG', 'EEGMAT']

    def run(self, subject_list, split, expand):
        """
        Parameters
        ----------
        subject_list: the subjects need to be processed
        split: (bool) whether to split one trial's data into shorter segment
        expand: (bool) whether to add an empty dimension for CNN

        Returns
        -------
        The processed data will be saved './data_<data_format>_<dataset>_<label_type>/sub0.hdf'
        """
        for sub in subject_list:
            self.save_metadata = {'source_dataset': self.dataset}
            result = self.load_data_per_subject(sub)
            if len(result) == 3:
                data_, label_, resting_data_ = result
            else:
                data_, label_ = result
                resting_data_ = None
            # select label type here
            label_ = self.label_selection(label_)
            self.save_metadata['era_label'] = np.asarray(label_, dtype=np.int64)

            data_, label_ = self.preprocess_data(data=data_, label=label_, split=split, expand=expand,
                                                 resting_data=resting_data_)

            if self.dataset == 'EEGMAT' and data_.ndim == 5 and data_.shape[0] == 1:
                data_ = data_.squeeze(0)
                label_ = label_.squeeze(0)

            print('Data and label prepared!')
            print('sample_' + str(sub + 1) + '.dat')
            print('data:' + str(data_.shape) + ' label:' + str(label_.shape))
            print('----------------------')
            self.save(data_, label_, sub)

        self.args.sampling_rate = self.args.target_rate

    def load_saved_subject_hdf(self, sub, dataset_name=None):
        save_path = os.getcwd()
        dataset_name = dataset_name or self.args.dataset
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, dataset_name, self.args.label_type)
        sub_code = 'sub' + str(sub) + '.hdf'
        path = os.path.join(save_path, data_type, sub_code)
        if not os.path.exists(path):
            raise FileNotFoundError('Subject HDF not found: {}'.format(path))
        with h5py.File(path, 'r') as dataset:
            data = np.array(dataset['data'])
            label = np.array(dataset['label'])
        return data, label

    def build_public_dataset(self, subject_list):
        """
        Build a public/common expert dataset by sampling a few representative
        samples from every subject HDF without changing the original subject-specific files.
        """
        if self.dataset != 'EMO':
            raise ValueError('Public expert is only supported for RESB/EMO dataset.')

        fused_data = []
        fused_label = []
        fused_subject = []
        rng = np.random.default_rng(self.args.random_seed)

        for round_idx in range(self.args.data_fuse):
            for sub in subject_list:
                data, label = self.load_saved_subject_hdf(sub, dataset_name='EMO')
                if data.ndim > 4:
                    data = np.concatenate(data, axis=0)
                    label = np.concatenate(label, axis=0)

                for cls in range(self.args.num_class):
                    cls_index = np.where(label == cls)[0]
                    if len(cls_index) == 0:
                        raise ValueError('Subject {} has no samples for class {}.'.format(sub, cls))
                    chosen_idx = int(rng.choice(cls_index))
                    fused_data.append(data[chosen_idx])
                    fused_label.append(label[chosen_idx])
                    fused_subject.append(sub)

        fused_data = np.asarray(fused_data, dtype=np.float32)
        fused_label = np.asarray(fused_label, dtype=np.int64)
        fused_subject = np.asarray(fused_subject, dtype=np.int64)

        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        save_dir = os.path.join(save_path, data_type)
        ensure_path(save_dir)
        public_path = os.path.join(save_dir, 'public_RESB.hdf')

        with h5py.File(public_path, 'w') as dataset:
            dataset['data'] = fused_data
            dataset['label'] = fused_label
            dataset['subject'] = fused_subject
            dataset.attrs['source_dataset'] = 'EMO'
            dataset.attrs['data_fuse'] = self.args.data_fuse
            dataset.attrs['num_subjects'] = len(subject_list)

        print('Public dataset built: data:{} label:{} path:{}'.format(
            fused_data.shape, fused_label.shape, public_path))
        return public_path

    def build_public_preprocessed_data(self, subject_list, split=True, expand=True):
        """
        The subject HDF files are already preprocessed and split. Therefore the
        public dataset only needs sample fusion and direct saving.
        """
        public_hdf_path = self.build_public_dataset(subject_list)
        with h5py.File(public_hdf_path, 'r') as dataset:
            public_data = np.array(dataset['data'])
            public_label = np.array(dataset['label'])
        print('Public processed dataset ready: data:{} label:{} path:{}'.format(
            public_data.shape, public_label.shape, public_hdf_path))
        return public_hdf_path, public_data, public_label

    def load_data_per_subject(self, sub):
        """
        This function loads the target subject's original file
        Parameters
        ----------
        sub: which subject to load

        Returns
        -------
        data: (40, 32, 7680) label: (40, 4)
        """
        sub += 1
        if self.dataset == 'MEEG':
            sub_code = str('sample_' + str(sub) + '.dat')
        elif self.dataset == 'DEAP':
            if sub < 10:
                sub_code = str('s0' + str(sub) + '.dat')
            else:
                sub_code = str('s' + str(sub) + '.dat')
        elif self.dataset == 'EMO':
            return self.load_bids_emo_subject(sub)
        elif self.dataset == 'EEGMAT':
            return self.load_eegmat_subject(sub)
        elif self.dataset == 'ISRUC':
            return self.load_isruc_subject(sub)

        subject_path = os.path.join(self.data_path, sub_code)
        subject = cPickle.load(open(subject_path, 'rb'), encoding='latin1')
        label = subject['labels']
        data = subject['data']
        self.save_metadata['original_label_raw'] = np.asarray(label)
        if self.is_external_era_dataset():
            data = self.align_external_channels_to_era_layout(data=data, channel_names=self.original_order)
        else:
            # reorder the EEG channel to build the local-global graphs
            data = self.reorder_channel(data=data, graph=self.graph_type)
        print('data:' + str(data.shape) + ' label:' + str(label.shape))
        return data, label

    def load_bids_emo_subject(self, sub):
        """
        Load one EMO subject from a BIDS BrainVision directory.

        Returns
        -------
        data: (trial, channel, target_samples)
        label: (trial,)
        """
        subject_dir = os.path.join(self.data_path, 'sub-{:02d}'.format(sub), 'eeg')
        vhdr_files = sorted(glob.glob(os.path.join(subject_dir, '*_eeg.vhdr')))
        if not vhdr_files:
            raise FileNotFoundError('No BrainVision header found under {}'.format(subject_dir))

        vhdr_path = vhdr_files[0]
        base_name = os.path.basename(vhdr_path).replace('_eeg.vhdr', '')
        eeg_json_path = os.path.join(subject_dir, base_name + '_eeg.json')
        events_path = os.path.join(subject_dir, base_name + '_events.tsv')
        channels_path = os.path.join(subject_dir, base_name + '_channels.tsv')

        eeg_file, num_channels, sampling_rate, binary_format = self.read_brainvision_header(vhdr_path)
        if os.path.exists(eeg_json_path):
            with open(eeg_json_path, 'r', encoding='utf-8') as f:
                sampling_rate = int(float(json.load(f)['SamplingFrequency']))

        if sampling_rate != 2000:
            raise ValueError('Expected EMO sampling rate 2000 Hz, got {} Hz in {}'.format(sampling_rate, vhdr_path))
        if binary_format != 'IEEE_FLOAT_32':
            raise ValueError('Unsupported BrainVision BinaryFormat {} in {}'.format(binary_format, vhdr_path))

        channel_names = self.read_bids_channels(channels_path)
        if len(channel_names) != num_channels:
            raise ValueError(
                'Channel count mismatch: header={}, channels.tsv={}'.format(num_channels, len(channel_names)))

        events = self.read_bids_events(events_path)
        eeg_path = os.path.join(subject_dir, eeg_file)
        raw = np.memmap(eeg_path, dtype='<f4', mode='r').reshape(-1, num_channels)

        # pre_seconds = 2
        # post_seconds = 2

        original_samples = int((self.args.pre_seconds + self.args.post_seconds) * sampling_rate)
        target_samples = int((self.args.pre_seconds + self.args.post_seconds) * self.args.target_rate)

        resting_epochs = []
        data = []
        label = []
        for event in events:
            trial_type = event['trial_type'].strip()

            # 收集静息态数据
            if trial_type in ('eye_closed', 'eyes_closed'):
                rest_start = event['sample']
                rest_stop = min(event['sample'] + int(120 * sampling_rate), raw.shape[0])
                if rest_start < 0 or rest_stop > raw.shape[0]:
                    continue
                rest_epoch = np.asarray(raw[rest_start:rest_stop, :], dtype=np.float32).T
                resting_epochs.append(rest_epoch)
                continue

            emo_label = self.map_emo_label(trial_type)
            if emo_label is None:
                continue

            start = event['sample'] - int(self.args.pre_seconds * sampling_rate)
            stop = event['sample'] + int(self.args.post_seconds * sampling_rate)

            if start < 0 or stop > raw.shape[0]:
                continue

            epoch = np.asarray(raw[start:stop, :], dtype=np.float32).T
            epoch = signal.resample_poly(epoch, self.args.target_rate, sampling_rate, axis=-1)
            epoch = epoch[:, :target_samples]
            if epoch.shape[-1] < target_samples:
                epoch = np.pad(epoch, ((0, 0), (0, target_samples - epoch.shape[-1])), mode='constant')

            data.append(epoch.astype(np.float32, copy=False))
            label.append(emo_label)

        if not data:
            raise ValueError('No complete {}s EMO trials found in {}'.format(self.args.trial_duration, events_path))

        data = np.stack(data, axis=0)
        label = np.asarray(label, dtype=np.int64)

        # 处理静息态数据：降采样并拼接
        resting_data = None
        resting_segments = None
        if resting_epochs:
            resting_resampled = []
            resting_segment_batches = []
            for rest_epoch in resting_epochs:
                rest_ds = signal.resample_poly(rest_epoch, self.args.target_rate, sampling_rate, axis=-1)
                resting_resampled.append(rest_ds)
                rest_segments = self.segment_resting_state(rest_ds, target_samples)
                if rest_segments.size > 0:
                    resting_segment_batches.append(rest_segments)
            resting_data = np.concatenate(resting_resampled, axis=-1).astype(np.float32)
            if resting_segment_batches:
                resting_segments = np.concatenate(resting_segment_batches, axis=0).astype(np.float32)

        data = self.reorder_channel(data=data, graph=self.graph_type, channel_names=channel_names)
        if resting_data is not None:
            resting_data = self.reorder_channel(data=resting_data[np.newaxis, :, :], graph=self.graph_type,
                                                channel_names=channel_names)[0]
        if resting_segments is not None:
            resting_segments = self.reorder_channel(
                data=resting_segments, graph=self.graph_type, channel_names=channel_names)
            resting_labels = np.full((resting_segments.shape[0],), 4, dtype=np.int64)
            data = np.concatenate([data, resting_segments], axis=0)
            label = np.concatenate([label, resting_labels], axis=0)

        self.args.sampling_rate = self.args.target_rate
        print('data:' + str(data.shape) + ' label:' + str(label.shape))
        if resting_data is not None:
            print('resting data:' + str(resting_data.shape))
        return data, label, resting_data

    def map_emo_label(self, trial_type):
        """
        Map EMO BIDS trial_type to binary labels for the selected label_type.

        V: positive(1)=GZQ/GZR, negative(0)=GFQ/GFR
        A: strong(1)=GZQ/GFQ, weak(0)=GZR/GFR
        D: rational(1)=LXQ/LXR, emotional(0)=GZQ/GZR/GFQ/GFR
        L: strong rational(1)=LXQ, weak rational(0)=LXR
        """
        trial_type = trial_type.strip()
        mapping = {
            'A': {'GZQ': 1, 'GFQ': 1, 'GZR': 0, 'GFR': 0},
            'V': {'GZQ': 1, 'GZR': 1, 'GFQ': 0, 'GFR': 0},
            'D': {'LXQ': 1, 'LXR': 1, 'GZQ': 0, 'GZR': 0, 'GFQ': 0, 'GFR': 0},
            'L': {'LXQ': 1, 'LXR': 0},
            'S': {'GZQ': 0, 'GZR': 1, 'GFQ': 2, 'GFR': 3, 'LXQ': 4, 'LXR': 5},
            'haha': {'LXR': 1, 'eye_closed': 0, 'eyes_closed': 0},
            'Axis': {
                'GZR': 0,
                'GFR': 0,
                'GZQ': 1,
                'GFQ': 1,
                'LXQ': 2,
                'LXR': 3,
                'eye_closed': 4,
                'eyes_closed': 4
            },
        }
        return mapping[self.label_type].get(trial_type)

    def segment_resting_state(self, resting_epoch, target_samples):
        """
        Slice a continuous resting-state recording into fixed-length samples so
        the fifth class is present in the supervised training set.
        """
        if resting_epoch.shape[-1] < target_samples:
            padded = np.pad(
                resting_epoch,
                ((0, 0), (0, target_samples - resting_epoch.shape[-1])),
                mode='constant'
            )
            return padded[np.newaxis, :, :].astype(np.float32)

        segments = []
        for start in range(0, resting_epoch.shape[-1] - target_samples + 1, target_samples):
            segment = resting_epoch[:, start:start + target_samples]
            segments.append(segment.astype(np.float32, copy=False))

        if not segments:
            return np.empty((0, resting_epoch.shape[0], target_samples), dtype=np.float32)
        return np.stack(segments, axis=0)

    def read_brainvision_header(self, vhdr_path):
        eeg_file = None
        num_channels = None
        sampling_interval = None
        binary_format = None

        with open(vhdr_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(';') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                if key == 'DataFile':
                    eeg_file = value
                elif key == 'NumberOfChannels':
                    num_channels = int(value)
                elif key == 'SamplingInterval':
                    sampling_interval = float(value)
                elif key == 'BinaryFormat':
                    binary_format = value

        if eeg_file is None or num_channels is None or sampling_interval is None:
            raise ValueError('Incomplete BrainVision header: {}'.format(vhdr_path))

        sampling_rate = int(round(1000000.0 / sampling_interval))
        return eeg_file, num_channels, sampling_rate, binary_format

    def read_bids_channels(self, channels_path):
        with open(channels_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            return [row['name'] for row in reader]

    def read_bids_events(self, events_path):
        events = []
        with open(events_path, 'r', encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                if row.get('sample') in [None, '', 'n/a'] or row.get('value') in [None, '', 'n/a']:
                    continue
                events.append({
                    'sample': int(float(row['sample'])),
                    'value': int(float(row['value'])),
                    'trial_type': row.get('trial_type', '')
                })
        return events

    """滑动窗口load_eegmat_subject"""



    def load_eegmat_subject(self, sub):
        """
        Load one EEGMAT subject from EDF files.

        Sliding-window version:
        - reads the continuous task recording (_2.edf)
        - resamples to target_rate first
        - cuts the recording into many overlapped windows
        - expands 20 channels to 32 channels
        - maps external_common_32_order -> era_target_original_order

        Returns
        -------
        data: (num_windows, 32, window_samples)
        label: (num_windows,)
        """

        EEGMAT_ORIGINAL_ORDER = [
            "Fp1", "Fp2", "F3", "F4", "F7", "F8", "T3", "T4",
            "C3", "C4", "T5", "T6", "P3", "P4", "O1", "O2",
            "Fz", "Cz", "Pz", "A2-A1"
        ]

        sub_code = 'Subject{:02d}'.format(sub - 1)
        edf_path = os.path.join(self.data_path, sub_code + '_2.edf')
        if not os.path.exists(edf_path):
            raise FileNotFoundError('EDF file not found: {}'.format(edf_path))

        reader = pyedflib.EdfReader(edf_path)
        try:
            channel_labels = reader.getSignalLabels()
            sampling_rate = int(reader.getSampleFrequencies()[0])

            eeg_channels = []
            eeg_indices = []
            for i, label in enumerate(channel_labels):
                clean_label = label.replace('EEG ', '').replace('ECG ', '').strip()
                if clean_label in EEGMAT_ORIGINAL_ORDER and clean_label != 'ECG':
                    eeg_channels.append(clean_label)
                    eeg_indices.append(i)

            expected_src_channels = len(EEGMAT_ORIGINAL_ORDER)
            if len(eeg_channels) != expected_src_channels:
                missing = [ch for ch in EEGMAT_ORIGINAL_ORDER if ch not in eeg_channels]
                raise ValueError(
                    'Expected {} source EEGMAT channels, found {}. Missing: {}'.format(
                        expected_src_channels, len(eeg_channels), missing
                    )
                )

            all_data = []
            for idx in eeg_indices:
                sig = reader.readSignal(idx)
                all_data.append(sig)

            data = np.stack(all_data, axis=0).astype(np.float32)  # (20, samples)
        finally:
            reader.close()

        csv_path = os.path.join(self.data_path, 'subject-info.csv')
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader_csv = csv.DictReader(f)
            for row in reader_csv:
                if row['Subject'] == sub_code:
                    label_value = int(float(row['Count quality']))
                    break
            else:
                raise ValueError('Subject {} not found in subject-info.csv'.format(sub_code))

        # 先统一重采样到 target_rate，再做滑动窗口
        if sampling_rate != self.args.target_rate:
            data = signal.resample_poly(
                data, self.args.target_rate, sampling_rate, axis=-1
            ).astype(np.float32)
            sampling_rate = self.args.target_rate

        # 窗口长度和步长（单位：sample）
        window_samples = int(round(self.args.segment * sampling_rate))
        step_samples = int(round(window_samples * (1 - self.args.overlap)))

        if window_samples <= 0:
            raise ValueError('Invalid segment length: {}'.format(self.args.segment))
        if step_samples <= 0:
            raise ValueError(
                'Invalid overlap: {}. It makes step_samples <= 0.'.format(self.args.overlap)
            )

        n_samples = data.shape[-1]
        if n_samples < window_samples:
            pad_len = window_samples - n_samples
            data = np.pad(data, ((0, 0), (0, pad_len)), mode='constant')
            n_samples = data.shape[-1]

        windows = []
        for start in range(0, n_samples - window_samples + 1, step_samples):
            window = data[:, start:start + window_samples]
            windows.append(window.astype(np.float32, copy=False))

        if not windows:
            windows.append(data[:, :window_samples].astype(np.float32, copy=False))

        data = np.stack(windows, axis=0)  # (num_windows, 20, window_samples)
        label = np.full((data.shape[0],), label_value, dtype=np.int64)

        # 20 -> 32 通道扩展
        expanded_windows = []
        for window in data:
            expanded = self.graph_interpolation_expand(window, sampling_rate)
            expanded_windows.append(expanded)
        data = np.stack(expanded_windows, axis=0).astype(np.float32)  # (num_windows, 32, window_samples)

        # external_common_32_order -> era_target_original_order
        data = self.align_external_channels_to_era_layout(
            data=data,
            channel_names=self.external_common_32_order,
            already_era_aligned=False
        )

        # 保证后续模型输入仍然是 32 通道
        self.args.channels = 32
        self.args.input_shape = (1, 32, int(self.args.segment * self.args.target_rate))
        self.save_metadata['original_label_raw'] = np.asarray(label)

        self.args.sampling_rate = int(sampling_rate)
        print('data:' + str(data.shape) + ' label:' + str(label.shape))
        return data, label

    def load_isruc_subject(self, sub):
        """
        Load ISRUC-Sleep dataset from EDF files.
        Each 30-second epoch is annotated with sleep stage (0=W, 1=N1, 2=N2, 3=N3, 5=REM).

        Returns
        -------
        data: (epoch, channel, 30s_samples)
        label: (epoch,) - sleep stage labels
        """
        # ISRUC directory structure: example/ISRUC/1/, example/ISRUC/2/
        subject_dir = str(sub)
        edf_path = os.path.join(self.data_path, subject_dir, subject_dir + '.edf')

        if not os.path.exists(edf_path):
            raise FileNotFoundError('EDF file not found: {}'.format(edf_path))

        # Read EDF file
        reader = pyedflib.EdfReader(edf_path)
        n_channels = reader.signals_in_file
        channel_labels = reader.getSignalLabels()
        sampling_rate = int(reader.getSampleFrequencies()[0])

        # Select EEG channels - support three naming conventions
        # Format 1: A1/A2 reference (e.g., F3-A2)
        # Format 2: M1/M2 reference (e.g., F3-M2)
        # Format 3: Monopolar (e.g., F3, C3, O1)
        eeg_channels = []
        eeg_indices = []
        channel_type = None  # Will detect which naming convention is used

        for i, label in enumerate(channel_labels):
            # Clean label format (may have spaces)
            clean_label = label.strip().replace(' ', '')

            # Try matching with A1/A2 naming (early subjects)
            if clean_label in self.original_order_a1a2:
                if channel_type is None:
                    channel_type = 'A1A2'
                eeg_channels.append(clean_label)
                eeg_indices.append(i)
            # Try matching with M1/M2 naming (later subjects)
            elif clean_label in self.original_order_m1m2:
                if channel_type is None:
                    channel_type = 'M1M2'
                eeg_channels.append(clean_label)
                eeg_indices.append(i)
            # Try matching with monopolar naming (some subjects)
            elif clean_label in self.original_order_mono:
                if channel_type is None:
                    channel_type = 'MONOPOLAR'
                eeg_channels.append(clean_label)
                eeg_indices.append(i)
        # Set original_order based on detected channel type
        if channel_type == 'M1M2':
            self.original_order = self.original_order_m1m2
            print(f"ISRUC subject {sub}: Using M1/M2 reference naming")
        elif channel_type == 'A1A2':
            self.original_order = self.original_order_a1a2
            print(f"ISRUC subject {sub}: Using A1/A2 reference naming")
        elif channel_type == 'MONOPOLAR':
            self.original_order = self.original_order_mono
            print(f"ISRUC subject {sub}: Using monopolar channel naming")
        else:
            print(f"Warning: Could not detect channel naming convention for subject {sub}")
            print(f"  Available channels: {channel_labels[:10]}")

        # Update graph structures to use actual channel names
        if channel_type == 'M1M2':
            self.graph_ISRUC_fro = [['F3-M2'], ['C3-M2'], ['O1-M2'],
                                    ['F4-M1'], ['C4-M1'], ['O2-M1']]
            self.graph_ISRUC_gen = [['F3-M2', 'F4-M1'],
                                    ['C3-M2', 'C4-M1'],
                                    ['O1-M2', 'O2-M1']]
            self.graph_ISRUC_hem = [['F3-M2', 'C3-M2', 'O1-M2'],
                                    ['F4-M1', 'C4-M1', 'O2-M1']]
        elif channel_type == 'MONOPOLAR':
            self.graph_ISRUC_fro = [['F3'], ['C3'], ['O1'],
                                    ['F4'], ['C4'], ['O2']]
            self.graph_ISRUC_gen = [['F3', 'F4'],
                                    ['C3', 'C4'],
                                    ['O1', 'O2']]
            self.graph_ISRUC_hem = [['F3', 'C3', 'O1'],
                                    ['F4', 'C4', 'O2']]

        # If we don't have all expected channels, adjust dynamically
        if len(eeg_channels) < self.args.channels:
            print(f"Warning: Expected {self.args.channels} EEG channels, found {len(eeg_channels)}: {eeg_channels}")
            self.args.channels = len(eeg_channels)
            self.original_order = eeg_channels

        # Read EEG data
        all_data = []
        for idx in eeg_indices:
            sig = reader.readSignal(idx)
            all_data.append(sig)
        reader.close()

        data = np.stack(all_data, axis=0).astype(np.float32)

        # Read sleep stage labels
        # Use label_type to select expert annotation (_1.txt or _2.txt)
        if self.label_type == 'V':
            label_file = subject_dir + '_2.txt'
        else:
            # Default to first expert's annotation (_1.txt)
            label_file = subject_dir + '_1.txt'

        label_path = os.path.join(self.data_path, subject_dir, label_file)
        if not os.path.exists(label_path):
            raise FileNotFoundError('Label file not found: {}'.format(label_path))

        with open(label_path, 'r') as f:
            labels = []
            for line in f:
                line = line.strip()
                if line:
                    labels.append(int(line))

        labels = np.array(labels, dtype=np.int64)

        # Split continuous data into 30-second epochs
        epoch_samples = int(self.args.segment * sampling_rate)
        n_epochs = len(labels)
        n_channels = data.shape[0]  # Use actual channel count

        # Check if data length matches expected epochs
        expected_samples = n_epochs * epoch_samples
        if data.shape[1] < expected_samples:
            # Pad if necessary
            pad_length = expected_samples - data.shape[1]
            data = np.pad(data, ((0, 0), (0, pad_length)), mode='constant')
        elif data.shape[1] > expected_samples:
            # Truncate if necessary
            data = data[:, :expected_samples]

        # Reshape to (epoch, channel, samples)
        # Use actual channel count (n_channels) instead of self.args.channels
        data = data.reshape(n_channels, n_epochs, epoch_samples)
        data = data.transpose(1, 0, 2)  # (epoch, channel, samples)

        # Reorder channels according to graph
        data = self.reorder_channel(data=data, graph=self.graph_type, channel_names=eeg_channels)

        self.args.sampling_rate = sampling_rate
        print('ISRUC data loaded: data shape {}, labels shape {}'.format(data.shape, labels.shape))
        return data, labels

    def reorder_channel(self, data, graph, channel_names=None):
        """
        This function reorder the channel according to different graph designs
        Parameters
        ----------
        data: (trial, channel, data)
        graph: graph type

        Returns
        -------
        reordered data: (trial, channel, data)
        """
        source_order = channel_names if channel_names is not None else self.original_order

        if self.dataset == 'EMO':
            if graph == 'fro':
                graph_idx = self.graph_fro_EMO
            elif graph == 'gen':
                graph_idx = self.graph_gen_EMO
            elif graph == 'hem':
                graph_idx = self.graph_hem_EMO
            elif graph == '10':
                graph_idx = self.graph_10_EMO
            else:
                graph_idx = self.graph_EMO
        elif self.dataset == 'EEGMAT':
            if graph == 'fro':
                graph_idx = self.graph_EEGMAT_fro
            elif graph == 'gen':
                graph_idx = self.graph_EEGMAT_gen
            elif graph == 'hem':
                graph_idx = self.graph_EEGMAT_hem
            elif graph == 'BL':
                graph_idx = self.original_order
            else:
                graph_idx = self.graph_EEGMAT
        elif self.dataset == 'ISRUC':
            if graph == 'fro':
                graph_idx = self.graph_ISRUC_fro
            elif graph == 'gen':
                graph_idx = self.graph_ISRUC_gen
            elif graph == 'hem':
                graph_idx = self.graph_ISRUC_hem
            elif graph == 'BL':
                graph_idx = self.original_order
            else:
                graph_idx = self.graph_ISRUC
        elif graph == 'fro':
            graph_idx = self.graph_fro_DEAP
        elif graph == 'gen':
            graph_idx = self.graph_gen_DEAP
        elif graph == 'hem':
            graph_idx = self.graph_hem_DEAP
        elif graph == 'BL':
            graph_idx = self.original_order

        idx = []
        if graph in ['BL'] and self.dataset != 'EMO':
            for chan in graph_idx:
                # idx.append(source_order.index(chan))
                if chan in source_order:
                    idx.append(source_order.index(chan))
                else:
                    print(f"Warning: Channel '{chan}' not found in source_order, skipping")
        else:
            num_chan_local_graph = []
            actual_num_chan = []
            for i in range(len(graph_idx)):
                num_chan_local_graph.append(len(graph_idx[i]))
                actual_count = 0
                for chan in graph_idx[i]:
                    # idx.append(source_order.index(chan))
                    if chan in source_order:
                        idx.append(source_order.index(chan))
                        actual_count += 1
                    else:
                        print(f"Warning: Channel '{chan}' not found in source_order, skipping")
                actual_num_chan.append(actual_count)

            # save the number of channels in local graph for building the LGG model in utils.py
            # Use actual counts instead of expected counts
            dataset = h5py.File('num_chan_local_graph_{}.hdf'.format(graph), 'w')
            dataset['data'] = actual_num_chan
            dataset.close()
        return data[:, idx, :]

    def get_era_graph_layout(self, graph):
        if graph == 'fro':
            return self.graph_fro_EMO
        if graph == 'gen':
            return self.graph_gen_EMO
        if graph == 'hem':
            return self.graph_hem_EMO
        if graph == '10':
            return self.graph_10_EMO
        return self.graph_EMO

    def reorder_to_era_layout(self, data, graph, channel_names):
        graph_idx = self.get_era_graph_layout(graph)
        idx = []
        actual_num_chan = []
        for local_group in graph_idx:
            actual_count = 0
            for chan in local_group:
                if chan in channel_names:
                    idx.append(channel_names.index(chan))
                    actual_count += 1
                else:
                    raise ValueError("ERA target channel '{}' missing after alignment".format(chan))
            actual_num_chan.append(actual_count)

        dataset = h5py.File('num_chan_local_graph_{}.hdf'.format(graph), 'w')
        dataset['data'] = actual_num_chan
        dataset.close()
        return data[:, idx, :]

    def remap_external32_to_era32(self, data, channel_names):
        target_data = np.zeros((data.shape[0], len(self.era_target_original_order), data.shape[-1]), dtype=np.float32)
        assigned_targets = set()
        for src_idx, src_ch in enumerate(channel_names):
            target_ch = self.external32_to_era32.get(src_ch)
            if target_ch is None:
                continue
            tgt_idx = self.era_target_original_order.index(target_ch)
            target_data[:, tgt_idx, :] = data[:, src_idx, :]
            assigned_targets.add(target_ch)

        missing = [ch for ch in self.era_target_original_order if ch not in assigned_targets]
        if missing:
            raise ValueError('Missing ERA target channels without interpolation: {}'.format(missing))
        return target_data

    def build_target_seed_from_source(self, samples_20):
        n_times = samples_20.shape[1]
        target_data = np.zeros((len(self.external_common_32_order), n_times), dtype=np.float32)
        known_mask = np.zeros(len(self.external_common_32_order), dtype=bool)
        source_to_target_rename = {
            'T3': 'T7',
            'T4': 'T8',
            'T5': 'P7',
            'T6': 'P8'
        }

        for src_idx, src_ch in enumerate(self.original_order):
            mapped_ch = source_to_target_rename.get(src_ch, src_ch)
            if mapped_ch in self.external_common_32_order:
                tgt_idx = self.external_common_32_order.index(mapped_ch)
                target_data[tgt_idx] = samples_20[src_idx]
                known_mask[tgt_idx] = True
        return target_data, known_mask

    def neighbor_average_fallback(self, data32, known_mask, k=4):
        montage = mne.channels.make_standard_montage('standard_1020')
        ch_pos = montage.get_positions()['ch_pos']
        coords = np.asarray([ch_pos[ch] for ch in self.external_common_32_order], dtype=np.float32)

        filled = data32.copy()
        known_idx = np.where(known_mask)[0]
        if len(known_idx) == 0:
            raise ValueError('No known channels available for fallback interpolation.')

        for idx in range(len(self.external_common_32_order)):
            ch_data = filled[idx]
            if np.isfinite(ch_data).all() and not np.allclose(ch_data, 0.0):
                continue

            dists = np.linalg.norm(coords[known_idx] - coords[idx], axis=1)
            dists = np.maximum(dists, 1e-6)
            order = np.argsort(dists)
            nn_idx = known_idx[order[:min(k, len(order))]]
            nn_dists = dists[order[:min(k, len(order))]]
            weights = 1.0 / nn_dists
            weights = weights / weights.sum()
            filled[idx] = np.sum(data32[nn_idx] * weights[:, None], axis=0)
        return filled.astype(np.float32)

    def graph_interpolation_expand(self, samples_20, sfreq):
        seed_data, known_mask = self.build_target_seed_from_source(samples_20)
        info = mne.create_info(ch_names=self.external_common_32_order, sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(seed_data.copy(), info, verbose=False)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')

        bads = [ch for ch, ok in zip(self.external_common_32_order, known_mask) if not ok]
        raw.info['bads'] = bads

        if known_mask.sum() >= 4 and len(bads) > 0:
            try:
                raw.interpolate_bads(reset_bads=False, mode='accurate', verbose=False)
            except Exception as e:
                print('[WARN] Graph interpolation failed, fallback to neighbor average: {}'.format(e))

        expanded = raw.get_data().astype(np.float32)
        expanded = self.neighbor_average_fallback(expanded, known_mask)
        return expanded

    def align_external_channels_to_era_layout(self, data, channel_names, already_era_aligned=False):
        if already_era_aligned:
            aligned = data.astype(np.float32)
        else:
            aligned = self.remap_external32_to_era32(data=data, channel_names=channel_names)
        return self.reorder_to_era_layout(
            data=aligned, graph=self.graph_type, channel_names=self.era_target_original_order)

    """老版label_selection"""

    def label_selection(self, label):
        """
        This function: 1. selects which dimension of labels to use
                       2. create binary label
        Parameters
        ----------
        label: (trial, 4)

        Returns
        -------
        label: (trial,)
        """
        if self.dataset == 'EMO':
            return label

        if self.dataset == 'EEGMAT':
            if self.label_type == 'Axis':
                return np.full(label.shape, 2, dtype=np.int64)
            return label

        if self.dataset == 'ISRUC':
            return label

        if self.label_type == 'Axis':

            # =====================
            # DEAP
            # =====================
            if self.dataset == 'DEAP':

                # DEAP:
                # label[:,1] = Arousal score
                arousal = label[:, 1]

                return np.where(
                    arousal <= 5,
                    0,  # 感性弱
                    1  # 感性强
                ).astype(np.int64)

            elif self.dataset == 'MEEG':

                # MEEG:
                # labels:
                # [:,0] Valence
                # [:,1] Arousal binary

                arousal_axis = label[:, 1]

                return np.where(
                    arousal_axis == 0,
                    0,  # 感性弱
                    1  # 感性强
                ).astype(np.int64)

        if self.label_type == 'V':
            label = label[:, 0]
        elif self.label_type == 'A':
            label = label[:, 1]
        elif self.label_type == 'D':
            label = label[:, 2]
        elif self.label_type == 'L':
            label = label[:, 3]
        if self.dataset == 'DEAP':
            label = np.where(label <= 5, 0, label)
            label = np.where(label > 5, 1, label)
        return label

    def save(self, data, label, sub):
        """
        This function save the processed data into target folder
        Parameters
        ----------
        data: the processed data
        label: the corresponding label
        sub: the subject ID

        Returns
        -------
        None
        """
        save_path = os.getcwd()
        data_type = 'data_{}_{}_{}'.format(self.args.data_format, self.args.dataset, self.args.label_type)
        save_path = os.path.join(save_path, data_type)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            pass
        name = 'sub' + str(sub) + '.hdf'
        save_path = os.path.join(save_path, name)
        with h5py.File(save_path, 'w') as dataset:
            dataset['data'] = data
            dataset['label'] = label
            dataset['era_label'] = label
            if 'original_label_raw' in self.save_metadata:
                dataset['original_label_raw'] = self.save_metadata['original_label_raw']

            if data.ndim == 5:
                trial_count, segment_count = data.shape[:2]
                trial_id = np.repeat(np.arange(trial_count, dtype=np.int64), segment_count)
                segment_id = np.tile(np.arange(segment_count, dtype=np.int64), trial_count)
            else:
                trial_id = np.arange(data.shape[0], dtype=np.int64)
                segment_id = np.zeros(data.shape[0], dtype=np.int64)
            dataset['sample_id'] = np.arange(trial_id.size, dtype=np.int64)
            dataset['trial_id'] = trial_id
            dataset['segment_id'] = segment_id
            dataset['subject_id'] = np.full(trial_id.size, sub, dtype=np.int64)

            graph_layout = self.get_era_graph_layout(self.graph_type) if self.dataset == 'EMO' else None
            channel_order = [channel for group in graph_layout for channel in group] if graph_layout else []
            dataset.attrs['source_dataset'] = self.save_metadata.get('source_dataset', self.dataset)
            dataset.attrs['graph_type'] = self.graph_type
            dataset.attrs['channel_order'] = json.dumps(channel_order, ensure_ascii=False)
            dataset.attrs['graph_structure'] = json.dumps(graph_layout or [], ensure_ascii=False)

    # 预处理数据
    def preprocess_data(self, data, label, split, expand, resting_data=None):
        """
        This function preprocess the data
        Parameters
        ----------
        data: (trial, channel, data)
        label: (trial,)
        split: (bool) whether to split one trial's data into shorter segment
        expand: (bool) whether to add an empty dimension for CNN
        resting_data: (channel, datapoint) resting state data for regression denoising (EMO only)
        Returns
        -------
        preprocessed
        data: (trial, channel, target_length)
        label: (trial,)
        """
        if expand:
            # expand one dimension for deep learning(CNNs)
            data = np.expand_dims(data, axis=-3)

        if self.args.dataset == 'MEEG':
            data = self.bandpass_filter(data=data, lowcut=self.args.bandpass[0], highcut=self.args.bandpass[1],
                                        fs=self.args.sampling_rate, order=5)
            data = self.notch_filter(data=data, fs=self.args.sampling_rate, Q=50)

        if self.args.dataset == 'EMO' and resting_data is not None and self.args.denoise:
            data = self.regression_denoise(data=data, resting_data=resting_data,
                                           n_components=self.args.denoise_components)

        if self.args.sampling_rate != self.args.target_rate:
            data, label = self.downsample_data(
                data=data, label=label, sampling_rate=self.args.sampling_rate,
                target_rate=self.args.target_rate)

        if split:
            data, label = self.split(data, label, self.args.segment, self.args.overlap, self.args.target_rate)

        return data, label

    def split(self, data, label, segment_length, overlap, sampling_rate):
        """
        This function split one trial's data into shorter segments
        Parameters
        ----------
        data: (trial, f, channel, data)
        label: (trial,)
        segment_length: how long each segment is (e.g. 1s, 2s,...)
        overlap: overlap rate
        sampling_rate: sampling rate

        Returns
        -------
        data:(tiral, num_segment, f, channel, segment_legnth)
        label:(trial, num_segment,)
        """
        data_shape = data.shape
        step = int(segment_length * sampling_rate * (1 - overlap))
        data_segment = sampling_rate * segment_length
        data_split = []

        number_segment = int((data_shape[-1] - data_segment) // step)
        for i in range(number_segment + 1):
            data_split.append(data[:, :, :, (i * step):(i * step + data_segment)])
        data_split_array = np.stack(data_split, axis=1)
        label = np.stack([np.repeat(label[i], int(number_segment + 1)) for i in range(len(label))], axis=0)
        print("The data and label are split: Data shape:" + str(data_split_array.shape) + " Label:" + str(
            label.shape))
        data = data_split_array
        assert len(data) == len(label)
        if self.args.model == 'DGCNN':
            data = self.extract_features(data, sampling_rate)
        return data, label

    def extract_features(self, data, sfreq):
        # data 的形状应该是 (20, 14, 1, 32, 800)
        num_trials, num_segments, _, num_channels, num_samples = data.shape
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 14),
            'beta': (14, 31),
            'gamma': (31, 50)
        }
        # 初始化输出数据结构
        features = np.zeros((num_trials, num_segments, 1, num_channels, len(bands)))

        # 遍历每个频带并应用滤波器
        for i, (band_name, band_range) in enumerate(bands.items()):
            for trial in range(num_trials):
                for segment in range(num_segments):
                    for channel in range(num_channels):
                        # 获取单个通道数据
                        channel_data = data[trial, segment, 0, channel, :]
                        # 应用带通滤波器
                        filtered_data = self.bandpass_filter(channel_data, band_range[0], band_range[1], sfreq)
                        # 计算差分熵
                        features[trial, segment, 0, channel, i] = np.log(np.var(filtered_data) + 1e-8)

        # 输出数据结构应该为 (20, 14, 1, 32, 5)
        return features

    def downsample_data(self, data, label, sampling_rate, target_rate):
        """
        This function downsample the data to target length
        Parameters
        ----------
        data: (trial, channel, data)
        label: (trial,)
        sampling_rate: original sampling rate
        target_rate: target sampling rate
        Returns
        -------
        downsampled data: (trial, channel, target_length)
        label: (trial,)
        """
        target_length = int(data.shape[-1] * target_rate / sampling_rate)
        downsampled_data = resample(data, target_length, axis=-1)
        return downsampled_data, label

    # 巴特沃斯带通滤波器
    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """
        This function applies bandpass filter to the data
        Parameters
        ----------
        data: (trial, channel, data)
        lowcut: low cut frequency
        highcut: high cut frequency
        fs: sampling rate
        order: filter order

        Returns
        -------
        filtered data: (trial, channel, data)
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='bandpass')
        filtered_data = signal.filtfilt(b, a, data, axis=-1)
        return filtered_data

    # 频率为50hz的陷波滤波器
    def notch_filter(self, data, fs, Q=50):
        """
        This function applies notch filter to the data
        Parameters
        ----------
        data: (trial, channel, data)
        fs: sampling rate
        Q: Q value for notch filter

        Returns
        -------
        filtered data: (trial, channel, data)
        """
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                self.notch_filter_per_channel(data[i, j, :], fs, Q)
        return data

    def notch_filter_per_channel(self, param, fs, Q):
        """
        This function applies notch filter to one channel
        Parameters
        ----------
        param: (data,)
        fs: sampling rate
        Q: Q value for notch filter

        Returns
        -------
        filtered data: (data,)
        """
        w0 = Q / fs
        b, a = signal.iirnotch(w0, Q)
        param = signal.filtfilt(b, a, param)
        return param

    def regression_denoise(self, data, resting_data, n_components=5):
        """
        利用静息态数据通过PCA提取噪声模式，从任务数据中回归去除
        Parameters
        ----------
        data: (trial, channel, datapoint) 或 (trial, f, channel, datapoint)
        resting_data: (channel, datapoint) 静息态数据
        n_components: PCA提取的噪声成分数

        Returns
        -------
        denoised data: 与输入形状相同
        """
        # 处理可能的额外维度 (trial, f, channel, datapoint)
        has_feature_dim = data.ndim == 4
        if has_feature_dim:
            original_shape = data.shape
            # 合并 trial 和 feature 维度
            data_3d = data.reshape(-1, data.shape[-2], data.shape[-1])
        else:
            data_3d = data

        n_trials, n_channels, n_samples = data_3d.shape
        n_rest_samples = resting_data.shape[-1]

        # 对静息态数据做PCA，提取主要噪声模式
        # resting_data: (channel, datapoint) -> 转置为 (datapoint, channel)
        rest_matrix = resting_data[:, :n_rest_samples].T  # (datapoint, channel)

        # 中心化
        rest_mean = np.mean(rest_matrix, axis=0, keepdims=True)
        rest_centered = rest_matrix - rest_mean

        # PCA拟合，提取前n_components个主成分作为噪声基
        pca = PCA(n_components=n_components)
        pca.fit(rest_centered)
        # noise_basis: (n_components, channel) 每行是一个噪声空间模式
        noise_basis = pca.components_

        # 对每个trial，用最小二乘回归去除噪声成分
        denoised = np.zeros_like(data_3d)
        for i in range(n_trials):
            trial_data = data_3d[i]  # (channel, datapoint)
            trial_matrix = trial_data.T  # (datapoint, channel)
            trial_centered = trial_matrix - rest_mean

            # 最小二乘: trial_centered ≈ coefficients @ noise_basis
            # coefficients: (datapoint, n_components)
            coefficients = trial_centered @ noise_basis.T  # (datapoint, n_components)
            noise_estimate = coefficients @ noise_basis  # (datapoint, channel)

            # 去噪
            denoised_trial = trial_matrix - noise_estimate
            denoised[i] = denoised_trial.T  # (channel, datapoint)

        if has_feature_dim:
            denoised = denoised.reshape(original_shape)

        print('Regression denoising applied: n_components={}'.format(n_components))
        return denoised
