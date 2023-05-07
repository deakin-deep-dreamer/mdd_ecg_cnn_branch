import os
import sys
import random
import traceback
import matplotlib.pyplot as plt

# import pyedflib
import numpy as np

from scipy.stats import zscore
from scipy import signal
from scipy.io import loadmat
from sklearn import preprocessing

import torch
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

import ts2vg
from ecgdetectors import Detectors
# https://github.com/berndporr/py-ecg-detectors


def log(msg):
    print(msg)


def read_mat_file(
    mat_file
):
    r"""Read mat file, expecting 3 signals."""
    data = loadmat(mat_file)['data']
    return data


def zscore_hz_wise(
    sig, hz=None
):
    r"""Scale raw signal in (n_sample,) or (n_sample, n_chan) format.

    Second-wise mean difference.
    """
    if hz is None:
        raise ValueError("Hz expected, received None.")
    i_window = 0
    is_done = False

    if len(sig.shape) == 1:
        return zscore_hz_wise_single_chan(sig, hz)

    n_samp, n_chan = sig.shape
    sig_new = np.zeros(sig.shape)
    while True:
        i_start = hz*i_window
        if n_samp - hz*i_window < hz:
            i_end_seg = i_start + (n_samp - hz*i_window)
            is_done = True
        else:
            i_end_seg = i_start + hz
        for i_chan in range(n_chan):
            sig_new[i_start:i_end_seg, i_chan] = zscore(
                sig[i_start:i_end_seg, i_chan])
        i_window += 1
        if is_done:
            break
    # print(f"[scale] new-signal: {new_sig.shape}")
    assert sig.shape[0] == sig_new.shape[0]
    return sig_new


def zscore_hz_wise_single_chan(
    sig, hz=None
):
    r"""Scale raw signal. Second-wise mean difference"""
    new_sig = []
    i_window = 0
    is_done = False
    while True:
        i_start = hz*i_window
        if len(sig) - hz*i_window < hz:
            i_end_seg = i_start + (len(sig) - hz*i_window)
            is_done = True
        else:
            i_end_seg = i_start + hz
        sec_seg = sig[i_start:i_end_seg]
        new_sig.extend(
            # (np.array(sec_seg) - np.mean(sec_seg))
            zscore(sec_seg)
        )
        i_window += 1
        if is_done:
            break
    new_sig = np.array(new_sig)
    # print(f"[scale] new-signal: {new_sig.shape}")
    assert sig.shape[0] == new_sig.shape[0]
    return new_sig


def get_rr_signal(y, hz, target_n_samp):
    detectors = Detectors(hz)
    r_peaks = detectors.pan_tompkins_detector(y)
    rr_signal = np.diff(r_peaks) / hz  # normalise for 0-1 range
    rr_signal = rr_signal[(rr_signal >= 0.5) & (rr_signal <= 1.0)]  # ignore < 0.5
    # print(rr_signal.shape, "after filter.")
    rr_signal = signal.resample(rr_signal, target_n_samp) if rr_signal.shape[0]>5 else None
    return rr_signal


class MddRrDataset_old(Dataset):
    def __init__(
            self, input_dir=None, seg_sz=None, hz=None, n_subjects=-1, 
            seg_slide_sec=1, hz_down_factor=None, max_sig_len=None, np_data=False, 
            hrv=True, log=print) -> None:
        super().__init__()
        self.log = log
        self.input_dir = input_dir
        self.seg_sz = seg_sz
        self.hz = hz
        self.n_subjects=n_subjects
        self.seg_slide_sec = seg_slide_sec
        self.np_data = np_data
        self.hrv = hrv

        self.record_names = []
        self.records = {}
        self.record_wise_segments = {}
        self.segments = []
        self.seg_labels = []

        self.initialise()

    def initialise(self):
        self.log(
            f"[{self.__class__.__name__}] hz:{self.hz}, "
            f"seg_sz:{self.seg_sz}, ")
        self.header_files = []
        count_rec = 0
        for f in os.listdir(self.input_dir):
            if not self.np_data and not f.endswith(".mat"):
                continue
            if self.np_data and not f.endswith(".csv"):
                continue

            r"temporary, use few records."
            count_rec += 1
            if self.n_subjects > -1 and count_rec > self.n_subjects:
                break

            g = os.path.join(self.input_dir, f)
            r"For each mat file."
            r"Extract recordname from header file path"
            if not self.np_data:
                self.record_names.append(f[:f.find('.mat')])
                r"initialise record-wise segment index storage"
                self.record_wise_segments[self.record_names[-1]] = []

                data = read_mat_file(g).reshape((-1))

                n_sig_per_modality = data.shape[0] // 3
                signals = np.zeros((n_sig_per_modality, 3))
                self.log(f"mat_signal:{data.shape}")
                i_sig = 0
                for i_signal in range(0, data.shape[0], n_sig_per_modality):
                    signals[:, i_sig] = data[i_signal:i_signal+n_sig_per_modality]
                    i_sig += 1
                    if i_sig >= 3:
                        break
                ecg_signal = signals[:, 2]

            else:
                self.record_names.append(f[:f.find('.csv')])
                r"initialise record-wise segment index storage"
                self.record_wise_segments[self.record_names[-1]] = []
                ecg_signal = np.loadtxt(f"{self.input_dir}/{f}")

            self.log(f"ECG signal:{ecg_signal.shape}")
            detectors = Detectors(self.hz)
            # r_peaks = detectors.pan_tompkins_detector(ecg_signal)
            r_peaks = detectors.engzee_detector(ecg_signal)
            rr_signal = np.diff(r_peaks)  # R-R signal
            rr_signal = zscore(rr_signal)  # normalise
            self.log(f"RR signal:{len(rr_signal)}")
            self.records[self.record_names[-1]] = rr_signal
            n_window = len(rr_signal) // self.seg_sz

            n_samples = rr_signal.shape[0]
            n_window = 1 + \
                (n_samples - self.seg_sz) // (self.seg_slide_sec)
            for i_window in range(n_window):
                # start = i_window * self.seg_sz
                start = i_window * self.seg_slide_sec
                t_segment = (self.record_names[-1], start)
                self.segments.append(t_segment)
                self.record_wise_segments[self.record_names[-1]].append(
                    len(self.segments)-1)

                r"Record names start with 0 (001.mat) are patients (45 patient)\
                and normal subject name starts with 1 (101.mat, 46 controls)."
                self.seg_labels.append(
                    1 if f[0] == '0' else 0
                )
        class_dist = np.unique(
            [self.seg_labels[i]
                for i in range(len(self.seg_labels))], return_counts=True
        )
        self.log(
            f"Segmentation done. Seg:{len(self.segments)}, "
            f"labels:{len(self.seg_labels)}, "
            f"class-dist:{class_dist}")


class MDDCsvRRDataset_old(Dataset):
    def __init__(self, input_dir=None, seg_sec=10, seg_slide_sec=None, n_seg_per_sub=None, log=print) -> None:
        super().__init__()
        self.log = log
        self.hz = 100
        self.input_dir = input_dir
        self.seg_sec = seg_sec
        self.seg_slide_sec = seg_slide_sec
        self.n_seg_per_sub = n_seg_per_sub
        self.seg_sz = seg_sec

        self.record_names = []
        self.record_wise_segments = {}
        self.segments = []
        self.seg_labels = []

        self.initialise()

    def initialise(self):
        self.log(
            f"[{self.__class__.__name__}] hz:{self.hz}, "
            f"seg_sec:{self.seg_sec}, seg_slide_sec:{self.seg_slide_sec}")
        self.header_files = []
        for f in os.listdir(self.input_dir):
            if not f.endswith(".csv"):
                continue

            rec_name = f[:f.find('.csv')]    
            self.record_names.append(rec_name)
            r"initialise record-wise segment index storage"
            self.record_wise_segments[rec_name] = []

            if self.seg_slide_sec is not None:
                "sliding window"
                pass
            else:
                "random segment selection"
                random.seed(2023)
                ecg_signal = np.loadtxt(f"{self.input_dir}/{f}")
                # self.log(f"ECG signal:{ecg_signal.shape}")
                detectors = Detectors(self.hz)
                r_peaks = detectors.pan_tompkins_detector(ecg_signal)
                rr_signal = np.diff(r_peaks)  # R-R signal
                rr_signal = zscore(rr_signal)  # normalise
                self.log(f"ECG {ecg_signal.shape}, RR:{rr_signal.shape}")
                for _ in range(self.n_seg_per_sub):
                    start = random.randint(0, rr_signal.shape[0]-self.seg_sz)
                    end = start + self.seg_sz
                    seg = rr_signal[start:end]
                    seg = np.expand_dims(seg, axis=0).T
                    self.segments.append(seg)
                    self.seg_labels.append(1 if f[0] == '0' else 0)
                    self.record_wise_segments[rec_name].append(len(self.segments)-1)

        class_dist = np.unique(
            [self.seg_labels[i]
                for i in range(len(self.seg_labels))], return_counts=True)
        self.log(
            f"Segmentation done. Seg:{len(self.segments)}, "
            f"labels:{len(self.seg_labels)}, "
            f"class-dist:{class_dist}")


class MDDCsvDataset(Dataset):
    def __init__(
            self, input_dir=None, seg_sec=10, seg_slide_sec=None, 
            n_seg_per_sub=None, hz=100, n_subjects=-1, log=print) -> None:
        super().__init__()
        self.log = log 
        self.hz = hz
        self.input_dir = input_dir
        self.seg_sec = seg_sec
        self.seg_slide_sec = seg_slide_sec
        self.n_seg_per_sub = n_seg_per_sub
        self.seg_sz = self.hz * seg_sec
        self.n_subjects = n_subjects

        self.record_names = []
        self.record_wise_segments = {}
        self.segments = []
        self.seg_labels = []

        self.initialise()

    def initialise(self):
        self.log(
            f"[{self.__class__.__name__}] hz:{self.hz}, "
            f"seg_sec:{self.seg_sec}, seg_slide_sec:{self.seg_slide_sec}")
        self.header_files = []
        count_rec = 0
        for f in os.listdir(self.input_dir):
            if not f.endswith(".csv"):
                continue

            count_rec += 1
            if self.n_subjects > -1 and count_rec > self.n_subjects:
                break

            rec_name = f[:f.find('.csv')]    
            self.record_names.append(rec_name)
            r"initialise record-wise segment index storage"
            self.record_wise_segments[rec_name] = []

            ecg_signal = np.loadtxt(f"{self.input_dir}/{f}")
            if self.seg_slide_sec is not None:
                "sliding window"
                n_samples = ecg_signal.shape[0]
                n_window = 1 + \
                    (n_samples - self.seg_sz) // (self.hz * self.seg_slide_sec)
                self.log(f"\t[{self.record_names[-1]}] segments:{n_window}")
                for i_window in range(n_window):
                    start = i_window * self.hz * self.seg_slide_sec
                    segment = ecg_signal[start:start+self.seg_sz]    

                    # sanity check if signal quality is acceptable
                    rr_signal = get_rr_signal(segment, self.hz, self.seg_sec)
                    if rr_signal is None:
                        continue

                    self.segments.append(segment)
                    self.record_wise_segments[self.record_names[-1]].append(
                        len(self.segments)-1)

                    r"Record names start with 0 (001.mat) are patients (45 patient)\
                    and normal subject name starts with 1 (101.mat, 46 controls)."
                    self.seg_labels.append(
                        1 if f[0] == '0' else 0
                    )
            else:
                "random segment selection"
                random.seed(2023)
                # ecg_signal = np.loadtxt(f"{self.input_dir}/{f}")
                for _ in range(self.n_seg_per_sub):
                    start = random.randint(0, ecg_signal.shape[0]-self.seg_sz)
                    end = start + self.seg_sz
                    seg = ecg_signal[start:end]
                    seg = np.expand_dims(seg, axis=0).T
                    self.segments.append(seg)
                    self.seg_labels.append(1 if f[0] == '0' else 0)
                    self.record_wise_segments[rec_name].append(len(self.segments)-1)

        class_dist = np.unique(
            [self.seg_labels[i]
                for i in range(len(self.seg_labels))], return_counts=True)
        self.log(
            f"Segmentation done. Seg:{len(self.segments)}, "
            f"labels:{len(self.seg_labels)}, "
            f"class-dist:{class_dist}")


class PartialCsvDataset(Dataset):
    r"""Deliver partial view of the parent dataset."""

    def __init__(
        self, dataset=None, seg_index=None, test=False, shuffle=False,
        as_np=False, data_cube=False, i_chan=None
    ):
        r"""Construct partial dataset instance."""
        self.memory_ds = dataset
        self.indexes = seg_index[:]
        self.test = test
        self.shuffle = shuffle
        self.as_np = as_np
        self.data_cube = data_cube

    def on_epoch_end(self):
        r"""End of epoch."""
        if self.shuffle and not self.test:
            np.random.shuffle(self.indexes)

    def __len__(self):
        r"""Dataset length."""
        return len(self.indexes)

    def __getitem__(self, idx):
        r"""Find and return item."""
        ID = self.indexes[idx]
        trainX = self.memory_ds.segments[ID]
        trainY = self.memory_ds.seg_labels[ID]


        if self.data_cube:
            "x: Stack segment of 1 sec"
            signal_stack = None
            block_sz = self.memory_ds.seg_sz // self.memory_ds.seg_sec
            for i in range(2):
                for i_block in range(self.memory_ds.seg_sec):
                    start = i_block * block_sz 
                    block = trainX[start:start+block_sz]
                    if signal_stack is None:
                        signal_stack = np.array(block)
                    else:
                        signal_stack = np.column_stack((signal_stack, np.array(block)))            
                    # print(f"block-{i_block}: {signal_stack.shape}")
            trainX = signal_stack

        if self.as_np:
            return trainX, trainY

        X_tensor = Variable(torch.from_numpy(trainX)).type(torch.FloatTensor)
        # print(f"X_tensor before: {X_tensor.size()}")
        r"numpy array shape: (n_samp, n_chan), reshape it to (n_chan, n-samp)."
        X_tensor = X_tensor.reshape(X_tensor.size()[1], -1)
        # print(f"X_tensor after: {X_tensor.size()}")
        # Y_tensor = torch.from_numpy(trainY).type(torch.FloatTensor)
        Y_tensor = trainY

        if torch.any(torch.isnan(X_tensor)):
            # self.memory_ds.log(f"[partial-ds] NaN found for idx:{idx}")
            X_tensor = torch.nan_to_num(X_tensor)

        return X_tensor, Y_tensor
    

class MddDataset(Dataset):
    r"""Major Depressive Disorder (MDD) dataset."""

    def __init__(
        self, input_dir=None, hz=None, hz_down_factor=None, 
        seg_sec=None, n_chan=1,seg_slide_sec=None, n_skip_segments=0, 
        max_sig_len=-1, n_subjects=-1, log=print, np_data=True
    ):
        r"""Instantiate database."""
        super(MddDataset, self).__init__()
        self.log = log
        self.input_dir = input_dir
        self.hz = hz
        self.hz_down_factor = hz_down_factor
        self.seg_sec = seg_sec
        self.n_chan = n_chan
        self.seg_slide_sec = seg_slide_sec
        self.n_skip_segments = n_skip_segments
        self.seg_sz = self.seg_sec * self.hz
        self.max_sig_len = max_sig_len
        self.n_subjects = n_subjects
        self.np_data = np_data
        self.ecg_detector = Detectors(self.hz)

        self.record_names = []
        self.records = {}
        self.record_wise_segments = {}
        self.segments = []
        self.seg_labels = []

        self.initialize()

        r"Shuffle segment global indexe."
        self.indexes = [i for i in range(len(self.segments))]
        np.random.shuffle(self.indexes)

        random.shuffle(self.record_names)

    def initialize(self):
        r"""Initialize database."""
        self.log(
            f"[{self.__class__.__name__}] hz:{self.hz}, "
            f"hz_down_factor:{self.hz_down_factor}, seg_sec:{self.seg_sec}, "
            f"seg_slide_sec:{self.seg_slide_sec}")
        self.header_files = []
        count_rec = 0
        for f in os.listdir(self.input_dir):
            # if not f.endswith(".mat"):
            #     continue
            if not self.np_data and not f.endswith(".mat"):
                continue
            if self.np_data and not f.endswith(".csv"):
                continue

            r"temporary, use few records."
            count_rec += 1
            if self.n_subjects > -1 and count_rec > self.n_subjects:
                break

            g = os.path.join(self.input_dir, f)
            r"For each mat file."
            r"Extract recordname from header file path"

            if not self.np_data:
                self.record_names.append(f[:f.find('.mat')])
                r"initialise record-wise segment index storage"
                self.record_wise_segments[self.record_names[-1]] = []

                data = read_mat_file(g).reshape((-1))

                n_sig_per_modality = data.shape[0] // 3
                signals = np.zeros((n_sig_per_modality, 3))
                self.log(f"mat_signal:{data.shape}")
                i_sig = 0
                for i_signal in range(0, data.shape[0], n_sig_per_modality):
                    signals[:, i_sig] = data[i_signal:i_signal+n_sig_per_modality]
                    i_sig += 1
                    if i_sig >= 3:
                        break
                data = signals[:, 2]

                src_hz = 1000
                target_num_sample = self.hz * (data.shape[0] // src_hz)
                self.log(
                f"Downsample {data.shape} to {target_num_sample}")    
                data = signal.resample(
                    data, target_num_sample)
                ecg_signal = zscore(data)

            else:
                self.record_names.append(f[:f.find('.csv')])
                r"initialise record-wise segment index storage"
                self.record_wise_segments[self.record_names[-1]] = []
                ecg_signal = np.loadtxt(f"{self.input_dir}/{f}")
                # CSV data already 100Hz and normalised.


            ecg_signal = np.expand_dims(ecg_signal, axis=1)
            
            # normalisation
            scaler = preprocessing.RobustScaler()
            ecg_signal = scaler.fit_transform(ecg_signal)
            scaler = preprocessing.MinMaxScaler()
            ecg_signal = scaler.fit_transform(ecg_signal)
            # signal = signal.flatten()

            # self.record_names.append(
            #     f[:f.find('.mat')])
            # r"initialise record-wise segment index storage"
            # self.record_wise_segments[self.record_names[-1]] = []
            # data = read_mat_file(g).reshape((-1))
            
            # src_hz = 1000
            # target_num_sample = self.hz * (data.shape[0] // src_hz)
            # self.log(
            # f"Downsample {data.shape} to {target_num_sample}")    
            # data = signal.resample(
            #     data, target_num_sample)

            # r"Scale data."
            # data = zscore_hz_wise(data, hz=2*self.hz)
            # data = zscore(data)

            # n_sig_per_modality = data.shape[0] // 3
            # signals = np.zeros((n_sig_per_modality, 3))
            # i_sig = 0
            # for i_signal in range(0, data.shape[0], n_sig_per_modality):
            #     signals[:, i_sig] = data[i_signal:i_signal+n_sig_per_modality]
            #     # signals[:, i_sig] = scale(
            #     #     data[i_signal:i_signal+n_sig_per_modality],
            #     #     sec_wise=True, hz=self.hz)
            #     i_sig += 1
            #     if i_sig >= 3:
            #         break
            # prev_shape = signals.shape
            # if self.max_sig_len > -1 and n_sig_per_modality > self.max_sig_len:
            #     r"Limit each data modality to the specified length."
            #     signals = signals[:self.max_sig_len, :]
            # self.log(
            #     f"signal:{prev_shape}, signal*:{signals.shape}")
            self.records[self.record_names[-1]] = ecg_signal

            r"Store records by record-name. \
            Store (rec-name, segment start index) to later form segment."
            n_samples = ecg_signal.shape[0]
            n_window = 1 + \
                (n_samples - self.seg_sz) // (self.hz * self.seg_slide_sec)
            self.log(f"\t[{self.record_names[-1]}] segments:{n_window}")
            for i_window in range(n_window):
                start = i_window * self.hz * self.seg_slide_sec
                t_segment = (self.record_names[-1], start)
                
                # self.segments.append(t_segment)
                segment = ecg_signal[start:start+self.seg_sz, :]
                # # r-r signal
                # rr_signal = np.zeros((segment.shape[0]))
                # r_peaks = self.ecg_detector.pan_tompkins_detector(segment[:, 0])
                # r_peak_diff = np.diff(r_peaks)
                # rr_signal[0:len(r_peak_diff)] = r_peak_diff
                # segment = np.vstack((segment.flatten(), rr_signal)).T

                self.segments.append(segment)

                self.record_wise_segments[self.record_names[-1]].append(
                    len(self.segments)-1)

                r"Record names start with 0 (001.mat) are patients (45 patient)\
                and normal subject name starts with 1 (101.mat, 46 controls)."
                self.seg_labels.append(
                    1 if f[0] == '0' else 0
                )
        class_dist = np.unique(
            [self.seg_labels[i]
                for i in range(len(self.seg_labels))], return_counts=True)
        self.log(
            f"Segmentation done. Seg:{len(self.segments)}, "
            f"labels:{len(self.seg_labels)}, "
            f"class-dist:{class_dist}")


class MddRRDataset(Dataset):
    r"""Major Depressive Disorder (MDD) dataset."""

    def __init__(
        self, input_dir=None, hz=None, hz_down_factor=None, 
        seg_sec=None, n_chan=1,seg_slide_sec=1, n_skip_segments=0, 
        max_sig_len=-1, n_subjects=-1, log=print
    ):
        r"""Instantiate database."""
        super(MddRRDataset, self).__init__()
        self.log = log
        self.input_dir = input_dir
        self.hz = hz
        self.hz_down_factor = hz_down_factor
        self.seg_sec = seg_sec
        self.n_chan = n_chan
        self.seg_slide_sec = seg_slide_sec
        self.n_skip_segments = n_skip_segments
        self.seg_sz = self.seg_sec * self.hz
        self.max_sig_len = max_sig_len
        self.n_subjects = n_subjects
        self.ecg_detector = Detectors(self.hz)

        self.record_names = []
        self.records = {}
        self.record_wise_segments = {}
        self.segments = []
        self.seg_labels = []

        self.initialize()

        r"Shuffle segment global indexe."
        self.indexes = [i for i in range(len(self.segments))]
        np.random.shuffle(self.indexes)

        random.shuffle(self.record_names)

    def initialize(self):
        r"""Initialize database."""
        self.log(
            f"[{self.__class__.__name__}] hz:{self.hz}, "
            f"hz_down_factor:{self.hz_down_factor}, seg_sec:{self.seg_sec}, "
            f"seg_slide_sec:{self.seg_slide_sec}")
        self.header_files = []
        count_rec = 0
        for f in os.listdir(self.input_dir):
            if not f.endswith(".csv"):
                continue

            r"temporary, use few records."
            count_rec += 1
            if self.n_subjects > -1 and count_rec > self.n_subjects:
                break
            self.record_names.append(f[:f.find('.csv')])
            r"initialise record-wise segment index storage"
            self.record_wise_segments[self.record_names[-1]] = []
            ecg_signal = np.loadtxt(os.path.join(self.input_dir, f))
            # self.records[self.record_names[-1]] = ecg_signal

            r"Store records by record-name. \
            Store (rec-name, segment start index) to later form segment."
            n_samples = ecg_signal.shape[0]
            n_window = 1 + \
                (n_samples - self.seg_sz) // (self.hz * self.seg_slide_sec)
            self.log(f"\t[{self.record_names[-1]}] segments:{n_window}")
            for i_window in range(n_window):
                start = i_window * self.hz * self.seg_slide_sec
                segment = ecg_signal[start:start+self.seg_sz]
                # r-r signal
                rr_signal = get_rr_signal(segment, self.hz, self.seg_sec)   
                if rr_signal is None:
                    continue
                # rr_signal = np.zeros((1*self.seg_sec))  # assume max 1 r peaks per seconds
                # r_peaks = self.ecg_detector.pan_tompkins_detector(segment)
                # r_peak_diff = np.diff(r_peaks)
                # if len(r_peak_diff) > len(rr_signal):  # trim excess R peaks
                #     r_peak_diff = r_peak_diff[:len(rr_signal)]
                # rr_signal[0:len(r_peak_diff)] = r_peak_diff
                # print(rr_signal)
                # rr_signal = zscore(rr_signal)  # normalise
                # rr_signal = np.expand_dims(rr_signal, axis=1)
                # scaler = preprocessing.RobustScaler()
                # rr_signal = scaler.fit_transform(rr_signal)
                # scaler = preprocessing.MinMaxScaler()
                # rr_signal = scaler.fit_transform(rr_signal)

                if len(rr_signal.shape) == 1:
                    rr_signal = np.expand_dims(rr_signal, axis=1)
                self.segments.append(rr_signal)

                self.record_wise_segments[self.record_names[-1]].append(
                    len(self.segments)-1)

                r"Record names start with 0 (001.mat) are patients (45 patient)\
                and normal subject name starts with 1 (101.mat, 46 controls)."
                self.seg_labels.append(
                    1 if f[0] == '0' else 0
                )
        class_dist = np.unique(
            [self.seg_labels[i]
                for i in range(len(self.seg_labels))], return_counts=True)
        self.log(
            f"Segmentation done. Seg:{len(self.segments)}, "
            f"labels:{len(self.seg_labels)}, "
            f"class-dist:{class_dist}")
        

class PartialRRDataset(Dataset):
    r"""Deliver partial view of the parent dataset."""

    def __init__(
        self, dataset=None, seg_index=None, test=False, shuffle=False, as_np=False,
        i_chan=-1, derived=False
    ):
        r"""Construct partial dataset instance."""
        self.memory_ds = dataset
        self.indexes = seg_index[:]
        self.test = test
        self.shuffle = shuffle
        self.as_np = as_np
        self.derived = derived
        self.g = ts2vg.HorizontalVG()

    def on_epoch_end(self):
        r"""End of epoch."""
        if self.shuffle and not self.test:
            np.random.shuffle(self.indexes)

    def __len__(self):
        r"""Dataset length."""
        return len(self.indexes)

    def __getitem__(self, idx):
        r"""Find and return item."""
        ID = self.indexes[idx]
        trainX = self.memory_ds.segments[ID]
        if self.derived:
            self.g.build(trainX.flatten(), only_degrees=True)
            trainX = np.vstack((trainX.flatten(), zscore(self.g.degrees.flatten()))).T
        trainY = self.memory_ds.seg_labels[ID]

        if self.as_np:
            return trainX, trainY

        X_tensor = Variable(torch.from_numpy(trainX)).type(torch.FloatTensor)
        # print(f"X_tensor before: {X_tensor.size()}")
        r"numpy array shape: (n_samp, n_chan), reshape it to (n_chan, n-samp)."
        X_tensor = X_tensor.reshape(X_tensor.size()[1], -1)
        # print(f"X_tensor after: {X_tensor.size()}")
        # Y_tensor = torch.from_numpy(trainY).type(torch.FloatTensor)
        Y_tensor = trainY

        if torch.any(torch.isnan(X_tensor)):
            # self.memory_ds.log(f"[partial-ds] NaN found for idx:{idx}")
            X_tensor = torch.nan_to_num(X_tensor)

        return X_tensor, Y_tensor
    

class PartialDataset(MddDataset):
    r"""Deliver partial view of the parent dataset."""

    def __init__(
        self, dataset=None, seg_index=None, test=False, shuffle=False, i_chan=-1,
        as_np=False
    ):
        r"""Construct partial dataset instance."""
        self.memory_ds = dataset
        self.indexes = seg_index[:]
        self.test = test
        self.shuffle = shuffle
        self.i_chan = i_chan
        self.as_np = as_np

    def on_epoch_end(self):
        r"""End of epoch."""
        if self.shuffle and not self.test:
            np.random.shuffle(self.indexes)

    def __len__(self):
        r"""Dataset length."""
        return len(self.indexes)

    def __getitem__(self, idx):
        r"""Find and return item."""
        ID = self.indexes[idx]
        # rec_name, idx_seg = self.memory_ds.segments[ID]
        # signal = self.memory_ds.records[rec_name]
        # # trainX = signal[idx_seg:idx_seg+self.memory_ds.seg_sz, :]
        # trainX = signal[idx_seg:idx_seg+self.memory_ds.seg_sz, :]
        trainX = self.memory_ds.segments[ID]

        # if self.i_chan == 13:
        #     # combination of channel 1 & 3
        #     trainX = np.c_[trainX[:, 0], trainX[:, 2]]
        # elif self.i_chan in [0, 1, 2]:
        #     trainX = trainX[:, self.i_chan]
        #     trainX = np.expand_dims(trainX, axis=1)
        trainY = self.memory_ds.seg_labels[ID]


        if self.as_np:
            return trainX, trainY

        X_tensor = Variable(torch.from_numpy(trainX)).type(torch.FloatTensor)
        # print(f"X_tensor before: {X_tensor.size()}")
        r"numpy array shape: (n_samp, n_chan), reshape it to (n_chan, n-samp)."
        X_tensor = X_tensor.reshape(X_tensor.size()[1], -1)
        # print(f"X_tensor after: {X_tensor.size()}")
        # Y_tensor = torch.from_numpy(trainY).type(torch.FloatTensor)
        Y_tensor = trainY

        if torch.any(torch.isnan(X_tensor)):
            # self.memory_ds.log(f"[partial-ds] NaN found for idx:{idx}")
            X_tensor = torch.nan_to_num(X_tensor)

        return X_tensor, Y_tensor


class PartialDerivedDataset(Dataset):
    r"""Deliver partial view of the parent dataset."""

    def __init__(
        self, dataset=None, seg_index=None, test=False, shuffle=False, i_chan=-1,
        rr_signal=True, as_np=False
    ):
        r"""Construct partial dataset instance."""
        self.memory_ds = dataset
        self.indexes = seg_index[:]
        self.test = test
        self.shuffle = shuffle
        self.i_chan = i_chan
        self.as_np = as_np
        self.rr_signal = rr_signal
        self.g = ts2vg.HorizontalVG()
        # self.memory_ds.log(
        #     f"[{self.__class__.__name__}] rr_signal:{rr_signal}, seg_index:{seg_index}")

    def on_epoch_end(self):
        r"""End of epoch."""
        if self.shuffle and not self.test:
            np.random.shuffle(self.indexes)

    def __len__(self):
        r"""Dataset length."""
        return len(self.indexes)

    def __getitem__(self, idx):
        r"""Find and return item."""
        ID = self.indexes[idx]
        trainX = self.memory_ds.segments[ID]

        if self.rr_signal:
            rr_signal = get_rr_signal(trainX, self.memory_ds.hz, self.memory_ds.seg_sz)
            trainX = np.vstack((trainX.flatten(), rr_signal.flatten())).T
        else:
            self.g.build(trainX.flatten(), only_degrees=True)
            trainX = np.vstack((trainX.flatten(), zscore(self.g.degrees))).T
            # print("trainX*:", trainX.shape)

        trainY = self.memory_ds.seg_labels[ID]


        if self.as_np:
            return trainX, trainY

        X_tensor = Variable(torch.from_numpy(trainX)).type(torch.FloatTensor)
        # print(f"X_tensor before: {X_tensor.size()}")
        r"numpy array shape: (n_samp, n_chan), reshape it to (n_chan, n-samp)."
        X_tensor = X_tensor.reshape(X_tensor.size()[1], -1)
        # print(f"X_tensor after: {X_tensor.size()}")
        # Y_tensor = torch.from_numpy(trainY).type(torch.FloatTensor)
        Y_tensor = trainY

        if torch.any(torch.isnan(X_tensor)):
            # self.memory_ds.log(f"[partial-ds] NaN found for idx:{idx}")
            X_tensor = torch.nan_to_num(X_tensor)

        return X_tensor, Y_tensor


class PartialRawAndRRDataset(Dataset):
    r"""Deliver partial view of the parent dataset."""

    def __init__(
        self, dataset=None, seg_index=None, test=False, shuffle=False, i_chan=-1,
        as_np=False, rr_signal=True
    ):
        r"""Construct partial dataset instance."""
        self.memory_ds = dataset
        self.indexes = seg_index[:]
        self.test = test
        self.shuffle = shuffle
        self.i_chan = i_chan
        self.as_np = as_np
        self.rr_signal = rr_signal
        self.ecg_detector = Detectors(self.hz)

    def on_epoch_end(self):
        r"""End of epoch."""
        if self.shuffle and not self.test:
            np.random.shuffle(self.indexes)

    def __len__(self):
        r"""Dataset length."""
        return len(self.indexes)

    def __getitem__(self, idx):
        r"""Find and return item."""
        ID = self.indexes[idx]
        trainX = self.memory_ds.segments[ID]
        if self.rr_signal:
            rr_signal = np.zeros((self.memory_ds.seg_sz)) 
            r_peaks = self.ecg_detector.pan_tompkins_detector(trainX)
            r_peak_diff = np.diff(r_peaks)
            # if len(r_peak_diff) > len(rr_signal):  # trim excess R peaks
            #     r_peak_diff = r_peak_diff[:len(rr_signal)]
            rr_signal[0:len(r_peak_diff)] = r_peak_diff
            print(rr_signal)
            rr_signal = zscore(rr_signal)  # normalise
            rr_signal = np.expand_dims(rr_signal, axis=1)
            scaler = preprocessing.RobustScaler()
            rr_signal = scaler.fit_transform(rr_signal)
            scaler = preprocessing.MinMaxScaler()
            rr_signal = scaler.fit_transform(rr_signal)
            trainX = np.vstack((trainX.flatten(), rr_signal.flatten())).T
        trainY = self.memory_ds.seg_labels[ID]
        if self.as_np:
            return trainX, trainY
        X_tensor = Variable(torch.from_numpy(trainX)).type(torch.FloatTensor)
        # print(f"X_tensor before: {X_tensor.size()}")
        r"numpy array shape: (n_samp, n_chan), reshape it to (n_chan, n-samp)."
        X_tensor = X_tensor.reshape(X_tensor.size()[1], -1)
        # print(f"X_tensor after: {X_tensor.size()}")
        # Y_tensor = torch.from_numpy(trainY).type(torch.FloatTensor)
        Y_tensor = trainY

        if torch.any(torch.isnan(X_tensor)):
            # self.memory_ds.log(f"[partial-ds] NaN found for idx:{idx}")
            X_tensor = torch.nan_to_num(X_tensor)

        return X_tensor, Y_tensor
    

class PartialHybridDataset(MddDataset):
    r"""Deliver partial view of the parent dataset."""

    def __init__(
        self, dataset=None, seg_index=None, test=False, shuffle=False, i_chan=-1,
        as_np=False
    ):
        r"""Construct partial dataset instance."""
        self.memory_ds = dataset
        self.indexes = seg_index[:]
        self.test = test
        self.shuffle = shuffle
        self.i_chan = i_chan
        self.as_np = as_np

    def on_epoch_end(self):
        r"""End of epoch."""
        if self.shuffle and not self.test:
            np.random.shuffle(self.indexes)

    def __len__(self):
        r"""Dataset length."""
        return len(self.indexes)

    def __getitem__(self, idx):
        r"""Find and return item."""
        ID = self.indexes[idx]
        rec_name, idx_seg = self.memory_ds.segments[ID]
        signal = self.memory_ds.records[rec_name]
        trainX = signal[idx_seg:idx_seg+self.memory_ds.seg_sz, :]

        if self.i_chan == 13:
            # combination of channel 1 & 3
            trainX = np.c_[trainX[:, 0], trainX[:, 2]]
        elif self.i_chan in [0, 1, 2]:
            trainX = trainX[:, self.i_chan]
            trainX = np.expand_dims(trainX, axis=1)
        trainY = self.memory_ds.seg_labels[ID]

        # print("signal:", trainX.shape)
        "x: Stack segment of 1 sec"
        whole_signal = trainX
        signal_stack = None
        block_sz = self.memory_ds.seg_sz // self.memory_ds.seg_sec
        for i in range(2):
            for i_block in range(self.memory_ds.seg_sec):
                start = i_block * block_sz 
                block = trainX[start:start+block_sz]
                if signal_stack is None:
                    signal_stack = np.array(block)
                else:
                    signal_stack = np.column_stack((signal_stack, np.array(block)))            
                # print(f"block-{i_block}: {signal_stack.shape}")
        trainX = signal_stack

        if self.as_np:
            return trainX, trainY

        X_tensor = Variable(torch.from_numpy(trainX)).type(torch.FloatTensor)
        # print(f"X_tensor before: {X_tensor.size()}")
        r"numpy array shape: (n_samp, n_chan), reshape it to (n_chan, n-samp)."
        X_tensor = X_tensor.reshape(X_tensor.size()[1], -1)
        # print(f"X_tensor after: {X_tensor.size()}")
        # Y_tensor = torch.from_numpy(trainY).type(torch.FloatTensor)
        Y_tensor = trainY

        if torch.any(torch.isnan(X_tensor)):
            # self.memory_ds.log(f"[partial-ds] NaN found for idx:{idx}")
            X_tensor = torch.nan_to_num(X_tensor)

        # X2_tensor = Variable(torch.from_numpy(whole_signal)).type(torch.FloatTensor)
        # X2_tensor = X2_tensor.reshape(X2_tensor.size()[1], -1)

        return X_tensor, Y_tensor
    

def test_rrDataset():
    # datasource = MddRrDataset(
    #     input_dir="data/mdd", hz=1000, seg_sz=30, n_subjects=1, np_data=False)
    datasource = MddRrDataset(
        input_dir="data/mdd/np_data", hz=100, seg_sz=30, seg_slide_sec=5,
        n_subjects=1, np_data=True)
    print(f"subjects:{datasource.record_names}")
    p_ds = PartialRRDataset(
        dataset=datasource, 
        seg_index=datasource.record_wise_segments[datasource.record_names[0]],
        as_np=True)
    for i in range(20):
        seg = p_ds[i][0]
        print(f"partial-ds, seg:{seg.shape}")
        plt.plot(range(seg.shape[0]), seg)
        plt.show()


def test_datasource():
    seg_sec = 30
    datasource = MddDataset(
        input_dir="data/mdd/np_data", seg_sec=seg_sec, hz=100,
        seg_slide_sec=10, max_sig_len=-1, n_subjects=1, np_data=True)
    print(f"subjects:{datasource.record_names}")
    p_ds = PartialCsvDataset(
        dataset=datasource, 
        seg_index=datasource.record_wise_segments[datasource.record_names[0]],
        as_np=True) 
    for i in range(5):
        seg = p_ds[i][0]
        print(f"partial-ds, seg:{seg.shape}")
        plt.plot(range(seg.shape[0]), seg)
        plt.show()

    # rr_signal = True
    # p_ds = PartialRawAndRRDataset(
    #         dataset=datasource, 
    #         seg_index=datasource.record_wise_segments[datasource.record_names[0]],
    #         i_chan=2, rr_signal=rr_signal, as_np=True) 
    # for i in range(5):
    #     seg = p_ds[i][0]
    #     print(f"partial-ds, seg:{seg.shape}")
    #     if rr_signal:
    #         plt.plot(range(seg.shape[0]), seg[:, 0])
    #         plt.show()
    #         plt.plot(range(seg_sec), seg[:seg_sec, 1])
    #         plt.show()
    #     else:
    #         plt.plot(range(seg.shape[0]), seg)
    #         plt.show()
        
    #     break


def test_csv_datasource():
    datasource = MDDCsvDataset(
        input_dir="data/mdd/np_data", n_seg_per_sub=10)
    print(f"subjects:{datasource.record_names}")
    p_ds = PartialCsvDataset(
        dataset=datasource,
        seg_index=datasource.record_wise_segments[datasource.record_names[0]],
        as_np=True, data_cube=True)
    for i in range(1):
        seg = p_ds[i][0]
        print(f"seg_type:{type(seg)}")
        print(f"partial-ds, seg:{seg.shape}")
        signal = seg[:, 0]
        for j in range(20):
            signal = seg[:, j]
            plt.plot(range(signal.shape[0]), signal)
            plt.show()  
        plt.plot(range(signal.shape[0]), signal)
        plt.show()       


def test_csv_derived_datasource():
    datasource = MDDCsvDataset(
        input_dir="data/mdd/np_data", seg_slide_sec=1)
    print(f"subjects:{datasource.record_names}")
    p_ds = PartialDerivedDataset(
        dataset=datasource,
        seg_index=datasource.record_wise_segments[datasource.record_names[0]],
        as_np=True)
    for i in range(2):
        seg = p_ds[i][0]
        print(f"partial-ds, seg:{seg.shape}")
        signal = seg[:, 0]
        plt.plot(range(signal.shape[0]), signal)
        plt.show()       
        signal = seg[:, 1]
        plt.plot(range(signal.shape[0]), signal)
        plt.show()       


def test_csv_rr_datasource():
    datasource = MddRRDataset(
        input_dir="data/mdd/np_data_500hz", hz=500, seg_sec=30, n_subjects=1)
    print(f"subjects:{datasource.record_names}")
    p_ds = PartialRRDataset(
        dataset=datasource,
        seg_index=datasource.record_wise_segments[datasource.record_names[0]],
        as_np=True)
    for i in range(10):
        seg = p_ds[i][0]
        print(f"seg_type:{type(seg)}")
        print(f"partial-ds, seg:{seg.shape}")
        signal = seg[:, 0]
        plt.plot(range(seg.shape[0]), signal)
        plt.show()               


def test_hybrid_datasource():
    datasource = MddDataset(
        input_dir="data/mdd", hz=300, seg_sec=10, 
        seg_slide_sec=10, max_sig_len=-1, n_subjects=1)
    print(f"subjects:{datasource.record_names}")
    p_ds = PartialHybridDataset(
        dataset=datasource, 
        seg_index=datasource.record_wise_segments[datasource.record_names[0]],
        i_chan=2,
        as_np=True) 
    for i in range(1):
        seg = p_ds[i][0]
        print(f"seg_type:{type(seg)}")
        print(f"partial-ds, seg:{seg.shape}")
        for j in range(20):
            signal = seg[:, j]
            plt.plot(range(signal.shape[0]), signal)
            plt.show()       
        # signal = seg[1][:, 0]
        # plt.plot(range(signal.shape[0]), signal) 
        # plt.show()


def main():
    test_csv_rr_datasource()



if __name__ == "__main__":
    main()