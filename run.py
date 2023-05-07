import warnings
warnings.filterwarnings('ignore') # setting ignore as a parameter

import os 
from datetime import datetime 
import logging
import argparse
import traceback

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

import torch
from torch import nn

import model_builder
import datasource
import model_training

logger = logging.getLogger(__name__)


def log(msg):
    logger.debug(msg)


def config_logger(log_file):
    r"""Config logger."""
    global logger
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(format)
    logger.addHandler(ch)
    # logging.basicConfig(format=format, level=logging.DEBUG, datefmt="%H:%M:%S")
    # logger = logging.getLogger(__name__)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setFormatter(format)
    fh.setLevel(logging.DEBUG)
    # add the handlers to the logger
    logger.addHandler(fh)


CUR_FOLD = -1
CONV0_OUT_CHAN = -1
TM_SIM_START = None
LOG_PATH = None


def debug_data(
    epoch, model=None, inputs=None, labels=None, preds=None, extra_out=None,
):
    global TM_SIM_START, LOG_PATH
    tm_sim_start = TM_SIM_START
    log_path = LOG_PATH
    debug_path = f"{log_path}/debug"
    if not os.path.exists(debug_path):
        os.makedirs(debug_path)

    if model.cfg.get("sinc_conv0"):
        r"Store band-pass filter optimisation."
        low_hz = model.low_conv_layers[1].low_hz_.data.detach().cpu().numpy()[:, 0]
        band_hz = model.low_conv_layers[1].band_hz_.data.detach().cpu().numpy()[:, 0]
        fband_line = f"Fold{CUR_FOLD} epoch:{epoch-1}, "
        for i_sinc_chan in range(CONV0_OUT_CHAN):
            fband_line += f"{low_hz[i_sinc_chan]:.02f}, {band_hz[i_sinc_chan]:.02f}, "
        r"Remove trailing ,"
        fband_line = fband_line[: fband_line.rfind(",")]
        with open(f"{debug_path}/{tm_sim_start}.fband", "a") as f:
            f.write(fband_line + "\n")


def debug_data_resnet(
    epoch,
    model=None,
    inputs=None,
    labels=None,
    preds=None,
    extra_out=None,
):
    global TM_SIM_START, LOG_PATH
    tm_sim_start = TM_SIM_START
    log_path = LOG_PATH
    debug_path = f"{log_path}/debug"
    if not os.path.exists(debug_path):
        os.makedirs(debug_path)

    if model.low_conv_options.get("sinc_kernel"):
        r"Store band-pass filter optimisation."
        low_hz = model.low_conv[0].low_hz_.data.detach().cpu().numpy()[:, 0]
        band_hz = model.low_conv[0].band_hz_.data.detach().cpu().numpy()[:, 0]
        fband_line = f"Fold{CUR_FOLD} epoch:{epoch-1}, "
        for i_sinc_chan in range(CONV0_OUT_CHAN):
            fband_line += f"{low_hz[i_sinc_chan]:.02f}, {band_hz[i_sinc_chan]:.02f}, "
        r"Remove trailing ,"
        fband_line = fband_line[: fband_line.rfind(",")]
        with open(f"{debug_path}/{tm_sim_start}.fband", "a") as f:
            f.write(fband_line + "\n")


def train_loo(
    data_path=None, base_path=None, log_path=None, model_path=None, class_w=None, 
    n_classes=None, hz=None, hz_down_factor=None, seg_sec=None, n_skip_seg=None, k_fold=5, 
    idx_chan=None, max_epoch=None, early_stop_patience=None, early_stop_delta=None, 
    lr_scheduler_patience=None, init_lr=None, w_decay=None, batch_sz=None, 
    n_subjects=None, seg_slide_sec=1, n_chan=1, max_signal_len=None, device=None,
    tm_sim_start=None, subjects_exclude=None, sinc_conv=None, data_cube=False
):   
    avg_global = []
    dataset = datasource.MDDCsvDataset(
        input_dir=data_path, seg_sec=seg_sec, seg_slide_sec=seg_slide_sec, hz=hz, log=log)
    for i_test_rec_name, test_rec_name in enumerate(dataset.record_names):
        if subjects_exclude is not None and test_rec_name in subjects_exclude.split(","):
                log(f"Exclude subject: {test_rec_name}")
                continue
        # validation set
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
        train_idx = []
        for train_rec_name in dataset.record_names:
            if train_rec_name in [test_rec_name]:
                continue
            train_idx.extend(dataset.record_wise_segments[train_rec_name])

        train_index, val_index = next(
            skf.split(
                np.zeros((len(train_idx), 1)),  # dummy train sample
                [dataset.seg_labels[i] for i in train_idx]
            )
        )
        # Translate skf indexes back in terms of train_idx
        train_index = [train_idx[i] for i in train_index]
        val_index = [train_idx[i] for i in val_index]
        # Form partial train/validation dataset
        train_dataset = datasource.PartialDerivedDataset(
            dataset, seg_index=train_index, shuffle=True, i_chan=idx_chan)
        val_dataset = datasource.PartialDerivedDataset(
            dataset, seg_index=val_index, i_chan=idx_chan)

        # net = model_builder.create_BranchRawAndRRConvResNet({
        #     'kernels_0': [
        #         [(21, 1, 1, 32), (21, 1, 1, 32), 'R', 'M', (11, 1, 1, 64), (11, 1, 1, 64), 'R', 'M', (7, 1, 1, 128), (7, 1, 1, 128), 'R', (5, 1, 1, 256), (5, 1, 1, 256), 'R'],
        #     ],
        #     'kernels_1': [
        #         [(21, 1, 1, 32), (21, 1, 1, 32), 'R', 'M', (11, 1, 1, 64), (11, 1, 1, 64), 'R', 'M', (7, 1, 1, 128), (7, 1, 1, 128), 'R', (5, 1, 1, 256), (5, 1, 1, 256), 'R'],
        #     ],
        #     'feat_per_branch': 128,
        #     'h': (128*2),
        #     'n_classes': n_classes,
        #     'dropout': 0.2,
        #     'hz': hz,
        # }, log=log)

        net = model_builder.create_BranchRawConvResNet({
            'kernels_0': [
                [(17, 1, 1, 32), (17, 1, 1, 32), 'R', 'M', (5, 1, 1, 64), (5, 1, 1, 64), 'R', 'M', (5, 1, 1, 128), (5, 1, 1, 128), 'R', (5, 1, 1, 256), (5, 1, 1, 256), 'R'],
            ],
            'kernels_1': [
                [(17, 1, 1, 32), (17, 1, 1, 32), 'R', 'M', (5, 1, 1, 64), (5, 1, 1, 64), 'R', 'M', (5, 1, 1, 128), (5, 1, 1, 128), 'R', (5, 1, 1, 256), (5, 1, 1, 256), 'R'],
            ],
            'feat_per_branch': 256,
            'h': (256*2),
            'n_classes': n_classes,
            'dropout': 0.2,
            'hz': hz,
        }, log=log)
            
        if i_test_rec_name == 0:
            log(net)

        train_dist = np.unique(
            [train_dataset.memory_ds.seg_labels[i]
                for i in train_dataset.indexes], return_counts=True
        )
        val_dist = np.unique(
            [val_dataset.memory_ds.seg_labels[i]
                for i in val_dataset.indexes], return_counts=True
        )

        logger.debug(
            f"Training starts. Test:{test_rec_name}, channel:{idx_chan}, "
            f"#params:{model_training.count_parameters(net)}, "
            f"train-dist:{train_dist}, "
            f"val-dist:{val_dist}, ")

        r"Training."
        model_file = f"{model_path}/{tm_sim_start}_chan{idx_chan}_test{test_rec_name}.pt"

        r"Criteria with weighted loss for imabalanced class."
        criterion = nn.CrossEntropyLoss(
            # weight=class_weights
        )
        test_ds = datasource.PartialDerivedDataset(
                dataset, 
                seg_index=dataset.record_wise_segments[test_rec_name], 
                test=True, i_chan=idx_chan)

        best_model, train_stat, val_scores = model_training.fit(
            net, train_dataset, val_dataset,
            test_dataset=test_ds,
            device=device,
            model_file=model_file,
            max_epoch=max_epoch,
            early_stop_patience=early_stop_patience,
            early_stop_delta=early_stop_delta,
            weight_decay=w_decay,
            lr_scheduler_patience=lr_scheduler_patience,
            batch_size=batch_sz,
            init_lr=init_lr,
            criterion=criterion,
            log=log,
            # epoch_wise_debug_fn=debug_data if sinc_conv==1 else None
            epoch_wise_debug_fn=debug_data_resnet if sinc_conv==1 else None
        )

        r"Plot training statistics."
        model_training.plot_training_stat(
            0, train_stat,
            log_image_path=f"{base_path}/training_stat",
            image_file_suffix=f"test{test_rec_name}")
        logger.debug(
            f"Test:{test_rec_name} finished.")

        preds = val_scores.get('preds')
        labels = val_scores.get('labels')

        r"Persist preds and labels."
        pred_path = f"{log_path}/preds/{tm_sim_start}"
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)
        pred_file = f"{pred_path}/test{test_rec_name}_preds.csv"
        df = pd.DataFrame({
            'preds': preds,
            'labels': labels
        })
        df.to_csv(pred_file, index=True)

        _prec, _recl, _f1, _acc, report_dict = model_training.score(
            labels, preds)
        confusion_mat = confusion_matrix(labels, preds)
        logger.info(
            f"@LOOV summary|rec:{test_rec_name}_chan{idx_chan}, "
            f"acc:{val_scores['acc']:.05f}, "
            f"prec:{val_scores['prec']:.05f}, "
            f"recl:{val_scores['recl']:.05f}, "
            f"f1:{val_scores['f1']:.05f}, "
            f"report: {report_dict}, all-fold-confusion_matrix:{confusion_mat}"
        )
        avg_global.append(val_scores['acc'])
    logger.info(
        f'@@EndOfSim, Global-avg:{np.average(avg_global)}, detail:{avg_global}')


def train_kfold(
    data_path=None, base_path=None, log_path=None, model_path=None, class_w=None, 
    n_classes=None, hz=None, hz_down_factor=None, seg_sec=None, n_skip_seg=None, k_fold=5, 
    idx_chan=None, max_epoch=None, early_stop_patience=None, early_stop_delta=None, 
    lr_scheduler_patience=None, init_lr=None, w_decay=None, batch_sz=None, 
    n_subjects=None, seg_slide_sec=1, n_chan=1, max_signal_len=None, device=None,
    tm_sim_start=None, subjects_exclude=None, sinc_conv=None, data_cube=False, conv_branching=None
):
    global TM_SIM_START, LOG_PATH
    TM_SIM_START = tm_sim_start
    LOG_PATH = log_path

    # dataset = datasource.MddDataset(
    #     input_dir=data_path, hz=hz, seg_sec=seg_sec, n_chan=n_chan, n_subjects=n_subjects,
    #     seg_slide_sec=seg_slide_sec, hz_down_factor=hz_down_factor, 
    #     max_sig_len=max_signal_len, log=log, np_data=True)
    
    dataset = datasource.MDDCsvDataset(
        input_dir=data_path, seg_sec=seg_sec, seg_slide_sec=seg_slide_sec, hz=hz, log=log)
    # dataset = datasource.MDDCsvDataset(
    #     input_dir=data_path, seg_sec=seg_sec, n_seg_per_sub=300, log=log)
    
    avg_global = []
    # Subject-wise stratified k-fold
    skf_subject = StratifiedKFold(
        n_splits=k_fold, shuffle=True, random_state=2021)
    skf_sub_split = skf_subject.split(
        np.zeros((len(dataset.record_names), 1)),  # dummy train sample
        [1 if f[0] == '0' else 0 for f in dataset.record_names]
    )
    for i_fold, (idx_kf_train, idx_kf_test) in enumerate(skf_sub_split):
        train_idx = []
        for train_rec_name in [dataset.record_names[i] for i in idx_kf_train]:
            # Exclude subjects, if required.
            # log(f"Train-rec-name:{train_rec_name} -> {subjects_exclude.split(',')}")
            if subjects_exclude is not None and train_rec_name in subjects_exclude.split(","):
                log(f"Exclude subject: {train_rec_name}")
                continue
            train_idx.extend(dataset.record_wise_segments[train_rec_name])

        r"Fold strategy: balanced labels in train/validation."
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
        train_index, val_index = next(
            skf.split(
                np.zeros((len(train_idx), 1)),  # dummy train sample
                [dataset.seg_labels[i] for i in train_idx]
            )
        )
        # Translate skf indexes back in terms of train_idx
        train_index = [train_idx[i] for i in train_index]
        val_index = [train_idx[i] for i in val_index]
        # Form partial train/validation dataset
        train_dataset = datasource.PartialDerivedDataset(
            dataset, seg_index=train_index, shuffle=True, i_chan=idx_chan)
        val_dataset = datasource.PartialDerivedDataset(
            dataset, seg_index=val_index, i_chan=idx_chan)

        # net = model_builder.create_base_PyramidNet(log=log)
        net = model_builder.create_BranchRawConvResNet({
            'kernels_0': [
                [(17, 1, 1, 32), (17, 1, 1, 32), 'R', 'M', (5, 1, 1, 64), (5, 1, 1, 64), 'R', 'M', (5, 1, 1, 128), (5, 1, 1, 128), 'R', (5, 1, 1, 256), (5, 1, 1, 256), 'R'],
            ],
            'kernels_1': [
                [(17, 1, 1, 32), (17, 1, 1, 32), 'R', 'M', (5, 1, 1, 64), (5, 1, 1, 64), 'R', 'M', (5, 1, 1, 128), (5, 1, 1, 128), 'R', (5, 1, 1, 256), (5, 1, 1, 256), 'R'],
            ] if conv_branching==1 else None,
            'feat_per_branch':  128 if conv_branching==1 else 256,
            'h': (256*(2 if conv_branching==1 else 1)),
            'n_classes': n_classes,
            'dropout': 0.2,
            'hz': hz,
        }, log=log)
        # if sinc_conv == 0:
        # net = model_builder.create_BranchRawAndRRConvResNet({
        #     'kernels_0': [
        #         [(21, 1, 1, 32), (21, 1, 1, 32), 'R', 'M', (11, 1, 1, 64), (11, 1, 1, 64), 'R', 'M', (7, 1, 1, 128), (7, 1, 1, 128), 'R', (5, 1, 1, 256), (5, 1, 1, 256), 'R'],
        #     ],
        #     'kernels_1': [
        #         [(21, 1, 1, 32), (21, 1, 1, 32), 'R', 'M', (11, 1, 1, 64), (11, 1, 1, 64), 'R', 'M', (7, 1, 1, 128), (7, 1, 1, 128), 'R', (5, 1, 1, 256), (5, 1, 1, 256), 'R'],
        #     ],
        #     'feat_per_branch': 128,
        #     'h': (128*2),
        #     'n_classes': n_classes,
        #     'dropout': 0.2,
        #     'hz': hz,
        # }, log=log)
        # else:
            # net = model_builder.create_model_few_kernel(
            #     conv0_out=8, low_hz=0.1, high_hz=hz, n_classes=2, seg_sz=hz*seg_sec,
            #     hz=hz, log=log, gru_module=True, sinc_conv=True,
            # )
            # net = model_builder.create_sinc_shallow_18240(
            #     hz=hz,
            #     seg_sz=hz * seg_sec,
            #     gru_module=False, log=log)
            # net = model_builder.create_sincResNet(
            #         conv0_out=16, low_hz=0.5, high_hz=50, seg_sz=hz*seg_sec, hz=hz,
            #         log=log, n_conv_per_blk=2, sinc_conv=True)
            
        if i_fold == 0:
            global CONV0_OUT_CHAN
            CONV0_OUT_CHAN = net.cfg["sinc_conv0"] if sinc_conv==1 else -1
            log(net)


        train_dist = np.unique(
            [train_dataset.memory_ds.seg_labels[i]
                for i in train_dataset.indexes], return_counts=True
        )
        val_dist = np.unique(
            [val_dataset.memory_ds.seg_labels[i]
                for i in val_dataset.indexes], return_counts=True
        )

        logger.debug(
            f"Training starts. Fold:{i_fold}, channel:{idx_chan}, "
            f"#params:{model_training.count_parameters(net)}, "
            f"train-dist:{train_dist}, "
            f"val-dist:{val_dist}, ")

        r"Training."
        model_file = f"{model_path}/{tm_sim_start}_chan{idx_chan}_fold{i_fold}.pt"

        r"Criteria with weighted loss for imabalanced class."
        criterion = nn.CrossEntropyLoss(
            # weight=class_weights
        )

        # Form a test database with only a single subject just for early tracking
        # test_ds = datasource.PartialHybridDataset(
        #         dataset, 
        #         seg_index=dataset.record_wise_segments[dataset.record_names[idx_kf_test[0]]], 
        #         test=True, i_chan=idx_chan)
        # test_ds = datasource.PartialCsvDataset(
        #         dataset, 
        #         seg_index=dataset.record_wise_segments[dataset.record_names[idx_kf_test[0]]], 
        #         test=True, data_cube=data_cube)
        test_ds = datasource.PartialDerivedDataset(
                dataset, 
                seg_index=dataset.record_wise_segments[dataset.record_names[idx_kf_test[0]]], 
                test=True, i_chan=idx_chan)

        global CUR_FOLD
        CUR_FOLD = i_fold
        best_model, train_stat, _ = model_training.fit(
            net, train_dataset, val_dataset,
            test_dataset=test_ds,
            device=device,
            model_file=model_file,
            max_epoch=max_epoch,
            early_stop_patience=early_stop_patience,
            early_stop_delta=early_stop_delta,
            weight_decay=w_decay,
            lr_scheduler_patience=lr_scheduler_patience,
            batch_size=batch_sz,
            init_lr=init_lr,
            criterion=criterion,
            log=log,
            epoch_wise_debug_fn=debug_data if sinc_conv==1 else None
            # epoch_wise_debug_fn=debug_data_resnet if sinc_conv==1 else None
        )

        r"Plot training statistics."
        model_training.plot_training_stat(
            0, train_stat,
            log_image_path=f"{base_path}/training_stat",
            image_file_suffix=f"{tm_sim_start}_fold{i_fold}")
        logger.debug(
            f"Train-fold:{i_fold} finished. Now testing ... ")

        r"Test all test subjects."
        for test_rec_name in [dataset.record_names[i] for i in idx_kf_test]:
            idx_test_samples = []
            idx_test_samples.extend(
                dataset.record_wise_segments[test_rec_name]
            )
            test_dataset = datasource.PartialDerivedDataset(
                dataset, seg_index=idx_test_samples, test=True, i_chan=idx_chan)
            # test_dataset = datasource.PartialCsvDataset(
            #     dataset, seg_index=idx_test_samples, test=True, data_cube=data_cube)
            logger.debug(
                f"Test-fold:{i_fold}, "
                f"test-subject:{test_rec_name}, samples:{len(test_dataset)}")

            val_scores = model_training.validate(
                net, test_dataset, device=device, evaluate=True
            )

            preds = val_scores.get('preds')
            labels = val_scores.get('labels')

            r"Persist preds and labels."
            pred_path = f"{log_path}/preds/{tm_sim_start}"
            if not os.path.exists(pred_path):
                os.makedirs(pred_path)
            pred_file = (
                f"{pred_path}/test{test_rec_name}_chan{idx_chan}_"
                f"fold{i_fold}_.preds.csv"
                # f"val{val_rec_name}.preds.csv"
            )
            df = pd.DataFrame({
                'preds': preds,
                'labels': labels
            })
            df.to_csv(pred_file, index=True)

            _prec, _recl, _f1, _acc, report_dict = model_training.score(
                labels, preds)
            confusion_mat = confusion_matrix(labels, preds)
            logger.info(
                f"@KFold summary|rec:{test_rec_name}_chan{idx_chan}, "
                f"acc:{val_scores['acc']:.05f}, "
                f"prec:{val_scores['prec']:.05f}, "
                f"recl:{val_scores['recl']:.05f}, "
                f"f1:{val_scores['f1']:.05f}, "
                f"report: {report_dict}, all-fold-confusion_matrix:{confusion_mat}"
            )
            avg_global.append(val_scores['acc'])
    logger.info(
        f'@@EndOfSim, Global-avg:{np.average(avg_global)}, detail:{avg_global}')



def load_config():
    parser = argparse.ArgumentParser(description="MDD detection")
    parser.add_argument("--i_cuda", default=0, help="CUDA")
    parser.add_argument("--class_w", default=True, help="Weighted class")
    parser.add_argument("--n_classes", default=2, help="MDD vs control")
    parser.add_argument("--hz", default=100, help="Target Hz")
    parser.add_argument("--hz_down_factor", default=None, help="Factor to divide the source Hz 1000Hz")
    parser.add_argument("--seg_sec", default=20, help="Segment len in sec")
    parser.add_argument("--seg_slide_sec", default=1, help="Segment sliding in sec")
    parser.add_argument("--n_skip_seg", default=2,
                        help="Skip initial N segments")
    parser.add_argument("--max_signal_len", default=-1, help="Max signal length in sec")
    parser.add_argument("--idx_chan", default=2, help="Signal channel number")
    parser.add_argument("--max_epoch", default=200, help="Max no. of epoch")
    parser.add_argument("--early_stop_patience", default=15,
                        help="Early stop patience")
    parser.add_argument("--early_stop_delta", default=0.001,
                        help="Early stop delta")
    parser.add_argument("--lr_scheduler_patience",
                        default=5, help="LR scheduler patience")
    parser.add_argument("--init_lr", default=0.001, help="Initial LR")
    parser.add_argument("--w_decay", default=0, help="LR weight decay")
    parser.add_argument("--base_path", default=None, help="Sim base path")
    parser.add_argument("--data_path", default=None, help="Data dir")
    parser.add_argument("--batch_sz", default=64, help="Batch size")
    parser.add_argument("--k_fold", default=10, help="K-fold subject-wise")
    parser.add_argument("--n_subjects", default=-1, help="max no. of subjects")
    parser.add_argument("--subjects_exclude", default="039,113", help="max no. of subjects")
    # parser.add_argument("--subjects_exclude", default="024,111,140,113", help="max no. of subjects")
    # parser.add_argument("--subjects_exclude", default=None, help="max no. of subjects")
    parser.add_argument("--sinc_conv", default=0, help="Sinc convolution")
    parser.add_argument("--conv_branching", default=1, help="Enable convolution branching")
    parser.add_argument("--alias", help="Simulation alias name")

    args = parser.parse_args()

    args.tm_sim_start = f"{args.alias}_{datetime.now():%Y%m%d%H%M%S}"
    if args.base_path is None:
        args.base_path = os.getcwd()
    args.log_path = f"{args.base_path}/logs"
    args.model_path = f"{args.base_path}/models"
    if args.data_path is None:
        # args.data_path = f"{args.base_path}/data/mdd"
        args.data_path = f"{args.base_path}/data/mdd/np_data_{args.hz}hz"

    # Convert commonly used parameters to integer, if required.
    if isinstance(args.hz, str):
        args.hz = int(args.hz)
    if isinstance(args.i_cuda, str):
        args.i_cuda = int(args.i_cuda)
    if isinstance(args.n_classes, str):
        args.n_classes = int(args.n_classes)
    if isinstance(args.batch_sz, str):
        args.batch_sz = int(args.batch_sz)
    if isinstance(args.max_epoch, str):
        args.max_epoch = int(args.max_epoch)
    if isinstance(args.n_subjects, str):
        args.n_subjects = int(args.n_subjects)
    if isinstance(args.idx_chan, str):
        args.idx_chan = int(args.idx_chan)
    if isinstance(args.k_fold, str):
        args.k_fold = int(args.k_fold)
    if isinstance(args.max_signal_len, str):
        args.max_signal_len = int(args.max_signal_len)
    if isinstance(args.seg_sec, str):
        args.seg_sec = int(args.seg_sec)
    if isinstance(args.seg_slide_sec, str):
        args.seg_slide_sec = int(args.seg_slide_sec)
    if isinstance(args.sinc_conv, str):
        args.sinc_conv = int(args.sinc_conv)
    if isinstance(args.conv_branching, str):
        args.conv_branching = int(args.conv_branching)

    if args.max_signal_len > -1:
        args.max_signal_len = int(args.max_signal_len)*int(args.hz)    

    # GPU device?
    if args.i_cuda > 0:
        args.device = torch.device(
            f"cuda:{args.i_cuda}" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "cpu":
            args.device = torch.device(
                f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
            )

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    return args


def main():
    model_training.fix_randomness()
    args = load_config()
    config_logger(f"{args.log_path}/{args.tm_sim_start}.log")
    param_dict = vars(args)
    log(param_dict)
    # Exclude non-existing arguments
    #
    param_dict.pop("i_cuda", None)
    param_dict.pop("alias", None)
    try:
        train_kfold(**param_dict)
        # train_loo(**param_dict)
    except Exception as e:
        log(f"Exception in kfold, {str(e)}, caused by - \n{traceback.format_exc()}")
        logger.exception(e)


if __name__ == '__main__':
    main()

"""
TODO:
- downsample signal from 1000Hz to 300Hz instead of 100Hz.
- normalise whole segment instead of sec wise.
- Model: 
    -- layer-1: sinc-conv, ReLU
    -- layer-2: depth-separable conv, ReLU
    -- Max pool
    -- (Flatten) Linear, ReLU
    -- Linear
- Training: consider model weight initialisation
"""