import os
import time
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.autograd import Variable

# from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def fix_randomness():
    r"Fix randomness."
    RAND_SEED = 2021
    random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)
    np.random.seed(RAND_SEED)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a
    given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        log=None,
        extra_meta=None,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss
                improved.
            verbose (bool): If True, prints a message for each validation loss
                improvement.
            delta (float): Minimum change in the monitored quantity to qualify
                as an improvement.
            path (str): Path for the checkpoint to be saved to.
            log : log function (TAG, msg).
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_extra = None  # Extra best other scores/info
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.log = log
        self.first_iter = False
        r"extra_meta is {'metric_name':'test_acc', 'max_val':99}"
        self.extra_meta = extra_meta

        self.print_(
            f"[{self.__class__.__name__}] patience:{patience}, delta:{delta}, model-path:{path}"
        )

    def print_(self, msg):
        if self.log is None:
            print(msg)
        else:
            self.log(f"[EarlyStopping] {msg}")

    def __call__(self, val_loss, model, extra=None):
        r"""extra is {'test_acc':90}. The key was passed at c'tor."""
        score = -val_loss

        if self.best_score is None:
            self.first_iter = True
            self.best_score = score
            self.best_extra = extra
            self.save_checkpoint(val_loss, model)
            self.first_iter = False
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.print_(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_extra = extra
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.print_(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

        r"If extra is passed in call, and extra_meta exists, terminate \
        training if condition is met."
        if (
            not self.first_iter
            and self.extra_meta is not None
            and self.best_extra is not None
            and self.extra_meta.get("metric_name") is not None
            and self.extra_meta.get("max_val") is not None
            and self.best_extra.get(self.extra_meta.get("metric_name")) is not None
            and self.best_extra.get(self.extra_meta.get("metric_name"))
            >= self.extra_meta.get("max_val")
        ):
            self.print_(
                f"{self.extra_meta.get('metric_name')}:"
                f"{self.best_extra.get(self.extra_meta.get('metric_name'))} "
                f">= {self.extra_meta.get('max_val')}"
            )
            self.early_stop = True


def count_parameters(model):
    r"""Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_stat(i_fold, stats=None, log_image_path=None, image_file_suffix=None):
    r"""Plot training statistics."""
    # image_path = LOG_FILE.replace('.log', f"_train_stat")
    # log_image_path = (
    #     f"{LOG_PATH}/training_stat/{image_path}")
    if not os.path.exists(log_image_path):
        os.makedirs(log_image_path)
    image_file = f"{log_image_path}/fold{i_fold}_{image_file_suffix}"
    plt.figure(figsize=(20, 7))
    _, axs = plt.subplots(1, 2)
    axs[0].plot(stats.get("train_loss"), color="orange", label="train loss")
    axs[0].plot(stats.get("val_loss"), color="red", label="validation loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[1].plot(stats.get("train_acc"), color="green", label="train accuracy")
    axs[1].plot(stats.get("val_acc"), color="blue", label="validation accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    # plt.set_legend()
    plt.savefig(f"{image_file}.png")

    df = pd.DataFrame.from_dict(
        {
            "train_loss": stats.get("train_loss"),
            "val_loss": stats.get("val_loss"),
            "train_acc": stats.get("train_acc"),
            "val_acc": stats.get("val_acc"),
            "lr": stats.get("lr"),
        },
        orient="index",
    ).transpose()
    df.to_csv(f"{image_file}.csv", index=True)


def get_class_weights(labels, n_class=2, log=print):
    r"""Calculate class frequency to calculate class weights.

    In order to calculate loss for imbalanced class.

    Arguments:
        partial_dataset. EcgDataset is extracted to get labels.

    Returns:
        frequency of labels.
        weights of frequency, max_freq/freq.
    """
    freq = np.zeros(n_class)
    for label in labels:
        freq[label] += 1

    # calculate weights
    max_freq = freq[np.argmax(freq)]
    weights = max_freq // freq
    log(f"freq:{freq}, weights:{weights}")
    return freq, weights


def score(labels, preds):
    r"""Calculate scores."""
    _preds = preds
    _labels = labels
    # _preds = _preds.argmax(axis=1)
    score_prec = precision_score(_labels, _preds, average="macro")
    score_recall = recall_score(_labels, _preds, average="macro")
    score_f1 = f1_score(_labels, _preds, average="macro")
    score_acc = accuracy_score(_labels, _preds)
    report_dict = classification_report(_labels, _preds, output_dict=True)
    return score_prec, score_recall, score_f1, score_acc, report_dict



def fit(
    model,
    train_dataset,
    val_dataset,
    test_dataset=None,
    max_epoch=100,
    device=None,
    model_file="model_",
    early_stop_patience=7,
    early_stop_delta=0,
    lr_scheduler_patience=5,
    batch_size=32,
    val_batch_size=None,
    opt_loss=True,
    init_lr=0.001,
    min_lr=1e-6,
    max_grad=5,
    weight_decay=0,
    criterion=None,
    mp_criteria=None,  # Multi purpose network loss criteria
    optimizer=None,
    lr_scheduler=None,
    log=print,
    epoch_wise_debug_fn=None,
):
    r"""Train the created model against the validation dataset.

    Early stops, if required.
    Arguments:
    - opt_loss: optimise loss, if True, otherwise acc.
    - debug_filename: .log file with absolute path (LOG_FILE).
    """
    training_stat = {
        "lr": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_acc": [],
    }
    avg_train_losses = []
    avg_valid_losses = []

    data_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    early_stopping = EarlyStopping(
        patience=early_stop_patience,
        path=model_file,
        delta=early_stop_delta,
        log=log,
        verbose=True,
        # extra_meta={"metric_name": "test_acc", "max_val": 0.999},
    )

    # # Gradient clipping. (https://spell.ml/blog/pytorch-training-tricks-YAnJqBEAACkARhgD)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)

    model.to(device)

    if optimizer is None:
        r"Default optimiser."
        optimizer = torch.optim.Adam(
            model.parameters(), lr=init_lr, weight_decay=weight_decay
        )
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=init_lr, momentum=0.9, nesterov=True,
    #     weight_decay=weight_decay)

    if lr_scheduler is None:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min" if opt_loss else "max",
            min_lr=min_lr,
            factor=0.5,
            patience=lr_scheduler_patience,
            verbose=True,
        )
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, max_lr=0.01, steps_per_epoch=len(data_loader), epochs=MAX_EPOCH)

    debug_epoch = 0

    log(
        f"[model_training:fit] model:{model.__class__.__name__}, "
        f"#params:{count_parameters(model)}, "
        f"train-db:{len(train_dataset)}, val-db:{len(val_dataset)}, "
        f"max-epoch:{max_epoch}, device:{device}, model_file:{model_file}, "
        f"early_stop_pt/delta:{early_stop_patience}/{early_stop_delta}, "
        f"lr_schd_pt:{lr_scheduler_patience}, batch-sz:{batch_size}, "
        f"opt_loss:{opt_loss}, init_lr:{init_lr}, min_lr:{min_lr}, "
        f"max_grad:{max_grad}, criterion:{criterion}, optimizer:{optimizer}, "
        f"lr_scheduler:{lr_scheduler},"
    )

    for epoch in range(1, max_epoch + 1):
        train_losses = []
        since = time.time()
        # enable training mode
        model.train()
        data_loader.dataset.on_epoch_end()
        i_batch = 0
        total_train_samp = 0
        train_running_correct = 0
        extra_out = None
        nan_loss_found = False
        # train_running_loss = 0
        for inputs, labels in data_loader:
            i_batch += 1

            # push data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # log(f"LABELS:{labels.size()}, size(0):{labels.size(0)}")
            total_train_samp += labels.size(0)

            # clear gradients for next train
            optimizer.zero_grad()
            # feed-forward the model and back propagate accordingly.
            with torch.set_grad_enabled(True):
                # if mp_criteria, the fn handles model call and loss calculation
                loss = 0.0
                if mp_criteria is not None:
                    loss, preds = mp_criteria(model, inputs, labels, epoch=epoch)
                else:
                    r"Model should return prediction and extra output (or None)."
                    preds = model(inputs)
                    r"Model's first of multi-output is preds."
                    if isinstance(preds, tuple):
                        preds, extra_out = preds
                    loss = criterion(preds, labels)
                
                loss.backward()  # backpropagation, compute gradients

                # Gradient clipping. (https://spell.ml/blog/pytorch-training-tricks-YAnJqBEAACkARhgD)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)

                optimizer.step()  # apply gradients

            # calculate loss
            loss_item = loss.detach().item()

            train_losses.append(loss_item)
            _, train_preds = torch.max(preds.data, 1)
            train_running_correct += (train_preds == labels).sum().item()

            r"Call epoch-wise debug functioin to see learning process."
            if epoch_wise_debug_fn is not None and epoch != debug_epoch:
                debug_epoch = epoch
                epoch_wise_debug_fn(
                    epoch=epoch,
                    model=model,
                    inputs=inputs.detach().cpu().numpy(),
                    labels=labels.detach().cpu().numpy(),
                    preds=preds.detach().cpu().numpy(),
                    extra_out=extra_out,
                )

        pass  # for
        # train_loss = train_running_loss / i_batch
        train_acc = train_running_correct / total_train_samp
        # Validation
        val_scores = validate(
            model,
            val_dataset=val_dataset,
            criterion=criterion,
            device=device,
            batch_size=val_batch_size if val_batch_size else batch_size,
            # evaluate=True  # Subject-wise validation.
            mp_criteria=mp_criteria,
        )
        val_loss = val_scores.get("loss")
        val_acc = val_scores.get("acc").item()
        # train_loss = np.average(train_losses)
        train_loss = np.nanmean(train_losses)

        r"Adjust learning rate"
        if opt_loss:
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step(val_acc)

        time_elapsed = time.time() - since

        r"Report test performance as well."
        if test_dataset:
            test_scores = validate(
                model, 
                test_dataset, 
                criterion=criterion, 
                device=device, 
                evaluate=True,
                mp_criteria=mp_criteria,
            )
            test_acc = test_scores.get("acc").item()

        # time_elapsed = time.time() - since
        log(
            f"[Train] Epoch:{epoch}, init_lr:{init_lr}, "
            f"cur_lr:{optimizer.param_groups[0]['lr']}, "
            f"train_loss:{train_loss:.4f}, "
            f"{'Train_Loss_nan, ' if nan_loss_found else ''}"
            f"train_acc:{train_acc:.4f}, "
            f"val-loss:{val_loss:.4f}, "
            f"val-acc:{val_acc:.4f}, "
            f'test-acc:{test_scores.get("acc") if test_dataset else 0:.4f}, '
            f"time {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )

        avg_train_losses.append(train_loss)
        avg_valid_losses.append(val_loss)
        train_losses = []

        r"Record lr."
        training_stat.get("lr").append(optimizer.param_groups[0]["lr"])
        training_stat.get("train_loss").append(train_loss.item())
        training_stat.get("val_loss").append(val_loss.item())
        training_stat.get("train_acc").append(train_acc)
        if test_dataset:
            training_stat.get("test_acc").append(test_acc)
        training_stat.get("val_acc").append(val_acc)

        "Early stopping"
        if opt_loss:
            early_stopping(
                val_loss, model, extra={"test_acc": test_acc} if test_dataset else None
            )
        else:
            early_stopping(
                -val_acc, model, extra={"test_acc": test_acc} if test_dataset else None
            )

        if early_stopping.early_stop:
            log(f"[Train] Early stopping at epoch:{epoch}")
            break
    pass  # for

    # load best model weights
    model.load_state_dict(torch.load(model_file))
    log("Training is done.")

    # Validation with best model
    # val_loss, val_acc = validate_model(
    best_test_scores = None
    if test_dataset:
        best_test_scores = validate(
            model, 
            test_dataset, 
            criterion=criterion, 
            device=device, 
            evaluate=True,
            mp_criteria=mp_criteria,
        )
        log(f'[Train] Validate with best model, val-acc:{best_test_scores.get("acc"):.2f}')

    return model, training_stat, best_test_scores


def validate(
    model,
    val_dataset,
    criterion=None,
    device="cpu",
    batch_size=4,
    evaluate=False,
    mp_criteria=None
):
    r"""Validate a model."""
    if evaluate:
        batch_size = 1
    test_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    losses = []
    _preds = []
    _labels = []
    # Set the model to evaluation mode.
    loss = None
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            # push data to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            if mp_criteria is not None:
                loss, preds = mp_criteria(model, inputs, labels)
            else:
                preds = model(inputs)
                r"Model's first of multi-output is preds."
                if isinstance(preds, tuple):
                    preds, extra_out = preds
                if criterion is not None:
                    loss = criterion(preds, labels)
            
            if loss is not None:
                loss_item = loss.detach().item()
                losses.append(loss_item)

            r"Accuracy"
            predicted = torch.nn.functional.softmax(preds, dim=1)
            _preds.extend(predicted.detach().cpu().numpy().argmax(axis=1))
            r"CrossEntropyLoss specific."
            _labels.extend(labels.detach().cpu().numpy())
        pass  # data loader
    pass  # with grad
    try:
        _prec, _recl, _f1, _acc, report_dict = 0, 0, 0, 0, {}
        _prec, _recl, _f1, _acc, report_dict = score(_labels, _preds)
        assert len(_labels) == len(_preds)
    except:
        print("Error in score, skip calculation")

    return {
        "prec": _prec,
        "recl": _recl,
        "f1": _f1,
        "acc": _acc,
        "report": report_dict,
        # "loss": np.nanmean(losses) if criterion is not None else None,
        "loss": np.nanmean(losses) if loss is not None else None,
        "preds": _preds,
        "labels": _labels,
    }


def predict_segment(model=None, segment=None, device="cpu", hidden_init=False):
    r"""Predict segment using specified model."""
    model.to(device)
    model.eval()
    if hidden_init:
        h = model.init_hidden(batch_size=1, device=device).data
    with torch.no_grad():
        X_tensor = Variable(torch.from_numpy(segment)).type(torch.FloatTensor)
        X_tensor = X_tensor.view(1, 1, X_tensor.size()[0])
        X_tensor = X_tensor.to(device)
        if hidden_init:
            preds, h = model(X_tensor, h)
        else:
            preds = model(X_tensor)
        predicted = torch.nn.functional.softmax(preds, dim=1)
        return predicted.detach().cpu().numpy().argmax(axis=1).flatten().T


def predict(model, val_dataset, device="cpu", collect_labels=False):
    r"""Validate a model."""
    batch_size = 1
    test_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    # losses = []
    _preds = []
    _labels = []

    # Set the model to evaluation mode.
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            # push data to device
            inputs = inputs.to(device)
            # labels = labels.to(device)
            preds = model(inputs)
            r"Model's first of multi-output is preds."
            if isinstance(preds, tuple):
                preds, extra_out = preds
            predicted = torch.nn.functional.softmax(preds, dim=1)
            _preds.extend(predicted.detach().cpu().numpy().argmax(axis=1))
            if collect_labels:
                _labels.extend(labels.detach().cpu().numpy())
    return _preds, _labels
