import os
import sys
import json
import torch
import random
import logging
import numpy as np


import copy
import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    label_ranking_average_precision_score,
    ndcg_score,
    roc_auc_score,
)


def pad_graph(G, max_nodes):
    while len(G.nodes) < max_nodes:
        G.add_node(len(G.nodes), m_idx=0)
    return G


def get_ori_task_types(merged_list):
    with open("data/testdev_balanced_questions.json", "r") as file:
        gqa = json.load(file)
    cnt = 0
    for _, v in gqa.items():
        # early exist
        if cnt > 10100:
            break
        cnt += 1
        imageId, question, types = v["imageId"], v["question"], v["types"]
        for item in merged_list:
            if item["imageId"] == imageId and item["question"] == question:
                item["types"] = types
    return merged_list


def print_and_record(output_file, content):
    print(content)
    with open(output_file, "a") as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print(content)
        sys.stdout = original_stdout


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    # Check if the file already exists
    if os.path.exists(filename):
        # If the file exists, append a number to the filename to avoid overwriting
        index = 1
        prefix, postfix = filename.split(".")
        while os.path.exists(f"{prefix}_{index}.{postfix}"):
            index += 1
        filename = f"{prefix}_{index}.{postfix}"

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# for metagl

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] [%(module)s.%(funcName)s] %(message)s"
)

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)
logger.addHandler(sh)


def setup_cuda(args):
    args.cuda = args.gpu >= 0 and torch.cuda.is_available()
    args.device = torch.device("cuda:{}".format(args.gpu) if args.cuda else "cpu")
    if args.cuda:
        torch.cuda.set_device(args.device)
    return args


# noinspection PyUnresolvedReferences
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def as_torch_tensor(X):
    if isinstance(X, torch.Tensor):
        return X
    elif isinstance(X, np.ndarray):
        return torch.from_numpy(X).float()
    else:
        raise TypeError(f"Invalid type: {type(X)}")


def create_eval_dict(metrics=None):
    if metrics is None:
        metrics = ["AUC", "AUC@1", "MRR", "MRR@1", "HR@1", "nDCG@1"]
    assert len(metrics) > 0
    return {metric: [] for metric in metrics}


def hit_rate_at_k(y_true, y_score, k=10):
    y_true_flat = np.array(y_true).flatten()
    idx_true = np.argsort(y_true_flat)[::-1]

    y_score_flat = np.array(y_score).flatten()
    idx_pred_score = np.argsort(y_score_flat)[::-1]
    # noinspection PyUnresolvedReferences
    return np.intersect1d(idx_pred_score[0:k], idx_true[0:k]).shape[0] / (1.0 * k)


def eval_metrics_single_graph(y_true, y_pred):
    assert len(y_true.shape) == 2 and y_true.shape[0] == 1, y_true.shape
    assert y_pred.shape == y_true.shape, y_pred.shape

    idx_best_model = np.argmax(y_true)
    num_models = y_true.shape[1]
    y_true_bin = np.matrix(np.zeros((1, num_models), dtype=int))
    y_true_bin[0, idx_best_model] = 1

    eval_dict = {
        "nDCG@1": ndcg_score(y_true, y_pred, k=1),
    }

    y_true_flatten = np.array(y_true).flatten()
    for k in list(filter(lambda x: x <= len(y_true_flatten), [1])):
        top_k_ind = np.argpartition(y_true_flatten, -k)[-k:]
        num_models = y_true.shape[1]
        y_true_bin = np.matrix(np.zeros((1, num_models), dtype=int))
        y_true_bin[0, top_k_ind] = 1

        eval_dict[f"AUC@{k}"] = roc_auc_score(
            np.array(y_true_bin).flatten(), np.array(y_pred).flatten()
        )
        # eval_dict[f'MAP@{k}'] = average_precision_score(np.array(y_true_bin).flatten(), np.array(y_pred).flatten())
        eval_dict[f"MRR@{k}"] = label_ranking_average_precision_score(
            y_true_bin, y_pred
        )

    return eval_dict


def binarize_perf(Y):
    """For each row, set the maximum element to 1, and all others to 0"""
    Y = np.asarray(Y)
    Y_bin = np.zeros_like(Y, dtype=int)
    Y_bin[np.arange(len(Y)), Y.argmax(1)] = 1
    return Y_bin


def eval_metrics(Y_true, Y_pred, Y_true_bin=None):
    assert len(Y_true.shape) == 2 and Y_true.shape == Y_pred.shape, (
        Y_true.shape,
        Y_pred.shape,
    )
    if isinstance(Y_pred, torch.Tensor):
        Y_pred = Y_pred.cpu().detach().numpy()
    if isinstance(Y_true, torch.Tensor):
        Y_true = Y_true.cpu().detach().numpy()
    Y_pred, Y_true = np.asarray(Y_pred), np.asarray(Y_true)

    if Y_true_bin is None:
        Y_true_bin = binarize_perf(Y_true)

    eval_dict = {}
    with mp.Pool(processes=None) as pool:
        binary_args = []
        for y_true_bin, y_pred in zip(Y_true_bin, Y_pred):
            binary_args.append(
                (np.array(y_true_bin).flatten(), np.array(y_pred).flatten())
            )

        eval_dict["AUC"] = np.mean(pool.starmap(roc_auc_score, binary_args))
        eval_dict["MRR"] = np.mean(pool.starmap(average_precision_score, binary_args))

    eval_dict["nDCG@1"] = ndcg_score(Y_true, Y_pred, k=1)

    return eval_dict


def report_performance(method_names, method_eval, dec_place=3):
    if not isinstance(method_names, list):
        method_names = [method_names]
    if not isinstance(method_eval, list):
        method_eval = [method_eval]
    assert len(method_names) == len(method_eval), (len(method_names), len(method_eval))
    metric_names = ["AUC@1", "MRR@1", "nDCG@1"]

    perf_avg = np.zeros((len(method_eval), len(metric_names)), dtype=float)
    perf_std = np.zeros((len(method_eval), len(metric_names)), dtype=float)

    for meth_idx, method in enumerate(method_eval):
        for midx, metric in enumerate(metric_names):
            perf_avg[meth_idx, midx] = np.round(np.mean(method[metric]), dec_place)
            perf_std[meth_idx, midx] = np.round(np.std(method[metric]), dec_place)

    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)

    print("\nPerf. Avg:")
    print(pd.DataFrame(perf_avg, index=method_names, columns=metric_names))

    print("\nPerf. Std:")
    print(pd.DataFrame(perf_std, index=method_names, columns=metric_names))


class EarlyStopping:
    def __init__(
        self, patience=30, minimizing_objective=False, logging=True, score_type="score"
    ):
        self.patience = patience
        self.minimizing_objective = minimizing_objective
        self.counter = 0
        self.early_stop = False
        self.logging = logging
        self.best_score = None
        self.best_model_state_dict = None
        self.score_type = score_type

    def step(self, score, model=None):
        if self.best_score is None or self.improved(score, self.best_score):
            self.best_score = score
            if model is not None:
                self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.logging:
                logger.info(
                    f"[EarlyStopping-{self.score_type}] counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop  # Return whether to early stop

    def improved(self, score, best_score):
        if self.minimizing_objective:
            return True if score < best_score else False
        else:
            return True if score > best_score else False

    def save_checkpoint(self, model):
        self.best_model_state_dict = copy.deepcopy(model.state_dict())

    def load_checkpoint(self, model):
        model.load_state_dict(self.best_model_state_dict)
