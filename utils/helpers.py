# -*- coding: utf-8 -*-

import os
import sys
import yaml
import torch
import torch.nn.functional as F
import random
import importlib
import numpy as np
from time import time
from scipy import stats
from datetime import datetime
from collections import defaultdict


def string_fr_now():
    now = datetime.now()
    return now.strftime('%d_%b_%H_%M_%S')


def ensure_dir(dir_path):
    r"""Make sure the directory exists, if it does not exist, create it
    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def config_reader(cfg_file):
    with open(cfg_file) as file:
        config = yaml.safe_load(file)
    return config


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


def get_gpu_usage(device=None):
    r"""Return the reserved memory and total memory of given device in a string.
    Args:
        device: cuda.device. It is the device that the model run on.
    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """

    reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3

    return "{:.2f} G/{:.2f} G".format(reserved, total)


def get_memory_allocated(tensor):
    return tensor.element_size() * tensor.nelement()


def get_rank(a, method='ordinal', axis=None, descending=False):
    if descending:
        a = np.array(a) * -1
    return stats.rankdata(a, method=method, axis=axis)


def rank_similarity(arr1, arr2, method='kendall', precision=5, p=0.999):
    """
    :param arr1:
    :param arr2:
    :param method
    :return:
    """
    arr1 = np.round(arr1, precision)
    arr2 = np.round(arr2, precision)
    if method == 'kendall':
        return stats.kendalltau(arr1, arr2)[0]
    elif method == 'spearman':
        return stats.spearmanr(arr1, arr2)[0]
    # elif method == 'rbo':
    #     return rbolib.RankingSimilarity(arr1, arr2).rbo(p=p)
    # elif method == 'yilmaz':
    #     return tauap_b(arr1, arr2)
    else:
        raise NotImplementedError


def topk_indices(matrix, k, ascending=False):
    """
    Return the indices of top-k elements of the last dimension, without ordering within
    :param matrix:
    :param k:
    :param ascending:
    :return:
    """
    if not ascending:
        indices = np.argpartition(matrix, -k, axis=-1)[..., -k:]
    else:
        indices = np.argpartition(matrix, k, axis=-1)[..., :k]
    return indices


def cut_off_rec(R, k):
    """
    :param R: m x n
    :param k:
    :return: m x n binary matrix, indicating the top k for each row
    """
    m, n = R.shape
    res = np.zeros((m, n))
    indices = topk_indices(R, k)
    for i, ri in enumerate(indices):
        res[i, ri] = 1
    return res


def get_onehot_rec(p, k):
    m, n = p.shape
    _, rec = torch.topk(p, k, dim=1)
    rec_onehot = F.one_hot(rec, num_classes=n).sum(1).float()
    return rec, rec_onehot


def early_stopping(value, best, cur_step, max_step, bigger=True):
    r""" validation-based early stopping
    Args:
        value (float): current result
        best (float): best result
        cur_step (int): the number of consecutive steps that did not exceed the best result
        max_step (int): threshold steps for stopping
        bigger (bool, optional): whether the bigger the better
    Returns:
        tuple:
        - float,
          best result after this step
        - int,
          the number of consecutive steps that did not exceed the best result after this step
        - bool,
          whether to stop
        - bool,
          whether to update
    """
    stop_flag = False
    update_flag = False
    if bigger:
        if value > best:  # TODO tolerant equal?
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def calculate_valid_score(valid_result, valid_metric=None):
    r""" return valid score from valid result
    Args:
        valid_result (dict): valid result
        valid_metric (str, optional): the selected metric in valid result for valid score
    Returns:
        float: valid score
    """
    if valid_metric:
        return valid_result[valid_metric]
    else:
        return list(valid_result.values())[0]


def dict2str(result_dict):
    r""" convert result dict to str
    Args:
        result_dict (dict): result dict
    Returns:
        str: result str
    """

    return '    '.join([str(metric) + ' : ' + str(value) for metric, value in result_dict.items()])


def cmdargs_to_dict():
    d = {}
    for k, v in ((k.lstrip('-'), v) for k, v in (a.split('=') for a in sys.argv[1:])):
        try:
            d[k] = eval(v)
        except:
            d[k] = v
    return d


def get_my_class(module_file, model_name):
    module = importlib.import_module(module_file)
    my_model = getattr(module, model_name)
    return my_model


def init_seed(seed, reproducibility):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn
    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False



