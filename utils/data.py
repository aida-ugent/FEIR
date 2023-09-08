# -*- coding: utf-8 -*-
"""
Generate synthetic data and post-processing
"""
import numpy as np
import random
from scipy import stats
from scipy.special import softmax
import pickle


def generate_random_data(m, n, seed=42, dist='uniform', normal_param=None, save_path=None, post_processing=None):
    np.random.seed(seed)
    if dist == 'uniform':
        data = np.random.rand(m, n)
    elif dist == 'normal':
        if normal_param is not None:
            lower, upper, mu, sigma = normal_param
        else:
            lower = 0
            upper = 1
            mu = 0.5
            sigma = 0.1
        data = stats.truncnorm.rvs(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=(m, n))
    if post_processing:
        data = post_processing(data)
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print('Data saved at {}.'.format(save_path))
    return data


def l1_normalization(data):
    assert len(data.shape) == 2
    return data / np.sum(data, axis=1)[:, None]


def softmax_with_invtemp(data, axis=1, inv_temp=1):
    data = data * inv_temp
    return softmax(data, axis=axis)


def scale_partial_data(data, ratio=0.5, scale=2):
    k = int(len(data) * ratio)
    data[:k] = data[:k] / scale
    return data


def generate_group_data(m, n, dim=1, top_prct=0.2, dist='normal', nparam1=(0.2, 1., 0.6, 0.05), \
                        nparam2=(0., 0.6, 0.3, 0.1), save=None):
    if dim == 1:
        m1 = m2 = m
        n1 = int(n * top_prct)
        n2 = n - n1
    elif dim == 0:
        m1 = int(m * top_prct)
        m2 = m - m1
        n1 = n2 = n
    else:
        raise NotImplementedError

    r1 = generate_random_data(m1, n1, seed=42, dist=dist, normal_param=nparam1, save_path=None,
                              post_processing=None)

    r2 = generate_random_data(m2, n2, seed=42, dist=dist, normal_param=nparam2, save_path=None,
                              post_processing=None)

    r_ = np.concatenate((r1, r2), axis=dim)
    if save is not None:
        with open(save, 'wb') as f:
            pickle.dump(r_, f)
    return r_
