# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys

sys.path.append('../')
from utils.helpers import timer_func


def envy_loss(R, P, k):
    m, n = R.shape[0], R.shape[1]
    # e
    eloss = 0.
    for u in range(m):
        for v in range(m):
            if u != v:
                eloss += F.relu(R[u] @ (P[v] - P[u]))
    return k * eloss / m


def envy_loss_vec(R, P, k, renorm=False):
    m = R.shape[0]
    if renorm:
        P = F.softmax(P, dim=1)
    res = R @ P.transpose(0, 1)
    envy_mat = (res - torch.diagonal(res, 0).reshape(-1, 1))
    return k * (torch.clamp(envy_mat, min=0.)).sum() / m


def inferiority_loss(R, P, k):
    m, n = R.shape[0], R.shape[1]
    iloss = 0.
    for u in range(m):
        for v in range(m):
            if u != v:
                for i in range(n):
                    iloss += max(0, R[v, i] - R[u, i]) * (1 - (1 - P[u, i]).pow(k)) * (1 - (1 - P[v, i]).pow(k))
    return iloss


def inferiority_loss_vec(R, P, k):
    m, n = P.shape
    first_term = torch.clamp(R - R.unsqueeze(1), min=0.)
    prob_mat = 1 - (1 - P).pow(k)  # m x n
    return ((first_term * prob_mat).sum(1) * prob_mat).sum() / m


def inferiority_loss_batch(R, P, idx, k):
    batch_size = P.shape[0]
    P_pow_k = 1 - (1 - P).pow(k)
    loss = ((torch.clamp(R.unsqueeze(0) - R[idx].unsqueeze(1), min=0.) * P_pow_k).sum(1) * P_pow_k[idx]).sum()
    return loss / batch_size


def utility_loss(R, P, k):
    m, n = R.shape[0], R.shape[1]
    uloss = k * torch.sum(R * P)
    return -uloss / m


def prob_const(P):
    return (P.sum(1) - 1).pow(2).mean()
