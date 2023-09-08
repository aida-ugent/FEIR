import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys

sys.path.append('../')
from model.loss import *
from model.base_model import BaseModel


class Popt(BaseModel):
    def __init__(self, config, ini=None):
        super().__init__()
        self.m = config['m']
        self.n = config['n']
        self.k = config['k']
        self.inv_temperature = config['sf_temp']
        self.device = config['device']
        self.weights = config['weights']
        self.activation = config['model_activation']
        if ini is None:
            self.P = torch.nn.Parameter(torch.Tensor(self.m, self.n))
            self.P.data.uniform_(0, 2/(self.m*self.n))
        else:
            self.P = torch.nn.Parameter(torch.Tensor(ini))

    def forward(self):
        if self.activation == 'hardtanh':
            return F.hardtanh(self.P, 0, 1 / self.k)
        elif self.activation == 'softmax':
            return F.softmax(self.P * self.inv_temperature, dim=1)

    def calculate_loss(self, R):
        p = self.forward()
        eloss = envy_loss_vec(R, p, self.k)
        iloss = inferiority_loss_vec(R, p, self.k)
        uloss = utility_loss(R, p, self.k)
        pconst = prob_const(p)
        return eloss, iloss, uloss, pconst


class PoptBatch(Popt):
    def calculate_loss(self, R, ids):
        p = self.forward()
        eloss = envy_loss_vec(R, p, self.k)
        iloss = inferiority_loss_batch(R, p, ids, self.k)
        uloss = utility_loss(R, p, self.k)
        pconst = prob_const(p)
        return eloss, iloss, uloss, pconst


class PoptSample(Popt):
    def calculate_loss(self, R, p, renorm=False):
        eloss = envy_loss_vec(R, p, self.k, renorm=renorm)
        iloss = inferiority_loss_vec(R, p, self.k)
        uloss = utility_loss(R, p, self.k)
        pconst = prob_const(p)
        return eloss, iloss, uloss, pconst

