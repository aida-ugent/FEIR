# -*- coding: utf-8 -*-
"""Abstract base model"""

# from abc import ABC, abstractmethod
# from utils.config import Config
import numpy as np
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Abstract Model class that is inherited to all models"""
    def __init__(self):
        # self.config = Config.from_json(cfg)
        super(BaseModel, self).__init__()

    def calculate_loss(self, data):
        raise NotImplementedError

    def other_parameter(self):
        if hasattr(self, 'other_parameter_name'):
            return {key: getattr(self, key) for key in self.other_parameter_name}
        return dict()

    def load_other_parameter(self, para):
        if para is None:
            return
        for key, value in para.items():
            setattr(self, key, value)


    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters' + f': {params}'



