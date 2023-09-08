# -*- coding: utf-8 -*-
import os
import argparse
import pickle
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.special import softmax
import hydra
from omegaconf import open_dict, DictConfig

from utils.helpers import string_fr_now, config_reader, cmdargs_to_dict, get_my_class, init_seed, get_gpu_usage
from utils.config import config_multi_thread
from utils.logger import set_color


def get_dataloader(config, mode='batch', cluster=False):
    if mode =='batch':
        return DataLoader(torch.arange(config['m'], dtype=int), batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    elif mode == 'sample':
        u_loader = DataLoader(torch.arange(config['m'], dtype=int), batch_size=config['u_sample_size'], shuffle=True, num_workers=2, pin_memory=True)
        u_loader1 = None
        if cluster:
            u_loader1 = pickle.load(open('data/zhilian_cluster_ids.pkl', 'rb'))
        i_loader = DataLoader(torch.arange(config['n'], dtype=int), batch_size=config['i_sample_size'], shuffle=True, num_workers=2, pin_memory=True)
        return u_loader, u_loader1, i_loader
    elif mode == 'mixture':
        u_loader = DataLoader(torch.arange(config['m'], dtype=int), batch_size=config.get('out_size', 60), shuffle=True, num_workers=2, pin_memory=True)
        u_loader1 = pickle.load(open('data/zhilian_cluster_ids.pkl', 'rb'))
        i_loader = DataLoader(torch.arange(config['n'], dtype=int), batch_size=config['i_sample_size'], shuffle=True, num_workers=2, pin_memory=True)
        return u_loader, u_loader1, i_loader


@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config:DictConfig):
    import random
    saved_model_file = '{}-{}-{}.pth'.format(config['model'], string_fr_now(), random.randint(0, 999999999))
    saved_model_file = os.path.join(config['checkpoint_dir'], saved_model_file)
    with open_dict(config):
        config.saved_model_file = saved_model_file
        config['weights'] = (config['envy_weight'], config['inferiority_weight'], config['utility_weight'], config['prob_weight'])
        config.param_prefix = ['learning_rate', 'weights']
        try:
            config.learning_rate = eval(config.learning_rate)
        except TypeError:
            pass

    # random seed
    init_seed(config['seed'], config['reproducibility'])

    # data
    data = pickle.load(open(config['datafile'], 'rb')).astype('f')
    train_data = torch.from_numpy(data)

    # model
    model_class = get_my_class('model.' + config['model_file'], config['model'])

    if config['sf_init'] > 0:
        Pinit = softmax(data*config['sf_init'], axis=1)
        model = model_class(config, Pinit)
    elif config['init_file']:
        init_pi = pickle.load(open(config['init_file'], 'rb'))
        print('init with ...')
        model = model_class(config, init_pi)
    else:
        model = model_class(config)
    model = model.to(config['device'])
    # train
    trainer_class = get_my_class('executor.trainer', config['trainer'])
    trainer = trainer_class(config, model)
    if 'batch' in config['trainer']:
        idx_loader = get_dataloader(config)
        trainer.fit((train_data.to(config['device']), idx_loader), valid_data=train_data)
    elif 'sample' in config['trainer']:
        if config.get('mixture'):
            u_idx_loader, u_idx_loader_1, i_idx_loader = get_dataloader(config, mode='mixture')
        else:
            u_idx_loader, u_idx_loader_1, i_idx_loader = get_dataloader(config, mode='sample', cluster=config['cluster'])
        trainer.fit((train_data, u_idx_loader, u_idx_loader_1, i_idx_loader), valid_data=train_data)
    else:
        trainer.fit(train_data, valid_data=train_data)

    # eval
    res = trainer.evaluate(train_data, agg=False)
    for k, v in res.items():
        try:
            v = v.cpu()
            res[k] = v
        except:
            continue
    print(res['envy'].sum(-1).mean(), res['inferiority'].sum(-1).mean(), res['utility'].mean())
    res.update({'config':config, 'p': model.forward().detach().cpu().numpy()})

    # save everything
    with open(config['save_dir'] + 'RES_' + res['model_path'] + '.pkl', 'wb') as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    config_multi_thread()
    main()
    
