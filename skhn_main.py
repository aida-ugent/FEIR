# -*- coding: utf-8 -*-
import os
from time import time
import argparse
import pickle
import pandas as pd
import hydra
from omegaconf import open_dict, DictConfig

from model.loss import *
from model.sinkhorn import SinkhornSolver
from evaluation.metrics import *
from utils.helpers import *


def valid_solution(s, k):
    flag = True
    for row_ in s:
        if (row_ > 0).sum() < k:
            flag = False
            return flag
    return flag


def sinkhorn_opt(R, k=10, eps_ranges=[0.1, 0.2, 0.3, 2.9, 3.1], device='cpu'):
    pis = {}
    for ix, i in enumerate(eps_ranges):
        epsilon = i
        start = time()
        solver = SinkhornSolver(epsilon=epsilon, ground_metric=None, iterations=10000, device=device)
        pi = solver.forward(R * (-1))
        print('Done {} in {}'.format(epsilon, time() - start))
        if not valid_solution(pi, k=k):
            print('Not valid at {}'.format(i))
            break
        pis[ix] = (epsilon, pi.detach().cpu())
    return pis


def eval(R, pis, k, device='cpu'):
    res = []
    for i, (epsilon, pi) in pis.items():
        rec_envy, rec_inferiority, rec_utility = eiu_cut_off(R, pi.to(device), k=k, agg=False)
        rec_envy = rec_envy.cpu().numpy()
        rec_inferiority = rec_inferiority.cpu().numpy()
        rec_utility = rec_utility.cpu().numpy()
        avg_ne = rec_envy.sum(-1).mean()
        avg_ni = rec_inferiority.sum(-1).mean()
        avg_nu = rec_utility.mean()
        std_ne = rec_envy.sum(-1).std()
        std_ni = rec_inferiority.sum(-1).std()
        std_nu = rec_utility.std()
        metrics = calculate_global_metrics(pi, R, k=k)
        metrics['eval/envy'] = avg_ne
        metrics['eval/inferiority'] = avg_ni
        metrics['eval/e+i'] = avg_ne + avg_ni
        metrics['eval/utility'] = avg_nu
        metrics['eval/envy_std'] = std_ne
        metrics['eval/inferiority_std'] = std_ni
        metrics['eval/utility_std'] = std_nu
        metrics['p_id'] = i
        metrics['epsilon'] = epsilon
        res.append(metrics)
    df = pd.DataFrame(res)
    return df


def save_sinkhorn_results(res, pis, save_dir='./', k=20):
    ensure_dir(save_dir)
    with open(save_dir + 'sinkhorn_results_{}.pkl'.format(k), 'wb') as f:
        pickle.dump(res, f)
    with open(save_dir + 'sinkhorn_pis_{}.pkl'.format(k), 'wb') as f:
        pickle.dump(pis, f)


@timer_func
@hydra.main(version_base=None, config_path='configs', config_name='sinkhorn_pplgroup')
def main(config: DictConfig):
    datafile = config['data_file']
    R = torch.from_numpy(pickle.load(open(datafile, 'rb')).astype('f')).to(config['device'])
    ks = config['ks']
    if 'eps' not in config:
        eps_range = np.concatenate(
            (np.linspace(config['eps_start'], config['eps_end'], config['eps_num'], endpoint=False), \
             np.arange(1, config['eps_range'] + 1) * config['eps_scale']))
    else:
        eps_range = config['eps']
    save_dir = config['save_dir']

    for k in ks:
        k_pis = sinkhorn_opt(R, k, eps_range, device=config['device'])
        k_res = eval(R, k_pis, k, device=config['device'])
        save_sinkhorn_results(k_res, k_pis, save_dir, k)


if __name__ == '__main__':
    main()
