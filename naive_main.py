# -*- coding: utf-8 -*-
import glob
import torch
import pickle
import hydra
from omegaconf import open_dict, DictConfig

from evaluation.metrics import *
from utils.helpers import *

"""
Calculate metrics for different ks
"""


def naive_rec_results(R, k=10):
    m, n = R.shape
    naive_rec_envy, naive_rec_inferiority, naive_rec_utility = eiu_cut_off(R, R, k=k, agg=False)
    # print(naive_rec_envy, naive_rec_inferiority, naive_rec_utility)
    avg_ne = naive_rec_envy.sum(-1).mean()
    avg_ni = naive_rec_inferiority.sum(-1).mean()
    avg_nu = naive_rec_utility.mean()
    std_ne = naive_rec_envy.sum(-1).std()
    std_ni = naive_rec_inferiority.sum(-1).std()
    std_nu = naive_rec_utility.std()

    _, naive_rec = torch.topk(R, k, dim=1)
    naive_rec_onehot = slow_onehot(naive_rec, R)
    res = {'R': R, 'naive_envy': naive_rec_envy, 'naive_inferiority': naive_rec_inferiority,
           'naive_utility': naive_rec_utility, \
           'eval/envy': avg_ne, 'eval/inferiority': avg_ni, 'eval/utility': avg_nu, 'eval/e+i': avg_ne + avg_ni, \
           'eval/envy_std': std_ne, 'eval/inferiority_std': std_ni, 'eval/utility_std': std_nu, \
           'naive_rec': naive_rec,
           'naive_rec_onehot': naive_rec_onehot}
    global_fair = calculate_global_metrics(R, R, k=k)
    res.update(global_fair)
    for k, v in res.items():
        try:
            v = v.cpu().numpy()
            res[k] = v
        except:
            continue
    return res


@timer_func
@hydra.main(version_base=None, config_path='configs', config_name='naive')
def main(config: DictConfig):
    data = pickle.load(open(config.datafile, 'rb'))
    res = {}
    for k in config.ks:
        res[k] = naive_rec_results(torch.from_numpy(data).float().to(config.device), k=k)

    with open(config.savepath, 'wb') as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    main()
