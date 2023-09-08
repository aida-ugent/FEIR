# -*- coding: utf-8 -*-

import sys

sys.path.append('../')

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from pymoo.indicators.hv import HV

np.set_printoptions(precision=4)
from scipy.stats import rankdata

from utils.helpers import get_rank

"""
Wealth inequality
"""


def gini(arr):
    ## Gini = \frac{2\sum_i^n i\times y_i}{n\sum_i^n y_i} - \frac{n+1}{n}
    sorted_arr = arr.copy()
    sorted_arr.sort()
    n = arr.size
    coef_ = 2. / n
    const_ = (n + 1.) / n
    weighted_sum = sum([(i + 1) * yi for i, yi in enumerate(sorted_arr)])
    return coef_ * weighted_sum / (sorted_arr.sum()) - const_


def lorenz_curve(arr, ax):
    # Ref: https://zhiyzuo.github.io/Plot-Lorenz/
    sorted_arr = arr.copy()
    sorted_arr.sort()
    X_lorenz = sorted_arr.cumsum() / arr.sum()
    X_lorenz = np.insert(X_lorenz, 0, 0)
    ax.plot(np.arange(X_lorenz.size) / (X_lorenz.size - 1), X_lorenz)
    ax.plot([0, 1], [0, 1], color='k')
    return ax


"""
Expected envy and inferiority under probabilistic recommendation as weighted sampling with replacement
"""


def expected_utility_u(Ru, ps, k):
    return Ru @ ps * k


def expected_utility(R, Pi, k):
    U = (R * Pi * k).sum(axis=1)
    return U


def expected_envy_u_v(Ru, pus, pvs, k):
    return Ru @ (pvs - pus) * k


def prob_in(ps, k):
    return 1 - (1 - ps) ** k


def prob_in_approx(ps, k):
    return k * ps


def expected_inferiority_u_v(Ru, Rv, pus, pvs, k, compensate=False, approx=False):
    differ = Rv - Ru
    if not compensate:
        differ = np.clip(differ, a_min=0, a_max=None)
    if not approx:
        return differ @ (prob_in(pus, k) * prob_in(pvs, k))
    else:
        return differ @ (prob_in_approx(pus, k) * prob_in_approx(pvs, k))


def expected_envy(R, Pi, k):
    """
    Measure expected envy for k-sized recommendation according to rec strategy Pi with respect to relevancy scores R
    :param R: m x n real-valued matrix
    :param Pi: m x n Markov matrix
    :return: E: m x n envy matrix where Euv = envy from u to v if not agg, sum of E if agg
    """
    assert np.all(np.isclose(Pi.sum(axis=1), 1.)) or np.array_equal(Pi,
                                                                    Pi.astype(bool))  # binary matrix for discrete rec
    m, n = len(R), len(R[0])
    E = np.zeros((m, m))
    for u in range(m):
        for v in range(m):
            if v == u:
                continue
            E[u, v] = expected_envy_u_v(R[u], Pi[u], Pi[v], k=k)
    E = np.clip(E, a_min=0., a_max=None)
    return E


def expected_inferiority(R, Pi, k, compensate=True, approx=False):
    """
    Measure expected inferiority for k-sized recommendation according to rec strategy Pi with respect to relevancy scores R
    :param R:
    :param Pi:
    :param k:
    :param agg:
    :return: I: m x n
    """
    assert np.all(np.isclose(Pi.sum(axis=1), 1.)) or np.array_equal(Pi,
                                                                    Pi.astype(bool))  # binary matrix for discrete rec
    m, n = len(R), len(R[0])
    I = np.zeros((m, m))
    for u in range(m):
        for v in range(m):
            if v == u:
                continue
            I[u, v] = expected_inferiority_u_v(R[u], R[v], Pi[u], Pi[v], k=k, approx=approx, compensate=compensate)

    I = np.clip(I, a_min=0., a_max=None)
    return I


def expected_envy_torch(R, Pi, k):
    m, n = len(R), len(R[0])
    E = torch.zeros(m, m)
    for u in range(m):
        for v in range(m):
            if v == u:
                continue
            E[u, v] = expected_envy_u_v(R[u], Pi[u], Pi[v], k=k)
    E = torch.clamp(E, min=0.)
    return E


def expected_envy_torch_vec(R, P, k):
    res = R @ P.transpose(0, 1)
    envy_mat = (res - torch.diagonal(res, 0).reshape(-1, 1))
    return k * (torch.clamp(envy_mat, min=0.))


def expected_inferiority_torch(R, Pi, k, compensate=False, approx=False):
    m, n = R.shape
    I = torch.zeros((m, m))
    for u in range(m):
        for v in range(m):
            if v == u:
                continue
            if not approx:
                joint_prob = prob_in(Pi[v], k) * prob_in(Pi[u], k)
            else:
                joint_prob = prob_in_approx(Pi[v], k) * prob_in_approx(Pi[u], k)

            if not compensate:
                I[u, v] = torch.clamp(R[v] - R[u], min=0., max=None) @ joint_prob
            else:
                I[u, v] = (R[v] - R[u]) @ joint_prob

    return torch.clamp(I, min=0.)


def expected_inferiority_torch_vec(R, P, k, compensate=False, approx=False):
    m, n = R.shape
    I = torch.zeros((m, m))
    P_pow_k = 1 - (1 - P).pow(k) if not approx else P * k
    for i in range(m):
        first_term = torch.clamp(R - R[i], min=0.) if not compensate else R - R[i]
        I[i] = (first_term * (P_pow_k[i] * P_pow_k)).sum(1)
    return I


def slow_onehot(idx, P):
    m = P.shape[0]
    res = torch.zeros_like(P)
    for i in range(m):
        res[i, idx[i]] = 1.
    return res


def eiu_cut_off(R, Pi, k, agg=True):
    """
    Evaluate envy, inferiority, utility based on top-k cut-off recommendation
    :param R:
    :param Pi:
    :return: envy, inferiority, utility
    """
    print('Start evaluation!')
    m, n = R.shape
    rec_onehot = slow_onehot(torch.topk(Pi, k, dim=1)[1], Pi)
    envy = expected_envy_torch_vec(R, rec_onehot, k=1)
    inferiority = expected_inferiority_torch_vec(R, rec_onehot, k=1, compensate=False, approx=False)
    utility = expected_utility(R, rec_onehot, k=1)
    if agg:
        envy = envy.sum(-1).mean()
        inferiority = inferiority.sum(-1).mean()
        utility = utility.mean()
    return envy, inferiority, utility


"""
Global congestion metrics
"""


def get_competitors(rec_per_job, rec):
    m = rec.shape[0]
    competitors = []
    for i in range(m):
        competitors.append(rec_per_job[rec[i]])
    return np.array(competitors)


def get_better_competitor_scores(rec, R):
    m, n = R.shape
    _, k = rec.shape
    user_ids_per_job = defaultdict(list)
    for i, r in enumerate(rec):
        for j in r:
            user_ids_per_job[j.item()].append(i)

    mean_competitor_scores_per_job = np.zeros((m, k))
    for i in range(m):
        my_rec_jobs = rec[i].numpy()
        my_mean_competitors = np.zeros(k)
        for j_, j in enumerate(my_rec_jobs):
            my_score = R[i, j]
            all_ids = user_ids_per_job[j].copy()
            all_ids.remove(i)
            other_scores = R[all_ids, j]
            if not all_ids:
                other_scores = np.zeros(1)
            my_mean_competitors[j_] = other_scores.mean() - my_score
        mean_competitor_scores_per_job[i] = my_mean_competitors
    return mean_competitor_scores_per_job


def get_num_better_competitors(rec, R):
    m, n = R.shape
    _, k = rec.shape
    user_ids_per_job = defaultdict(list)
    for i, r in enumerate(rec):
        for j in r:
            user_ids_per_job[j.item()].append(i)

    num_better_competitors = np.zeros((m, k))
    for i in range(m):
        my_rec_jobs = rec[i].numpy()

        better_competitors = np.zeros(k)
        for j_, j in enumerate(my_rec_jobs):
            my_score = R[i, j]
            all_ids = user_ids_per_job[j].copy()
            all_ids.remove(i)
            other_scores = R[all_ids, j]
            better_competitors[j_] = ((other_scores - my_score) > 0).sum()
        num_better_competitors[i] = better_competitors
    return num_better_competitors


def get_scores_ids_per_job(rec, R):
    scores_per_job = defaultdict(list)
    ids_per_job = defaultdict(list)

    for i in range(len(rec)):
        u = rec[i]
        for jb in u:
            jb = jb.item()
            ids_per_job[jb].append(i)
            scores_per_job[jb].append(R[i, jb].item())
    return scores_per_job, ids_per_job


def get_ranks_per_job(scores_rec):
    ranks_per_job = defaultdict(list)
    for jb in scores_rec:
        ranks_per_job[jb] = get_rank(scores_rec[jb], descending=True)
    return ranks_per_job


def get_ranks_per_user(ranks_per_job, ids_per_job):
    m = max([item for sublist in ids_per_job.values() for item in sublist]) + 1
    ranks_per_user = defaultdict(list)
    for k, v in ids_per_job.items():
        rks = ranks_per_job[k]
        for i, u in enumerate(v):
            ranks_per_user[u].append(rks[i])
    return ranks_per_user


def calculate_global_metrics(res, R, k=10):
    # get rec
    m, n = res.shape
    if not torch.is_tensor(res):
        res = torch.from_numpy(res)
    _, rec = torch.topk(res, k, dim=1)
    rec_onehot = slow_onehot(rec, res)
    try:
        rec_per_job = rec_onehot.sum(axis=0).numpy()
    except:
        rec_per_job = rec_onehot.sum(axis=0).cpu().numpy()
        rec = rec.cpu()
        R = R.cpu()
    opt_competitors = get_competitors(rec_per_job, rec)

    # mean competitors per person
    mean_competitors = opt_competitors.mean()

    # mean better competitors per person
    mean_better_competitors = get_num_better_competitors(rec, R).mean()

    # mean competitor scores - my score
    mean_diff_scores = get_better_competitor_scores(rec, R)
    mean_diff_scores[mean_diff_scores < 0] = 0.
    mean_diff_scores = mean_diff_scores.mean()

    # mean rank
    scores_opt, ids_opt = get_scores_ids_per_job(rec, R)
    ranks_opt = get_ranks_per_job(scores_opt)
    ranks_per_user_opt = get_ranks_per_user(ranks_opt, ids_opt)
    mean_rank = np.array(list(ranks_per_user_opt.values())).mean()

    # gini
    gini_index = gini(rec_per_job)

    return {'mean_competitors': mean_competitors, 'mean_better_competitors': mean_better_competitors, \
            'mean_scores_diff': mean_diff_scores, 'mean_rank': mean_rank, 'gini_index': gini_index}


def get_HV(pfs, key='ei_vs_u_n', ref=np.array([1., -0.9])):
    our_sk = pfs[key][0]
    their_sk = pfs[key][1]

    our_sk = np.array([[i[0], -i[1]] for i in our_sk])

    their_sk = np.array([[i[0], -i[1]] for i in their_sk])

    ind = HV(ref_point=ref)

    print('Ours: {:.3f}\tTheirs: {:.3f}'.format(ind(our_sk), ind(their_sk)))


def get_all_HVs(pfs, keys=['ei_vs_u_n', 'i_vs_u_n'], ref=np.array([1, -0.8])):
    for key in keys:
        print(key)
        get_HV(pfs, key=key, ref=ref)


def find_min_above_threshold(tuples, threshold):
    # filter the list of tuples to only include the tuples where the first element is greater than the threshold
    filtered_tuples = filter(lambda x: x[1] >= threshold, tuples)
    # find the minimum value of the second element of the filtered tuples
    try:
        return min(filtered_tuples, key=lambda x: x[0])[0]
    except ValueError:
        return np.nan


def get_min_ei_above_u(pfs, key='ei_vs_u_n', threshold=0.9):
    our_sk = pfs[key][0]
    their_sk = pfs[key][1]

    our_sk = np.array([[i[0], i[1]] for i in our_sk])

    their_sk = np.array([[i[0], i[1]] for i in their_sk])

    print('Ours: {:.3f}\tTheirs: {:.3f}'.format(find_min_above_threshold(our_sk, threshold),
                                                find_min_above_threshold(their_sk, threshold)))
