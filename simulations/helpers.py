
# Project imports (must exist in your environment)
from rashomon.hasse import enumerate_policies, enumerate_profiles, policy_to_profile
from rashomon.aggregate import RAggregate
from rashomon.extract_pools import extract_pools, aggregate_pools
from rashomon.loss import compute_pool_means, compute_policy_means
from rashomon.metrics import make_predictions

from allocation import (
    compute_policy_variances,
    allocate_wave1,
    allocate_wave2_pools,
    create_assignments_from_alloc,
)
import numpy as np


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def _feat_groups_from_sigma_rows(R, sigma_rows):
    """Contiguous groups per feature induced by sigma rows (0=cut, 1=keep)."""
    M = len(R)
    feat_group = []
    for i in range(M):
        g = []
        cur = 0
        for lvl in range(int(R[i])):
            if lvl > 0 and len(sigma_rows[i]) > (lvl - 1) and sigma_rows[i][lvl - 1] == 0:
                cur += 1
            g.append(cur)
        feat_group.append(g)
    return feat_group

def _pool_count_from_sigma(R, all_policies, profile_separate, sigma_rows):
    """How many pools result from sigma rows (respecting profiles if requested)."""
    feat_group = _feat_groups_from_sigma_rows(R, sigma_rows)
    pools_key = set()
    for pol in all_policies:
        profile = tuple(1 if int(lv) > 0 else 0 for lv in pol) if profile_separate else None
        gtuple = tuple(feat_group[i][int(pol[i])] for i in range(len(R)))
        pools_key.add((profile, gtuple))
    return len(pools_key)

def pools_to_vector(pools, K):
    """Map {pool_id: [members]} -> np.array[policy_id -> pool_id]."""
    v = np.empty(K, dtype=int)
    for pid, members in pools.items():
        v[np.array(members, dtype=int)] = int(pid)
    return v

def _partition_map_from_sigma(R, all_policies, profile_separate, sigma_rows):
    """Return {policy_idx -> pool_id} induced by sigma rows."""
    feat_group = _feat_groups_from_sigma_rows(R, sigma_rows)
    pools_key_to_id = {}
    next_id = 0
    partition_map = {}
    for j, pol in enumerate(all_policies):
        profile = tuple(1 if int(lv) > 0 else 0 for lv in pol) if profile_separate else None
        gtuple = tuple(feat_group[i][int(pol[i])] for i in range(len(R)))
        key = (profile, gtuple)
        if key not in pools_key_to_id:
            pools_key_to_id[key] = next_id
            next_id += 1
        partition_map[j] = pools_key_to_id[key]
    return partition_map

def pools_from_partition_map(partition_map):
    """Invert {policy->pool} into {pool->list[policy]}."""
    pools = {}
    for pol, pid in partition_map.items():
        pools.setdefault(int(pid), []).append(int(pol))
    return pools

def partition_iou(pools_a, pools_b):
    # Jaccard over unordered pair sets within pools (same as comparing clusterings at the pair level)
    def pairs(pools):
        s = set()
        for members in pools.values():
            ms = list(map(int, members))
            for i in range(len(ms)):
                for j in range(i+1, len(ms)):
                    s.add((ms[i], ms[j]))
        return s
    A, B = pairs(pools_a), pairs(pools_b)
    return len(A & B) / max(len(A | B), 1)


# Simple UCB/Thompson at policy level
def _policy_stats(D, y, K):
    means = np.zeros(K); counts = np.zeros(K, dtype=int); vars_ = np.ones(K)
    if len(D) == 0: return means, counts, vars_
    for j in range(K):
        mask = (D[:,0] == j)
        c = int(mask.sum()); counts[j] = c
        if c > 0:
            vals = y[mask].flatten()
            means[j] = float(vals.mean())
            vars_[j] = float(np.var(vals, ddof=1)) if c > 1 else 1.0
    return means, counts, vars_

def _pool_ids_sorted(pools):
    return sorted(int(pid) for pid in pools.keys())

def _pool_index(pools):
    """Map pool_id -> row index (0..P-1) for aligned arrays."""
    pids = _pool_ids_sorted(pools)
    return {pid: i for i, pid in enumerate(pids)}, pids

def _pool_counts_and_means_from_stats(stats, pools):
    """
    stats: output of compute_policy_means(D,y,K): [:,0]=sum_y, [:,1]=count
    returns (counts_vec, means_vec) aligned to _pool_ids_sorted(pools)
    """
    idx_map, pids = _pool_index(pools)
    P = len(pids)
    counts = np.zeros(P, dtype=int)
    sums = np.zeros(P, dtype=float)
    for pid, members in pools.items():
        i = idx_map[int(pid)]
        m = np.array(members, dtype=int)
        counts[i] = int(stats[m, 1].sum())
        sums[i] = float(stats[m, 0].sum())
    means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return counts, means

def _pool_prior_from_wave1(D_w1, y_w1, K, pools, obs_var, prior_inflation=1.0):
    """
    Build a Normal prior per pool from Wave-1 (MAP pools):
      μ0_g = pooled empirical mean from Wave-1
      v0_g = (obs_var / n_g) * prior_inflation, with a fallback if n_g=0
    returns: (pool_ids, mu0_vec, v0_vec, n0_vec)
    """
    stats_w1 = compute_policy_means(D_w1, y_w1, K)
    n0, mu0 = _pool_counts_and_means_from_stats(stats_w1, pools)
    v0 = np.where(
        n0 > 0,
        (obs_var / np.maximum(n0, 1)) * float(prior_inflation),
        obs_var * float(prior_inflation) * 10.0  # diffuse if no Wave-1 obs
    )
    _, pids = _pool_index(pools)
    return pids, mu0, v0, n0

def _ucb_pick(means, counts, t, c=2.0):
    # classic UCB1 (Gaussian proxy)
    bonus = np.sqrt(np.maximum(0.0, (2.0 * np.log(max(t,1))) / np.maximum(counts, 1)))
    scores = means + c * bonus
    return int(np.argmax(scores))

def _thompson_pick(means, counts, prior_var=1.0, obs_var=1.0):
    # Gaussian Thompson: Normal posterior per arm (mean, var = (1/prior + n/obs_var)^-1)
    post_var = 1.0 / (1.0/prior_var + counts/obs_var)
    post_std = np.sqrt(post_var)
    samples = np.random.normal(means, np.maximum(post_std,1e-8))
    return int(np.argmax(samples))