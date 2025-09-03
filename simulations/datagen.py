import numpy as np
from helpers import (
    pools_from_partition_map,
    _pool_count_from_sigma,
    _partition_map_from_sigma,
)

def generate_data_from_assignments(D, all_policies, oracle_outcomes, sig=1.0):
    """
    Generates data from oracle outcomes by adding Gaussian noise.

    Args:
        D (np.ndarray): (n, 1) array of assigned policy indices (0 to num_policies-1)
        all_policies (list): List of all policy tuples (in global index order)
        oracle_outcomes (np.ndarray): Vector of oracle outcomes, indexed by policy ID
        sig (float): Standard deviation of outcome noise

    Returns:
        X (np.ndarray): (n, M) array of policy feature vectors
        y (np.ndarray): (n, 1) array of noisy outcomes
    """
    n = D.shape[0]
    M = len(all_policies[0])
    X = np.zeros((n, M))
    y = np.zeros((n, 1))
    for i in range(n):
        policy_idx = D[i, 0]
        policy = all_policies[policy_idx]
        X[i, :] = policy
        mu = oracle_outcomes[policy_idx]
        y[i, 0] = np.random.normal(loc=mu, scale=sig)
    return X, y

# -------------------------------------------------------------
# causal function registry
def phi_basic(policy):
    x = np.array(policy)
    return 2 * x[0] + 0.5 * x[1]**2 + x[0]*x[2] + 0.2 * x[1] * x[2] + 0.5*x[1]*x[3]

def phi_linear_interact(policy, R, shift=0.0):
    x = np.array(policy) / (np.array(R) - 1)
    lin = np.dot(x, np.linspace(1, 2, len(x)))  # increasing weights
    pairwise = sum(x[i]*x[i+1] for i in range(len(x)-1))
    return lin + 0.3 * pairwise + shift

def phi_peak(policy, R, center=None, w_scale=1.0):
    x = np.array(policy) / (np.array(R) - 1)
    M = len(x)
    if center is None:
        center = np.ones(M) * 0.5  # center in [0,1]^M
    w = np.linspace(1, 2, M) * w_scale
    value = -np.sum(w * (x - center)**2)  # peak at center
    return float(value)

def phi_grouped_smooth2(policy, R):
    x = np.array(policy) / (np.array(R) - 1)
    centers = [np.full_like(x, val) for val in [0.2, 0.5, 0.8]]
    weights = [1.0, 3.0, 5.0]
    sharpness = 0.5

    group_vals = [
        w * np.exp(-np.sum((x - c)**2) / sharpness)
        for c, w in zip(centers, weights)
    ]
    lin = np.dot(x, np.linspace(0.2, 1.0, len(x)))

    return sum(group_vals) + 0.3 * lin

def phi_grouped_coarse(policy, R, freq=2.0, sharpness=0.15):
    x = np.array(policy) / (np.array(R) - 1)
    x_mean = np.mean(x)
    x_sum = np.sum(x)

    # Broad sinusoidal ridges across average and total activation
    ridge = np.sin(freq * np.pi * x_mean) + np.cos(freq * np.pi * x_sum / len(x))

    # Smooth global interaction bump (moderately centered)
    bump = np.exp(-np.sum((x - 0.5)**2) / sharpness)

    return 2.0 + ridge + 2.0 * bump

def phi_grouped_smooth(policy, R, shift=0.0):
    x = np.array(policy) / (np.array(R) - 1)  # normalize to [0, 1]
    centers = [
        np.full_like(x, 0.2),
        np.full_like(x, 0.5),
        np.full_like(x, 0.8),
    ]
    values = [np.exp(-np.sum((x - c)**2) / 0.05) for c in centers]
    interact = sum(x[i]*x[i+1] for i in range(len(x)-1))
    return 1.0 + 2.0 * max(values) + 0.3 * interact + shift

def phi_5d(policy):
    x = np.array(policy)
    return 2 * x[0] + 0.5 * x[1]**2 + x[0]*x[2] + 0.2 * x[1] * x[2] + 0.5*x[1]*x[3] + 0.1 * x[4]**2 + x[4]*x[0]

# -------------------------------------------------------------
# generate a *fixed-size* valid partition with H_true pools

def _ordered_pool_ids(pools, order="id_asc"):
    """
    Return a stable ordering of pool ids.
    order: "id_asc" | "size_asc" | "size_desc"
    """
    pids = list(pools.keys())
    if order == "id_asc":
        return sorted(pids)
    sizes = {pid: len(pools[pid]) for pid in pids}
    rev = (order == "size_desc")
    return [pid for pid, _ in sorted(sizes.items(), key=lambda kv: kv[1], reverse=rev)]


def _assign_pool_means(
    pools,
    mean_spec=None,               # None | ("normal", mu0, tau) | ("uniform", a, b) | ("linspace", a, b) | ("custom", values)
    order="id_asc",               # ordering used for "linspace" and "custom" sequence assignment
    rng=None,
    force_unique_best=False,      # if True, enforce a unique best pool by raising its mean by best_gap over the 2nd best
    best_gap=0.0,                 # Δ to add above runner-up for the chosen best pool
    best_pool_choice="smallest"   # "smallest" | "largest" | int pool_id
):
    """
    Return mu_pool_used: {pool_id -> mean}

    mean_spec:
      - None or ("normal", mu0, tau): draw each pool mean ~ N(mu0, tau)
      - ("uniform", a, b): draw each mean ~ U[a, b]
      - ("linspace", a, b): assign linearly spaced means across ordered pool ids
      - ("custom", values): if 'values' is a dict {pid: mean}, use it directly (others drawn normal with mu0=0,tau=1).
                            if 'values' is a list/tuple, align by the chosen 'order' of pool ids.

    If force_unique_best=True, pick a pool to be best (per 'best_pool_choice') and
    raise its mean so it's at least best_gap above the second largest mean.
    """
    if rng is None:
        rng = np.random

    pids = _ordered_pool_ids(pools, order=order)
    G = len(pids)

    # defaults
    mode = "normal"; mu0 = 0.0; tau = 1.0
    if mean_spec is None:
        pass
    elif isinstance(mean_spec, tuple):
        mode = mean_spec[0]
        if mode == "normal":
            mu0, tau = float(mean_spec[1]), float(mean_spec[2])
        elif mode in ("uniform", "linspace"):
            a, b = float(mean_spec[1]), float(mean_spec[2])
        elif mode == "custom":
            values = mean_spec[1]
        else:
            raise ValueError(f"Unknown mean_spec mode: {mode}")
    else:
        raise ValueError("mean_spec must be None or a tuple.")

    mu_pool_used = {}

    if mode == "normal":
        for pid in pids:
            mu_pool_used[pid] = float(rng.normal(mu0, tau))

    elif mode == "uniform":
        for pid in pids:
            mu_pool_used[pid] = float(rng.uniform(a, b))

    elif mode == "linspace":
        if G == 1:
            mu_pool_used[pids[0]] = float(a)
        else:
            vals = np.linspace(a, b, G)
            for pid, val in zip(pids, vals):
                mu_pool_used[pid] = float(val)

    elif mode == "custom":
        if isinstance(values, dict):
            # use provided; backfill missing with N(0,1)
            for pid in pids:
                if pid in values:
                    mu_pool_used[pid] = float(values[pid])
                else:
                    mu_pool_used[pid] = float(rng.normal(0.0, 1.0))
        elif isinstance(values, (list, tuple, np.ndarray)):
            if len(values) != G:
                raise ValueError(f"custom sequence length {len(values)} != #pools {G}")
            for pid, val in zip(pids, values):
                mu_pool_used[pid] = float(val)
        else:
            raise ValueError("custom values must be dict or sequence")
    else:
        raise ValueError(f"Unsupported mean_spec mode: {mode}")

    # Optional: enforce a unique best with gap Δ
    if force_unique_best and G >= 2:
        if isinstance(best_pool_choice, int):
            best_pid = int(best_pool_choice)
            if best_pid not in mu_pool_used:
                raise ValueError(f"best_pool_choice pid {best_pid} not in pools")
        elif best_pool_choice == "smallest":
            sizes = {pid: len(pools[pid]) for pid in pids}
            best_pid = min(pids, key=lambda pid: sizes[pid])
        elif best_pool_choice == "largest":
            sizes = {pid: len(pools[pid]) for pid in pids}
            best_pid = max(pids, key=lambda pid: sizes[pid])
        else:
            raise ValueError("best_pool_choice must be 'smallest','largest', or a pool id")

        others = [mu_pool_used[pid] for pid in pids if pid != best_pid]
        runner_up = max(others) if others else mu_pool_used[best_pid]
        mu_pool_used[best_pid] = float(runner_up + float(best_gap))

    return mu_pool_used


def _broadcast_pool_means_to_policies(partition_map, mu_pool_used, K):
    oracle_outcomes = np.zeros(K, dtype=float)
    for j in range(K):
        oracle_outcomes[j] = mu_pool_used[int(partition_map[j])]
    return oracle_outcomes

def generate_true_partition_from_means(
    all_policies, R, partition_map,
    pool_means,                  # dict {pool_id: mean} OR sequence aligned to order="id_asc"/"size_*"
    order="id_asc"
):
    """
    Use an existing valid partition (partition_map) and a set of means to create oracle_outcomes.

    pool_means:
      - dict {pid: mean} -> used directly
      - sequence -> assigned to pools in the specified 'order' of pool ids
    """
    pools = pools_from_partition_map(partition_map)
    if isinstance(pool_means, dict):
        mu_pool_used = {int(pid): float(mu) for pid, mu in pool_means.items()}
        # backfill any missing pool ids (rare) with 0.0
        for pid in pools.keys():
            if pid not in mu_pool_used:
                mu_pool_used[pid] = 0.0
    else:
        pids = _ordered_pool_ids(pools, order=order)
        vals = list(pool_means)
        if len(vals) != len(pids):
            raise ValueError(f"pool_means length {len(vals)} != #pools {len(pids)}")
        mu_pool_used = {pid: float(val) for pid, val in zip(pids, vals)}

    oracle_outcomes = _broadcast_pool_means_to_policies(partition_map, mu_pool_used, len(all_policies))
    # Shared outcome bank for all branches (common random numbers)
    return oracle_outcomes, mu_pool_used

def generate_true_partition_fixed_pools(
    all_policies, R, H_true,
    mu0=0.0, tau=1.0, random_seed=None, profile_separate=True, max_iters=5000,
    mean_spec=None,                # NEW: see _assign_pool_means docstring
    force_unique_best=False,       # NEW
    best_gap=0.0,                  # NEW
    best_pool_choice="smallest",   # NEW
    mean_order="id_asc"            # NEW: ordering for linspace/custom sequence
):
    """
    Construct a permissible partition with exactly (or as close as possible to) H_true pools,
    then assign pool means per 'mean_spec'. Optionally enforce a unique best pool with gap Δ.

    Returns:
      partition_map: {policy_idx -> pool_id}
      oracle_outcomes: (K,) true mean per policy (distinct per pool)
      mu_pool_used: {pool_id -> mean}
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    rng = np.random

    M = len(R)
    # start fully pooled within each profile: sigma=1 everywhere (keep)
    sigma_rows = []
    for i in range(M):
        nb = int(R[i]) - 1
        sigma_rows.append(np.ones(max(nb, 0), dtype=int))

    # Greedy random search to hit H_true
    all_pols = all_policies

    def count_current():
        return _pool_count_from_sigma(R, all_pols, profile_separate, sigma_rows)

    cur = count_current()
    it = 0
    while cur != H_true and it < max_iters:
        it += 1
        candidates = []
        for i in range(M):
            for b in range(len(sigma_rows[i])):
                candidates.append((i, b))
        if len(candidates) == 0:
            break
        i, b = candidates[np.random.randint(0, len(candidates))]
        old = sigma_rows[i][b]
        if cur < H_true:
            sigma_rows[i][b] = 0  # create a split
        else:
            sigma_rows[i][b] = 1  # merge
        newc = count_current()
        if abs(newc - H_true) <= abs(cur - H_true) or np.random.rand() < 0.2:
            cur = newc
        else:
            sigma_rows[i][b] = old

    # Build final partition and pools
    partition_map = _partition_map_from_sigma(R, all_pols, profile_separate, sigma_rows)
    pools = pools_from_partition_map(partition_map)

    # Assign pool means per spec
    if mean_spec is None:
        # backward-compatible default: normal(mu0, tau)
        mean_spec_eff = ("normal", mu0, tau)
    else:
        mean_spec_eff = mean_spec

    mu_pool_used = _assign_pool_means(
        pools,
        mean_spec=mean_spec_eff,
        order=mean_order,
        rng=rng,
        force_unique_best=force_unique_best,
        best_gap=best_gap,
        best_pool_choice=best_pool_choice
    )

    # Broadcast to policies
    oracle_outcomes = _broadcast_pool_means_to_policies(partition_map, mu_pool_used, len(all_pols))
    return partition_map, oracle_outcomes, mu_pool_used
