import numpy as np
from rashomon.extract_pools import lattice_edges, extract_pools, aggregate_pools

import numpy as np

def get_initial_coverage_allocations(policies_ids_profiles):
    """
    Returns a list with one (arbitrary) policy index from each profile.
    Ensures every profile is covered in the first allocation round.
    """
    D = []
    for profile_idx, policy_idxs in policies_ids_profiles.items():
        # Just pick a single policy in this profile
        D.append(np.random.choice(policy_idxs)) # or just policy_idxs[0]
    return D

def compute_adaptive_proxy_scores(policy_stats, map_idx, pi_policies_r_list, posterior_weights):
    num_policies = policy_stats.shape[0]
    # 1. Coverage: 1 if unobserved, else 0
    coverage = np.array([1 if policy_stats[v, 1] == 0 else 0 for v in range(num_policies)])

    # 2. Pool variance: 1 / (# observed in pool + 1)
    pool_assignments = [pi_policies_r_list[map_idx][v] for v in range(num_policies)]
    unique_pools = set(pool_assignments)
    pool_obs_counts = {p: 0 for p in unique_pools}
    for v in range(num_policies):
        if policy_stats[v, 1] > 0:
            pool_obs_counts[pool_assignments[v]] += 1
    pool_var = np.array([1 / (pool_obs_counts[pool_assignments[v]] + 1) for v in range(num_policies)])

    # 3. Entropy: posterior over pool assignments from RPS ensemble
    entropy = np.zeros(num_policies)
    for v in range(num_policies):
        pool_hist = {}
        total_weight = 0
        for r, pi_policies_r in enumerate(pi_policies_r_list):
            g = pi_policies_r[v]
            w = posterior_weights[r]
            pool_hist[g] = pool_hist.get(g, 0) + w
            total_weight += w
        p_g = np.array(list(pool_hist.values())) / (total_weight + 1e-12)
        entropy[v] = -np.sum(p_g * np.log(p_g + 1e-12))

    # Normalization to [0,1]
    def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-12)
    cov_norm = norm(coverage)
    var_norm = norm(pool_var)
    ent_norm = norm(entropy)
    score = cov_norm + var_norm + ent_norm  # or use np.max([cov_norm, var_norm, ent_norm], axis=0)
    return score, cov_norm, var_norm, ent_norm

# ---- boundary probabilities work ----
def get_prob_allocations(probs, n1):
    """
    Allocates a total sample size n1 across arms according to a probability vector,
    returning an integer allocation vector that sums exactly to n1.

    Args:
        probs (np.ndarray): Probability weights for each arm/policy (length = num_policies).
        n1 (int): Total number of samples to allocate.

    Returns:
        np.ndarray: Integer allocation for each policy (length = num_policies), sum = n1.
    """
    total_prob = probs.sum()
    alloc_floats = n1 * probs / total_prob
    # Integer allocation: round, then correct for sum
    alloc_ints = np.floor(alloc_floats).astype(int)
    remainder = n1 - alloc_ints.sum()
    # Distribute remaining units to policies with largest fractional part
    if remainder > 0:
        frac = alloc_floats - alloc_ints
        idx_sorted = np.argsort(-frac)  # descending fractional part
        for i in idx_sorted[:remainder]:
            alloc_ints[i] += 1
    return alloc_ints

def boundary_probability(policy, R, H):
    """
    Calculates the probability that a given policy lies on the boundary of a pool,
    under the canonical first-wave Rashomon allocation heuristic.

    Args:
        policy (tuple or list): The policy as a vector of categorical choices (length = M).
        R (list or array): Number of levels for each feature (length = M).
        H (int): Pooling parameter (max pool size).

    Returns:
        float: Theoretical boundary probability for this policy.
    """
    prod = 1.0
    for i in range(len(policy)):
        vi = policy[i]
        Ri = R[i]
        numerator = 2 * min(vi, Ri - 1 - vi)
        denom = Ri - 1
        if denom == 0:
            continue
        factor = 1 - numerator / denom
        prod *= factor**(H-1)
    return 1 - prod

def compute_boundary_probabilities(all_policies, R, H):
    """
    Computes the boundary probability for each policy in the lattice.

    Args:
        all_policies (list of tuples): List of policies.
        R (list or array): Number of levels for each feature.
        H (int): Pooling parameter.

    Returns:
        np.ndarray: Array of boundary probabilities for each policy (same order as all_policies).
    """
    return np.array([boundary_probability(p, R, H) for p in all_policies])

def create_assignments_from_alloc(alloc):
    """
    Converts an integer allocation vector into an array of assigned policy indices.
    Each policy index is repeated according to its allocation count;
    the array D matches each observation to its assigned policy.

    Args:
        alloc (np.ndarray): Integer allocation for each policy (length = num_policies).

    Returns:
        np.ndarray: (n1, 1) array of assigned policy indices, where n1 = alloc.sum().
    """
    D = []
    for i, count in enumerate(alloc):
        if count > 0:
            D.extend([i] * count)
    D = np.array(D, dtype=int).reshape(-1, 1)
    return D