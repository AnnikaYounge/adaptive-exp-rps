
import numpy as np
# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _largest_remainder(weights, N, floors=None):
    """
    Largest remainder (Hamilton) apportionment with safe handling of NaN/inf/negatives.
    Returns an integer vector summing to N (plus floors if provided).
    """
    w = np.asarray(weights, dtype=float).reshape(-1)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w[w < 0] = 0.0
    total = w.sum()
    if total <= 0:
        # uniform if all zero/invalid
        w = np.ones_like(w, dtype=float)
        total = w.sum()
    w = w / total

    if floors is None:
        floors = np.zeros_like(w, dtype=int)
    else:
        floors = np.asarray(floors, dtype=int).reshape(-1)
        if floors.size != w.size:
            raise ValueError("floors length must match weights")
        floors[floors < 0] = 0

    # remaining quota after floors
    rem_quota = int(N) - int(floors.sum())
    if rem_quota < 0:
        # if floors over-allocate, trim proportionally
        over = -rem_quota
        frac = floors / max(floors.sum(), 1)
        cut = np.minimum(floors, np.floor(frac * over).astype(int))
        floors = floors - cut
        rem_quota = 0

    quotas = w * rem_quota
    base = np.floor(quotas).astype(int)
    rem = rem_quota - int(base.sum())

    frac = quotas - base
    order = np.argsort(-frac)
    add = np.zeros_like(base)
    if rem > 0:
        add[order[:rem]] = 1

    return floors + base + add


# Variance utilities
def compute_policy_variances(D, y, num_policies, ddof=1):
    """
    Compute per-policy (unadjusted) sample variances and counts from (policy index, outcome) observations.
    """
    D = np.asarray(D).reshape(-1).astype(int)
    y = np.asarray(y).reshape(-1).astype(float)
    K = int(num_policies)

    n = np.bincount(D, minlength=K).astype(int)
    s1 = np.bincount(D, weights=y, minlength=K)
    s2 = np.bincount(D, weights=y*y, minlength=K)

    var = np.full(K, np.nan, dtype=float)
    mask = n > 1
    with np.errstate(invalid="ignore", divide="ignore"):
        if np.any(mask):
            var[mask] = (s2[mask] - (s1[mask]**2)/n[mask]) / (n[mask] - ddof)
            var[mask] = np.clip(var[mask], 0.0, None)
    return var, n


# Wave-1 allocation (policy-level) with enforced one-per-policy coverage
# and variance inflation for unobserved

def allocate_wave1(rule, sigmas, counts, N):
    """
    Allocate Wave-1 budget across policies using a classical objective:
      - 'neyman_policy': minimise average MSE  -> weights ~ sigma_i
      - 'minimax_policy' (equal_precision): minimise worst SE -> weights ~ sigma_i^2
      - 'best_arm' (OCBA proxy): weights ~ sigma_i^2 / gap_i^2   (gaps not provided here; fallback to Neyman)

    Refinements:
      * Enforce >=1 for any unobserved policy (requires N >= #unobserved).
      * Inflate unobserved sigmas so proportional rules prioritise covering them.
    """
    rule = str(rule).lower()
    s = np.asarray(sigmas, dtype=float).reshape(-1)
    c = np.asarray(counts, dtype=int).reshape(-1)
    K = s.size
    if c.size != K:
        raise ValueError("counts and sigmas must have same length")
    N = int(N)

    # Enforce one-per-unobserved (hard coverage)
    need = (c == 0)
    m = int(need.sum())
    if m > 0 and N < m:
        raise ValueError(f"Wave-1 budget N={N} is less than #unobserved m={m}. Increase N.")

    floors = np.zeros(K, dtype=int)
    if m > 0:
        floors[need] = 1  # one-per-unobserved

    # Variance inflation for unobserved (huge sigma so rules pick them)
    s_eff = s.copy()
    finite_mask = np.isfinite(s_eff)
    s_max = np.max(s_eff[finite_mask]) if np.any(finite_mask) else 1.0
    s_eff[~finite_mask] = s_max
    s_eff[c == 0] = max(s_max, 1.0) * 1e6

    # Build weights
    if rule in ("neyman"):
        weights = s_eff
    elif rule in ("minimax"):
        weights = s_eff**2
    elif rule in ("best_arm"):
        # gap-aware OCBA not implemented in Wave-1 here; default to Neyman weights
        weights = s_eff
    else:
        raise ValueError("rule must be one of {'neyman','minimax','best_arm'}")

    alloc = _largest_remainder(weights, N, floors=floors)
    return alloc


# -----------------------------------------------------------------------------
# Wave-2 allocation (pool-level) -- same three objectives, at pool granularity
# -----------------------------------------------------------------------------


def allocate_wave2(rule, pool_sigmas, N, gaps=None, floors_per_pool=0):
    """
    Allocate pooled sample sizes for Wave-2 across POOLS.

    Normalized rule names (no suffix):
      'uniform' | 'neyman' | 'minimax' | 'best_pool'

    Backward-compatible: if a caller passes 'neyman_pool' or 'minimax_pool',
    we strip the '_pool' suffix.
    """
    rule = str(rule).lower()
    s = np.asarray(pool_sigmas, dtype=float).reshape(-1)
    P = s.size
    floors = np.full(P, int(floors_per_pool), dtype=int)

    if rule == "uniform":
        weights = np.ones(P, dtype=float)

    elif rule == "neyman":
        weights = s

    elif rule == "minimax":
        weights = s**2

    elif rule == "best_pool":
        if gaps is None:
            weights = s
        else:
            g = np.asarray(gaps, dtype=float).reshape(-1)
            if g.size != P:
                raise ValueError("gaps length must match number of pools")
            g = np.maximum(g, 1e-12)
            weights = (s**2) / (g**2)

    else:
        raise ValueError("rule must be one of {'uniform','neyman','minimax','best_pool'}")

    return _largest_remainder(weights, int(N), floors=floors)

def allocate_wave2_pools(
    rule,
    pool_to_policies,       # dict: pool_id -> list of policy indices
    policy_sigmas,          # per-policy SDs (sqrt of variance)
    N,
    policy_counts=None,
    pool_weights=None,      # optional dict: pool_id -> weights for members (sums to 1)
    pool_gaps=None,         # needed only if using 'best_pool'
    within_rule="neyman",   # {'neyman', 'minimax', 'uniform'}
):
    """
    Two-stage Wave-2 allocation:
      1) allocate counts across POOLS by objective `rule` (neyman_pool/minimax_pool/best_pool),
      2) split each pool's count across its member POLICIES by `within_rule`.

    Returns
    -------
    alloc_policy : integer counts per policy summing to N
    """
    policy_sigmas = np.asarray(policy_sigmas, dtype=float)
    K = policy_sigmas.size
    counts = np.zeros(K, dtype=int) if policy_counts is None else np.asarray(policy_counts, dtype=int)
    alloc_policy = np.zeros(K, dtype=int)

    # ---- 1) compute a per-pool SE proxy and allocate across pools
    # proxy SE for pool mean: s_g = sqrt( sum_{i in g} w_i^2 * sigma_i^2 / max(n_i,1) )
    pool_ids = sorted(pool_to_policies.keys())
    P = len(pool_ids)
    s_pool = np.zeros(P, dtype=float)

    for j, g in enumerate(pool_ids):
        members = list(pool_to_policies[g])
        m = len(members)
        s = policy_sigmas[members]
        n = np.maximum(counts[members], 1)
        if pool_weights is not None and g in pool_weights:
            w = np.asarray(pool_weights[g], dtype=float)
            if w.size != m:
                raise ValueError(f"pool_weights[{g}] has wrong length")
            w = w / (w.sum() + 1e-12)
        else:
            w = np.full(m, 1.0/m)
        var_g = np.sum((w**2) * (s**2) / n)
        s_pool[j] = np.sqrt(max(var_g, 1e-12))

    pool_alloc = allocate_wave2(rule, s_pool, N, gaps=pool_gaps)

    # ---- 2) split within each pool by within_rule
    for j, g in enumerate(pool_ids):
        members = list(pool_to_policies[g])
        cnt = int(pool_alloc[j])
        if cnt <= 0:
            continue
        s = policy_sigmas[members]
        m = len(members)

        if pool_weights is not None and g in pool_weights:
            w = np.asarray(pool_weights[g], dtype=float)
            w = w / (w.sum() + 1e-12)
        else:
            w = np.full(m, 1.0/m)

        if within_rule == "neyman":
            weights = w * s
        elif within_rule == "minimax":
            weights = s**2
        elif within_rule == "uniform":
            weights = np.ones(m, dtype=float)
        else:
            raise ValueError("within_rule must be one of {'neyman','minimax','uniform'}")

        if not np.isfinite(weights).any() or np.all(weights <= 0):
            weights = np.ones(m, dtype=float)

        alloc_members = _largest_remainder(weights, cnt)
        alloc_policy[members] += alloc_members

    return alloc_policy


# -----------------------------------------------------------------------------
# Turn allocations into assignment indices
# -----------------------------------------------------------------------------

def create_assignments_from_alloc(alloc):
    """
    Repeat policy indices according to their allocated counts (returns 1-D index array).
    """
    alloc = np.asarray(alloc, dtype=int).reshape(-1)
    return np.repeat(np.arange(alloc.size), np.clip(alloc, 0, None))

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
    return np.array([boundary_probability(p, R, H) for p in all_policies])
