import numpy as np

def boundary_prob_start(v, R, H: int) -> float:
    # Ensure R_arr is an array of length M
    M = len(v)
    if np.isscalar(R):
        R_arr = np.array([R] * M, dtype=int)
    else:
        R_arr = np.array(R, dtype=int)
        if R_arr.size != M:
            raise ValueError(f"R must be an int or list/array of length M={M}, instead got size {R_arr.size}.")
    # from formula in paper
    term = 1.0
    for i in range(M):
        R_i = R_arr[i]
        ratio = 2 * min(int(v[i]), R_i - 1 - int(v[i])) / (R_i - 1)
        term *= (1 - ratio) ** (H - 1)
    return 1 - term


def compute_initial_boundary_probs(policies: list, R, H: int) -> np.ndarray:
    K = len(policies)
    probs = np.zeros(K, dtype=float)
    for idx in range(K):
        v = policies[idx]
        probs[idx] = boundary_prob_start(v, R, H)
    return probs

from rashomon import extract_pools

def compute_wave_boundary_probs(R_set, R_profiles, policies, profiles, profile_to_policies, profile_to_indices, nonempty_profile_ids) -> np.ndarray:
    # precompute all lattice edges
    lattice_ed = extract_pools.lattice_edges(policies)  # policies holds the full enumerated lattice

    K = len(policies)
    boundary_counts = np.zeros(K, float)

    # compute posterior weights for each RPS partition
    Q = np.array([
        sum(R_profiles[k].loss[part[k]] for k in range(len(R_profiles)))
        for part in R_set
    ])
    post_weights = np.exp(-Q - Q.min())
    post_weights /= post_weights.sum()

    for part, w_i in zip(R_set, post_weights):
        pi_policies_profiles = {}
        for k, rp in enumerate(R_profiles):
            profile_id = nonempty_profile_ids[k]
            profile_mask = np.array(profiles[profile_id], dtype=bool)

            # Mask and remap as in RPS construction
            local_policies = [tuple(np.array(p)[profile_mask]) for p in profile_to_policies[profile_id]]
            if len(local_policies) == 0:
                continue  # skip empty
            arr = np.array(local_policies)
            for j in range(arr.shape[1]):
                _, arr[:, j] = np.unique(arr[:, j], return_inverse=True)
            local_policies_remap = [tuple(row) for row in arr]

            # Use the same remapped local_policies as when RPS was constructed
            sigma = rp.sigma[part[k]]
            _, pi_policies_local = extract_pools.extract_pools(local_policies_remap, sigma)

            # Map from local index to global index
            for local_idx, global_idx in enumerate(profile_to_indices[profile_id]):
                pi_policies_profiles[global_idx] = pi_policies_local[local_idx]

        # Build full K-vector: -1 for not-in-any-profile
        pi_policies = np.full(K, -1, dtype=int)
        for global_idx, pool_id in pi_policies_profiles.items():
            pi_policies[global_idx] = pool_id

        # Only count boundaries between *observed* nodes
        for u, v in lattice_ed:
            if pi_policies[u] != -1 and pi_policies[v] != -1 and pi_policies[u] != pi_policies[v]:
                boundary_counts[u] += w_i
                boundary_counts[v] += w_i

    if boundary_counts.sum() == 0:
        raise ValueError("No boundaries detected in second-wave allocation. Check RPS content.")

    boundary_probs_2 = boundary_counts / boundary_counts.sum()

    return boundary_probs_2


def allocate_wave(boundary_probs: np.ndarray, n1: int) -> np.ndarray:
    # normalize to get weights, keep proportional to boundary probabilities
    total_prob = boundary_probs.sum()
    if total_prob == 0:
        raise ValueError("Sum of boundary probabilities is zero. Check H or R.")
    normalized = boundary_probs / total_prob

    # get direct allocation (could be fractional)
    dir_alloc = normalized * n1

    # floor to integers
    n1_alloc = np.floor(dir_alloc).astype(int)
    remainder = dir_alloc - n1_alloc
    shortage = n1 - int(n1_alloc.sum())

    # distribute the remaining samples to the largest remainders
    if shortage > 0:
        idx_sorted = np.argsort(remainder)
        top_indices = idx_sorted[-shortage:]
        n1_alloc[top_indices] += 1

    return n1_alloc


def assign_treatments(n_alloc: np.ndarray) -> np.ndarray:
    total = int(n_alloc.sum())
    D = np.zeros(total, dtype=int)
    pos = 0
    for idx, count in enumerate(n_alloc):
        if count > 0:
            D[pos : pos + count] = idx
            pos += count
    return D