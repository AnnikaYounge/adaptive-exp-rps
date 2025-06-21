import numpy as np
from itertools import product
from rashomon.aggregate import RAggregate_profile, find_profile_lower_bound
from rashomon.hasse import enumerate_profiles, enumerate_policies, policy_to_profile
from rashomon import loss

def observed_rps(M, R_vec, H, D1, y1, lambda_r, eps=0.05, N=None):
    """
    Observed Rashomon Partition Set from assignments D1 and outcomes y1.
    """
    D = np.asarray(D1)
    y = np.asarray(y1)
    if D.ndim != 1:
        raise ValueError(f"D should be 1D (policy indices), got shape {D.shape}")
    if y.ndim != 1:
        y = y.ravel()
    N = len(D)
    if len(y) != N:
        raise ValueError(f"y and D must have same length: got {len(y)} and {N}")

    # Enumerate all policies and profiles
    profiles, profile_map = enumerate_profiles(M)
    all_policies = enumerate_policies(M, R_vec)
    policy_to_pid = {tuple(pol): profile_map[policy_to_profile(pol)] for pol in all_policies}
    policy_to_indices = {pid: [] for pid in range(len(profiles))}
    for i, pol in enumerate(all_policies):
        pid = policy_to_pid[tuple(pol)]
        policy_to_indices[pid].append(i)

    # Identify active profiles and compute lower bounds
    valid_pids = []
    lb_k = []
    for pid, profile in enumerate(profiles):
        indices = policy_to_indices[pid]
        idx_mask = np.isin(D, indices)
        if np.sum(idx_mask) > 0:
            Dk_policyidx = D[idx_mask] # Global policy indices for this profile
            yk = y[idx_mask]
            profile_mask = np.array(profile, dtype=bool)
            # Map to reduced policies (profile-local tuples)
            policies_k = [tuple(np.array(all_policies[pol_idx])[profile_mask]) for pol_idx in Dk_policyidx]
            policies_k_unique = list(sorted(set(policies_k)))
            # Build tuple->local index mapping for this profile
            tuple_to_local_idx = {p: i for i, p in enumerate(policies_k_unique)}
            Dk_local = np.array([tuple_to_local_idx[p] for p in policies_k])
            pm = loss.compute_policy_means(Dk_local, yk, len(policies_k_unique))
            raw_lb = find_profile_lower_bound(Dk_local, yk.reshape(-1, 1), pm)
            lb_k.append(raw_lb / N)
            valid_pids.append(pid)

        else:
            lb_k.append(0.0)

    lb_k_arr = np.array(lb_k)
    total_lb = lb_k_arr.sum()
    theta_global = total_lb * (1 + eps)

    # build per-profile RashomonSets (always using local indices!)
    R_profiles = []
    for i, pid in enumerate(valid_pids):
        indices = policy_to_indices[pid]
        idx_mask = np.isin(D, indices)
        Dk_policyidx = D[idx_mask]
        yk = y[idx_mask]
        profile_mask = np.array(profiles[pid], dtype=bool)
        M_k = profile_mask.sum()
        R_k = R_vec[profile_mask]
        policies_k = [tuple(np.array(all_policies[pol_idx])[profile_mask]) for pol_idx in Dk_policyidx]
        policies_k_unique = list(sorted(set(policies_k)))
        tuple_to_local_idx = {p: j for j, p in enumerate(policies_k_unique)}
        Dk_local = np.array([tuple_to_local_idx[p] for p in policies_k])
        pm = loss.compute_policy_means(Dk_local, yk, len(policies_k_unique))
        theta_k = max(0.0, theta_global - (total_lb - lb_k[pid]))
        print(f"Profile {pid}: M_k={M_k}, #policies={len(policies_k_unique)}, theta_k={theta_k:.5f}")
        rp = RAggregate_profile(
            M=M_k,
            R=R_k,
            H=H,
            D=Dk_local.reshape(-1, 1), # 1D array of local indices
            y=yk.reshape(-1, 1), # 1D array of outcomes
            theta=theta_k,
            profile=tuple(profiles[pid]),
            reg=lambda_r,
            policies=policies_k_unique,
            policy_means=pm,
            normalize=N
        )
        R_profiles.append(rp)

    nonempty_idx = [i for i, rp in enumerate(R_profiles) if len(rp) > 0]
    nonempty_profiles = [R_profiles[i] for i in nonempty_idx]
    R_set_partial = list(product(*[range(len(rp)) for rp in nonempty_profiles]))

    return R_set_partial, R_profiles, nonempty_idx, profiles