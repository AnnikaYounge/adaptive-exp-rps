import numpy as np
from rashomon.hasse import enumerate_profiles, policy_to_profile
from rashomon.aggregate import find_feasible_combinations
from rashomon.aggregate import (
    RAggregate_profile,
    subset_data,
    find_profile_lower_bound,
)
from rashomon import loss

def construct_RPS(policies, M, R, D, y, H, eps=0.05, lambda_r=0.01, verbose=False):
    N = len(D)
    # Build profiles and maps between policies
    profiles, profile_map = enumerate_profiles(M)
    profile_to_policies = {k: [] for k in range(len(profiles))}
    profile_to_indices = {k: [] for k in range(len(profiles))}
    for i, pol in enumerate(policies):
        pid = profile_map[policy_to_profile(pol)]
        profile_to_policies[pid].append(pol)
        profile_to_indices[pid].append(i)

    # Get just the profiles and profile_ids with data and track losses
    valid_pids = []
    lower_bounds = []
    for profile_id, profile in enumerate(profiles):
        Dk, yk = subset_data(D, y, profile_to_indices[profile_id]) # using rashomon.aggregate, get correct subset of data
        if Dk is None:
            continue
        mask = np.array(profile, dtype=bool)
        # corresponding policies for this profile id
        reduced_policies = [tuple(np.array(p)[mask]) for p in profile_to_policies[profile_id]]

        # get losses and track lower bounds
        pm = loss.compute_policy_means(Dk, yk, len(reduced_policies))
        profile_lb = find_profile_lower_bound(Dk, yk, pm)
        lower_bounds.append(profile_lb / N)
        valid_pids.append(profile_id)

    lower_bounds = np.array(lower_bounds)
    lower_bounds = np.array(lower_bounds)
    total_lb = lower_bounds.sum()

    # calculate rashomon threshold
    theta_global = total_lb * (1 + eps)  # get loss threshold in absolute reference to sum of lower bounds
    if verbose:
        print(f"theta_global = {theta_global:.5f} from sum of lower bounds {total_lb:.5f}")

    R_profiles = []
    nonempty_profile_ids = []
    for i, profile_id in enumerate(valid_pids):
        profile_mask = np.array(profiles[profile_id], dtype=bool)
        M_k = profile_mask.sum()

        # Compute reduced policies using only active features for this profile
        reduced_policies = [tuple(np.array(p)[profile_mask]) for p in profile_to_policies[profile_id]]

        # Compute number of levels for each local (profile) feature
        R_k = np.array([len(set([p[feat] for p in reduced_policies])) for feat in range(M_k)])

        # Remap each feature in reduced_policies to contiguous 0-based values
        reduced_policies_arr = np.array(reduced_policies)
        for j in range(reduced_policies_arr.shape[1]):
            _, reduced_policies_arr[:, j] = np.unique(reduced_policies_arr[:, j], return_inverse=True)
        reduced_policies = [tuple(row) for row in reduced_policies_arr]
        R_k = np.array([len(np.unique(reduced_policies_arr[:, j])) for j in range(M_k)])

        # Value-mapping-based subsetting and remapping
        # This gives Dk (local indices) and yk (outcomes)
        policy_indices_this_profile = profile_to_indices[profile_id]
        mask = np.isin(D, policy_indices_this_profile)
        Dk = D[mask]
        yk = y[mask]

        # Now remap Dk from global policy indices to local indices in reduced_policies
        Dk = np.asarray(Dk).reshape(-1)
        policy_map = {idx: j for j, idx in enumerate(policy_indices_this_profile)}
        assert all(ix in policy_map for ix in Dk), f"Found Dk values not in mapping for profile {profile_id}"
        Dk_local = np.vectorize(policy_map.get)(Dk)  # map to local indices, shape (n,)
        assert yk.shape[0] == Dk_local.shape[0], "Dk_local and yk must have the same length"

        # Need to have Dk as a 1D array for the loss functions
        # Compute policy means with local indices
        pm = loss.compute_policy_means(Dk_local, yk, len(reduced_policies))
        assert pm.shape[0] == len(reduced_policies), "policy_means length mismatch"

        # get profile threshold
        theta_k = max(0.0, theta_global - (total_lb - lower_bounds[i]))

        # Need to reshape np array because the RAggregate_profile expects shape (n,1)
        Dk_local = Dk_local.reshape(-1, 1)
        yk = yk.reshape(-1, 1)
        # get rashomon set for each profile
        rashomon_profile = RAggregate_profile(
            M=M_k,
            R=R_k,
            H=H,
            D=Dk_local,  # Already local indices
            y=yk,
            theta=theta_k,
            profile=tuple(profiles[profile_id]),
            reg=lambda_r,
            policies=reduced_policies,
            policy_means=pm,
            normalize=N
        )

        # calculate losses for non-empty profiles and add to list of profiles
        Dk = np.asarray(Dk).reshape(-1)  # loss functions again want a 1d array for D, but keep y 2d
        if len(rashomon_profile) > 0:
            rashomon_profile.calculate_loss(Dk_local, yk, reduced_policies, pm, lambda_r, normalize=N)
            R_profiles.append(rashomon_profile)
            nonempty_profile_ids.append(profile_id)
        if verbose:
            print(
                f"Profile {profile_id}: M_k={M_k}, #policies={len(reduced_policies)}, theta_k={theta_k:.5f}, RPS size={len(rashomon_profile)}")

    if len(R_profiles) == 0:
        if verbose:
            print("No profiles have feasible Rashomon sets; global RPS is empty.")

    excluded_profiles = [profile_id for profile_id in valid_pids if profile_id not in nonempty_profile_ids]
    if verbose:
        if len(excluded_profiles) > 0:
            print(f"Skipped profile number due to empty Rashomon set: {excluded_profiles}")
        else:
            print("All profiles with data have non-empty Rashomon sets.")

    for idx, rp in enumerate(R_profiles):
        losses = np.array(rp.loss)

    R_set = find_feasible_combinations(R_profiles, theta_global, H)

    return R_set, R_profiles, theta_global, policies, profiles, profile_to_policies, profile_to_indices, nonempty_profile_ids