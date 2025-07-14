import numpy as np
from rashomon.hasse import policy_to_profile
from rashomon.aggregate import RAggregate
from rashomon.extract_pools import extract_pools, aggregate_pools
from rashomon.loss import compute_pool_means
from rashomon.metrics import make_predictions

from helpers_rps import find_top_k_policies


# basic RPS construction
def construct_RPS_basic(M, R, H, D1, y1, theta, reg=0.001, num_workers=1, verbose=False):
    """
    Constructs the Rashomon set and profile-wise partitions for a single dataset and loss threshold.

    This function directly wraps RAggregate: it runs the Rashomon enumeration with the
    supplied pooling and regularization parameters, and returns all feasible partitions at the specified threshold.

    Args:
        M (int): number of features
        R (list or array): number of levels for each feature
        H (int): max pool size
        D1 (np.ndarray): mapping of assigned policy indices for observations
        y1 (np.ndarray): observed outcomes for the assignments
        theta (float): loss threshold for feasible partitions
        reg (float, optional): regularization penalty (default 0.001)
        num_workers (int, optional): for parallel computation (default 1)
        verbose (bool, optional): print progress and diagnostic info (default False)

    Returns:
        R_set (list): all feasible Rashomon set partitions (global indexing)
        R_profiles (list): list of feasible poolings for each profile
    """
    R_set, R_profiles = RAggregate(M, R, H, D1, y1, theta, reg=reg, num_workers=num_workers, verbose=verbose)
    return R_set, R_profiles

def construct_RPS_adaptive(
    M, R, H, D_full, y_full, num_top, policies_profiles_masked, policies_ids_profiles,
    profiles, all_policies, top_k_indices,
    theta_init, reg=0.1, max_factor=2.0, step=5, verbose=True,
    adaptive=False, recovery_type="arm"
):
    """
    Adaptively searches for a Rashomon set large enough to include the oracle-best arm or profile,
    by increasing the loss threshold theta as needed.

    Returns the Rashomon set, per-profile poolings, the theta used, and traces of the search.
    Primarily for simulation/interpretation: remove top_k_indices for real applications.

    Args:
        (see function definition for full parameter list)

    Returns:
        R_set, R_profiles, theta, found, theta_trace, rps_size_trace
    """

    # TODO later will need to remove the top_k_indices argument to make this usable without oracle. Just for interpretation right now.

    # get starting threshold
    theta = theta_init
    max_theta = theta_init * max_factor
    found = False
    iteration = 0
    theta_trace = []
    rps_size_trace = []

    # True best policies/profiles
    if recovery_type == "profile":
        best_targets = set(policy_to_profile(all_policies[idx]) for idx in top_k_indices)
    else:
        best_targets = set(top_k_indices)

    from rashomon.loss import compute_policy_means
    policy_means = compute_policy_means(D_full, y_full, len(all_policies))


    # update theta until actual best arm or profile is included
    while not found and theta <= max_theta:
        if verbose and adaptive:
            print(f"Trying theta={theta:.6f}")
        R_set, R_profiles = RAggregate(
            M, R, H, D_full, y_full, theta, reg=reg, num_workers=1, verbose=verbose
        )
        theta_trace.append(theta)
        rps_size_trace.append(len(R_set))
        if len(R_set) == 0 and adaptive:
            theta += step
            iteration += 1
            continue

        # For each partition in the Rashomon set, check recovery
        for r_set in R_set:
            # Assign pools for each profile
            pi_policies_profiles_r = {}
            for k, profile in enumerate(profiles):
                if len(R_profiles[k]) == 0:
                    continue
                sigma_k = R_profiles[k].sigma[r_set[k]]
                if sigma_k is None:
                    n_policies_profile = len(policies_profiles_masked[k])
                    pi_policies_r_k = {i: 0 for i in range(n_policies_profile)}
                else:
                    _, pi_policies_r_k = extract_pools(
                        policies_profiles_masked[k], sigma_k)
                pi_policies_profiles_r[k] = pi_policies_r_k

            pi_pools_r, pi_policies_r = aggregate_pools(
                pi_policies_profiles_r, policies_ids_profiles)
            pool_means_r = compute_pool_means(policy_means, pi_pools_r)
            y_r_est = make_predictions(D_full, pi_policies_r, pool_means_r)
            best_pred_indices = find_top_k_policies(D_full, y_r_est, num_top)

            if recovery_type == "profile":
                # check if any predicted best matched the best
                found_profiles = set(policy_to_profile(all_policies[idx]) for idx in best_pred_indices)
                if best_targets & found_profiles:
                    found = True
                    if verbose:
                        print("----FOUND-----")
                        print("actual best profile(s): ", best_targets)
                        print("found profile(s): ", found_profiles)
                    break
            else:  # "arm"
                if best_targets & set(best_pred_indices):
                    found = True
                    if verbose:
                        print("----FOUND-----")
                        print("actual best arms: ", [int(x) for x in best_targets])
                        print("best predicted arms: ", [int(x) for x in set(best_pred_indices)])
                    break

        if not adaptive:
            break
        if not found:
            theta += step
            iteration += 1

    if verbose and not found:
        if adaptive: print(f"Warning: max_theta reached without finding a Rashomon partition covering the best {recovery_type}.")
        else: print("Warning: No Rashomon partition found with an oracle-best arm within threshold.")

    if verbose and found:
        print("Rashomon set contains a partition that recovers at least one oracle-best arm.")

    if adaptive and verbose:
        print(f"Final theta for RPS: {theta:.4f} (after {iteration} iterations)")

    return R_set, R_profiles, theta, found, theta_trace, rps_size_trace