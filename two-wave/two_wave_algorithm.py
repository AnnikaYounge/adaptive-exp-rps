import numpy as np
from rashomon.hasse import policy_to_profile, enumerate_policies, enumerate_profiles
from datagen import generate_data_from_assignments
from boundary import (
    compute_boundary_probabilities, get_allocations,
    create_assignments_from_alloc, get_policy_neighbors, compute_global_boundary_matrix
)
from helpers_rps import (
    subset_wave_data_by_profile, compute_profile_policy_outcomes,
    build_global_wave_data
)
from enumerate_rps import construct_RPS_adaptive
from evaluation import get_partition_losses

def two_wave_algorithm(
    M, R, theta_start, lambda_reg, epsilon, n,
    oracle_outcomes, top_k_indices,
    sig=1.0, verbose_algo=True, verbose_rps=True
):
    """
    Runs a full two-wave adaptive Rashomon Partition Set (RPS) algorithm:
    1. Wave 1 allocation using theoretical boundary probabilities
    2. RPS construction and posterior boundary computation
    3. Wave 2 allocation guided by RPS posterior
    4. Final RPS re-enumeration using all data
    # TODO change docstring

    Returns:
        Dictionary containing data, RPS sets, posterior weights, and losses
    """

    # split number of observations
    # TODO change so unpacks observations / create a multi wave version
    n1 = n[0]
    n2 = n[1]

    # enumerate all policies and profiles
    all_policies = enumerate_policies(M, R)
    profiles, profile_map = enumerate_profiles(M)

    # profile index mappings
    policies_profiles = {}
    policies_ids_profiles = {}
    for k, profile in enumerate(profiles):
        policy_indices = [i for i, p in enumerate(all_policies) if policy_to_profile(p) == profile]
        policies_ids_profiles[k] = policy_indices
        policies_profiles[k] = [all_policies[i] for i in policy_indices]

    # max pool size checks for sparsity
    max_pool_size = max(len(policies) for policies in policies_profiles.values())
    H = max_pool_size

    policies_profiles_masked = {}
    for k, profile in enumerate(profiles):
        profile_mask = [bool(v) for v in profile]  # t/f map of which features are active
        masked_policies = [tuple([pol[i] for i in range(M) if profile_mask[i]]) for pol in
                           policies_profiles[k]]
        policies_profiles_masked[k] = masked_policies

    # === Wave 1 ===
    boundary_probs = compute_boundary_probabilities(all_policies, R, H)
    alloc1 = get_allocations(boundary_probs, n1)
    D1 = create_assignments_from_alloc(alloc1)
    X1, y1 = generate_data_from_assignments(D1, all_policies, oracle_outcomes, sig=sig)

    if verbose_algo:
        print("Starting Wave 1")
        print("--- Mean and std. dev of first wave outcomes:", np.mean(y1), np.std(y1))

    D1_profiles, y1_profiles, _ = subset_wave_data_by_profile(D1, y1, policies_ids_profiles)
    D1_full, y1_full = build_global_wave_data(D1_profiles, y1_profiles, policies_ids_profiles)

    # TODO will need to remove oracle part for later use
    R_set, R_profiles, theta_final, found_best, theta_trace, rps_size_trace = construct_RPS_adaptive(
        M, R, H, D1_full, y1_full, len(top_k_indices), policies_profiles_masked,
        policies_ids_profiles, profiles, all_policies, top_k_indices, theta_start,
        reg=lambda_reg,
        adaptive=False,
        verbose=verbose_rps,
        recovery_type="arm"
    )
    if verbose_algo:
        print(f"--- First-wave Rashomon set: {len(R_set)} feasible global partitions (combinations of per-profile poolings).")
    if verbose_rps:
        for k, rprof in enumerate(R_profiles):
            print(f"------ Profile {k}: {len(rprof)} poolings in RPS (if observed)")

    neighbors = get_policy_neighbors(all_policies)
    boundary_matrix_1 = compute_global_boundary_matrix(
        R_set, R_profiles, neighbors, profiles,
        policies_profiles_masked, policies_ids_profiles, all_policies
    )
    binary_boundary_matrix_1 = (boundary_matrix_1 > 0).astype(float)
    partition_losses_1, posterior_weights_1 = get_partition_losses(R_set, R_profiles)

    # === Wave 2 ===
    if verbose_algo:
        print("Starting Wave 2")
    posterior_boundary_probs_1 = np.average(binary_boundary_matrix_1, axis=0, weights=posterior_weights_1)
    posterior_boundary_probs_1 = np.round(posterior_boundary_probs_1, decimals=8)
    alloc2 = get_allocations(posterior_boundary_probs_1, n2)
    D2 = create_assignments_from_alloc(alloc2)
    X2, y2 = generate_data_from_assignments(D2, all_policies, oracle_outcomes, sig=sig)

    if verbose_algo:
        print("--- Mean and std. dev of first wave outcomes:", np.mean(y2), np.std(y2))

    D_total = np.vstack([D1, D2])
    y_total = np.vstack([y1, y2])
    D_total_profiles, y_total_profiles, _ = subset_wave_data_by_profile(D_total, y_total, policies_ids_profiles)
    D_total_full, y_total_full = build_global_wave_data(D_total_profiles, y_total_profiles, policies_ids_profiles)

    R_set_2, R_profiles_2, theta_final_2, found_best_2, theta_trace_2, rps_size_trace_2 = construct_RPS_adaptive(
        M, R, H, D_total_full, y_total_full, len(top_k_indices), policies_profiles_masked,
        policies_ids_profiles, profiles, all_policies, top_k_indices, theta_final,
        reg=lambda_reg,
        adaptive=False,
        verbose=verbose_rps,
        recovery_type="arm"
    )
    if verbose_algo:
        print(f"--- Second-wave Rashomon set: {len(R_set)} feasible global partitions (combinations of per-profile poolings).")
    if verbose_rps:
        for k, rprof in enumerate(R_profiles):
            print(f"------ Profile {k}: {len(rprof)} poolings in RPS (if observed)")

    partition_losses_2, posterior_weights_2 = get_partition_losses(R_set_2, R_profiles_2)

    return {
        "D1": D1, "y1": y1, "D2": D2, "y2": y2,
        "D_total": D_total, "y_total": y_total,
        "R_set_1": R_set,
        "R_profiles_1": R_profiles,
        "posterior_boundary_probs_1": posterior_boundary_probs_1,
        "posterior_weights_1": posterior_weights_1,
        "R_set_2": R_set_2,
        "R_profiles_2": R_profiles_2,
        "posterior_weights_2": posterior_weights_2,
        "partition_losses_2": partition_losses_2,
        "theta_final_2": theta_final_2
    }