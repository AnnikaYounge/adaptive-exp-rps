import numpy as np
from rashomon.extract_pools import extract_pools, aggregate_pools
from rashomon.loss import compute_policy_means, compute_pool_means


def get_partition_losses(R_set, R_profiles):
    """
    Compute the total loss and normalized posterior weight for each Rashomon partition.

    Returns:
        partition_losses (np.ndarray): sum of per-profile losses for each partition
        weights (np.ndarray): posterior probability weights (exp(-loss), normalized)
    """
    partition_losses = np.array([sum(R_profiles[k].loss[r[k]] for k in range(len(r))) for r in R_set])
    weights = np.exp(-partition_losses)
    weights /= weights.sum() if weights.sum() > 0 else 1.0

    return partition_losses, weights

# TODO go through old sims and update with new functions
def summarize_rps_evaluation(
    R_set, R_profiles,
    D_full, y_full,
    policies_profiles_masked, policies_ids_profiles, profiles,
    oracle_outcomes, top_k_indices,
    all_policies
):
    """
    Summarizes posterior-weighted RPS evaluation information.

    Returns:
        dict with full evaluation payload (MAP info, regret, posterior structure, etc.)
    """


    regrets = []
    contains_best = []
    best_pred_indices_all = []
    policy_indices_all = []
    policy_means_all = []

    # Policy means from all data
    policy_means_total = compute_policy_means(D_full, y_full, len(all_policies))

    # Evaluate each partition
    for r_set in R_set:
        pi_policies_profiles_r = {}
        for k, profile in enumerate(profiles):
            if len(R_profiles[k]) == 0:
                continue
            sigma_k = R_profiles[k].sigma[r_set[k]]
            if sigma_k is None:
                n_policies_profile = len(policies_profiles_masked[k])
                pi_policies_r_k = {i: 0 for i in range(n_policies_profile)}
            else:
                _, pi_policies_r_k = extract_pools(policies_profiles_masked[k], sigma_k)
            pi_policies_profiles_r[k] = pi_policies_r_k

        pi_pools_r, pi_policies_r = aggregate_pools(pi_policies_profiles_r, policies_ids_profiles)
        pool_means_r = compute_pool_means(policy_means_total, pi_pools_r)

        policy_indices = np.array(list(pi_policies_r.keys()))
        policy_means = np.array([pool_means_r[pi_policies_r[idx]] for idx in policy_indices])

        policy_indices_all.append(policy_indices)
        policy_means_all.append(policy_means)

        best_pred_idx = policy_indices[np.argmax(policy_means)]
        regret = float(oracle_outcomes[top_k_indices[0]] - pool_means_r[pi_policies_r[best_pred_idx]])

        regrets.append(regret)
        contains_best.append(int(best_pred_idx == top_k_indices[0]))
        best_pred_indices_all.append(best_pred_idx)

    # Compute losses and posterior weights
    partition_losses, posterior_weights = get_partition_losses(R_set, R_profiles)
    map_idx = np.argmin(partition_losses)

    return {
        "partition_losses": partition_losses,
        "posterior_weights": posterior_weights,
        "regrets": np.array(regrets),
        "contains_best": np.array(contains_best),
        "best_pred_indices_all": best_pred_indices_all,
        "policy_indices_all": policy_indices_all,
        "policy_means_all": policy_means_all,
        "map_idx": map_idx,
        "map_policy_indices": policy_indices_all[map_idx],
        "map_policy_means": policy_means_all[map_idx]
    }


def get_partition_policy_summary(
    policy_indices_all, policy_means_all, partition_idx,
    oracle_outcomes, oracle_policy_to_rank, top_k_indices, n_predicted
):
    """
    Summarizes the top-N predicted policies (by MAP) from a given Rashomon partition.

    Args:
        policy_indices_all (list[np.ndarray]): List of arrays of policy indices for each partition.
        policy_means_all (list[np.ndarray]): Corresponding predicted means.
        partition_idx (int): Index of partition to summarize (e.g. map_idx).
        oracle_outcomes (np.ndarray): Ground truth outcomes per policy index.
        oracle_policy_to_rank (np.ndarray): Maps policy index to oracle rank (1 = best).
        top_k_indices (list[int]): Indices of oracle top-k policies.
        n_predicted (int): Number of top predicted policies to include in summary.

    Returns:
        dict: {
            'sorted_idx': top-N predicted policy indices,
            'sorted_means': predicted means for those,
            'oracle_values': true outcomes,
            'oracle_ranks': ranks by oracle,
            'is_topk': list of booleans (True if in oracle top-k)
        }
    """
    policy_indices = policy_indices_all[partition_idx]
    policy_means = policy_means_all[partition_idx]
    order = np.argsort(-policy_means)

    sorted_idx = policy_indices[order][:n_predicted]
    sorted_means = policy_means[order][:n_predicted]
    oracle_values = oracle_outcomes[sorted_idx]
    oracle_ranks = oracle_policy_to_rank[sorted_idx]
    is_topk = [i in top_k_indices for i in sorted_idx]

    return {
        "sorted_idx": sorted_idx,
        "sorted_means": sorted_means,
        "oracle_values": oracle_values,
        "oracle_ranks": oracle_ranks,
        "is_topk": is_topk,
    }