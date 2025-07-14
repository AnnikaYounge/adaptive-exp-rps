import numpy as np
from rashomon.aggregate import subset_data, find_profile_lower_bound
from rashomon.loss import compute_policy_means, compute_Q
from rashomon.hasse import policy_to_profile

def subset_wave_data_by_profile(D1, y1, policies_ids_profiles):
    """
    For each profile, subset the data to units assigned to that profile and remap global policy indices to profile-local indices.

    Returns:
        D1_profiles: dict (profile idx -> policy indices, profile-local, shape (n_k,1))
        y1_profiles: dict (profile idx -> outcomes, shape (n_k,1))
        global_to_local: dict (profile idx -> dict mapping global policy idx to profile-local idx)
    """
    D1_profiles = {}
    y1_profiles = {}
    global_to_local = {}
    for k, idxs in policies_ids_profiles.items():
        # Build set for fast membership
        idxs_set = set(idxs)
        # Mask: which units in D1 assigned to this profile?
        mask = np.array([d[0] in idxs_set for d in D1])
        if not np.any(mask):
            continue  # no data for this profile TODO confirm masking for this type - if not same masking
        D1_k = D1[mask].flatten()
        y1_k = y1[mask]
        # Remap global indices to profile-local (as expected by RAggregate)
        glob2loc = {glob: loc for loc, glob in enumerate(idxs)} # TODO confirm local to global
        D1_k_profile = np.array([glob2loc[d] for d in D1_k]).reshape(-1, 1)
        D1_profiles[k] = D1_k_profile
        y1_profiles[k] = y1_k
        global_to_local[k] = glob2loc
    return D1_profiles, y1_profiles, global_to_local # can work through and remove last param


def compute_profile_policy_outcomes(D1_profiles, y1_profiles, policies_profiles):
    """
    For each profile, tabulates the sum of observed outcomes and count of observations for every policy in that profile.

    Returns:
        dict: profile idx -> array of shape (num_policies_in_profile, 2), columns: [sum(y), count]
        (If a policy has not been observed, both entries are zero.)
    """
    profile_policy_outcomes = {}
    for k in D1_profiles.keys():
        n_policies = len(policies_profiles[k])
        pm = compute_policy_means(D1_profiles[k], y1_profiles[k], n_policies)
        profile_policy_outcomes[k] = pm
    return profile_policy_outcomes


def build_global_wave_data(D1_profiles, y1_profiles, policies_ids_profiles):
    """
    Concatenates all observed data across profiles,
    (!!!) converts profile-local policy indices back to global indices.
    This prepares the data for global Rashomon set construction.

    Returns:
        D1_full (np.ndarray): all observed units' global policy indices
        y1_full (np.ndarray): all observed outcomes
    """
    # TODO make sure local to global is applied everywhere
    D_list, y_list = [], []
    for k in D1_profiles.keys():
        idxs = policies_ids_profiles[k]
        D1_k_profile = D1_profiles[k].flatten()
        # Map each local index back to its global policy index for this profile
        D1_k_global = np.array([idxs[local_idx] for local_idx in D1_k_profile]).reshape(-1, 1)
        D_list.append(D1_k_global)
        y_list.append(y1_profiles[k])
    D1_full = np.vstack(D_list)
    y1_full = np.vstack(y_list)
    return D1_full, y1_full

# get top k policies for adaptive work
def find_top_k_policies(D, y_pred, k):
    """
    Returns the top-k unique policy indices (from D) ranked by descending y_pred.
    If multiple units are assigned to the same policy, only the first occurrence is kept.

    Returns:
        np.ndarray: array of k policy indices
    """
    idx_sorted = np.argsort(-y_pred.flatten())
    D_flat = D.flatten()[idx_sorted]
    _, unique_idx = np.unique(D_flat, return_index=True)
    top_k = D_flat[np.sort(unique_idx)][:k]
    return top_k

def get_observed_profiles(D, all_policies):
    """
    Given observed assignments, returns a mapping of profiles to their observed policy indices.

    Args:
        D (np.array): Assigned global policy indices (n, 1 or n,)
        all_policies (list): List of all policy tuples (global order)

    Returns:
        observed_policies_per_profile (dict): profile tuple -> set of observed global policy indices
        observed_profiles (set): Set of observed profile tuples
    """
    observed_policy_indices = set(D.flatten())
    observed_policies_per_profile = {}
    observed_profiles = set()
    for idx in observed_policy_indices:
        profile = policy_to_profile(all_policies[idx])
        observed_profiles.add(profile)
        if profile not in observed_policies_per_profile:
            observed_policies_per_profile[profile] = set()
        observed_policies_per_profile[profile].add(idx)
    return observed_policies_per_profile, observed_profiles

# TODO REMOVE
# def get_profiles_lower_bounds(D, y, profiles, policies_ids_profiles, policies_profiles, epsilon=0.05):
#     """
#     Computes per-profile lower bounds using loss-consistent logic from RAggregate:
#     - Uses no-pooling assumption (each policy is its own pool)
#     - Remaps global policy indices to local indices within each profile
#     - Normalizes by global sample size (to match RAggregate scaling)
#
#     Returns:
#         profile_lower_bounds (list[float])
#         theta (float)
#     """
#     profile_lower_bounds = []
#     total_n = D.shape[0]
#
#     for k, profile in enumerate(profiles):
#         D_k, y_k = subset_data(D, y, policies_ids_profiles[k])
#         if D_k is None or len(D_k) == 0:
#             profile_lower_bounds.append(0.0)
#             continue
#
#         D_k = np.asarray(D_k).reshape(-1)
#         y_k = np.asarray(y_k).reshape(-1, 1)
#
#         policy_ids = set(policies_ids_profiles[k])
#         mask = np.isin(D_k, list(policy_ids))
#         D_k = D_k[mask]
#         y_k = y_k[mask]
#
#         if len(D_k) == 0:
#             profile_lower_bounds.append(0.0)
#             continue
#
#         policy_map = {pid: j for j, pid in enumerate(policies_ids_profiles[k])}
#         D_k_local = np.vectorize(policy_map.get)(D_k).astype(int).reshape(-1, 1)
#
#         # Compute no-pooling means and loss
#         pm_k = compute_policy_means(D_k_local, y_k, len(policies_ids_profiles[k]))
#         lb_k = find_profile_lower_bound(D_k_local, y_k, pm_k)
#
#         profile_lower_bounds.append(lb_k / total_n)
#
#     theta = sum(profile_lower_bounds) * (1 + epsilon)
#     return profile_lower_bounds, theta