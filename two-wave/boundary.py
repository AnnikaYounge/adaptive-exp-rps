import numpy as np
from rashomon.extract_pools import lattice_edges, extract_pools, aggregate_pools

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


def get_allocations(probs, n1):
    """
    Allocates a total sample size n1 across arms according to a probability vector,
    returning an integer allocation vector that sums exactly to n1.

    Args:
        probs (array): Probability weights for each arm/policy (length = num_policies).
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

def create_assignments_from_alloc(alloc):
    """
    Converts an integer allocation vector into an array of assigned policy indices.
    Each policy index is repeated according to its allocation count;
    the array D matches each observation to its assigned policy.

    Args:
        alloc (array): Integer allocation for each policy (length = num_policies).

    Returns:
        np.ndarray: (n1, 1) array of assigned policy indices, where n1 = alloc.sum().
    """
    D = []
    for i, count in enumerate(alloc):
        if count > 0:
            D.extend([i] * count)
    D = np.array(D, dtype=int).reshape(-1, 1)
    return D


def get_policy_neighbors(all_policies):
    """
    Builds a neighbor dictionary for each policy in the lattice, where neighbors
    are policies differing by one level (i.e., adjacent in the lattice graph).

    Args:
        all_policies (list of tuples): Each tuple is a policy (action vector).

    Returns:
        dict: Mapping from policy index to the policy indices of neighbors.
    """
    edges = lattice_edges(all_policies)  # list of (i, j)
    neighbors = {i: [] for i in range(len(all_policies))}
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)  # edges are undirected
    return neighbors

# Usage:
# neighbors = get_policy_neighbors_from_edges(all_policies)

def compute_global_boundary_matrix(
    R_set, R_profiles, neighbors, profiles, policies_profiles_masked, policies_ids_profiles, all_policies
):
    """
    Computes, for each Rashomon partition, the number of nearby boundaries (neighboring policies in a different pool)
    for every policy in the policy lattice. Returns a matrix of shape (n_partitions, num_policies).
    # TODO make it such that also a boundary if reach outside of given profile

    Args:
        R_set (list): List of Rashomon set partitions (poolings).
        R_profiles (list): Profile-wise Rashomon partitions.
        neighbors (dict): Neighbor map from policy index to list of neighbor indices.
        profiles (list): List of profiles.
        policies_profiles_masked (dict): Masked policy sets by profile.
        policies_ids_profiles (dict): Mapping from profile to global policy indices.
        all_policies (list): List of all policy tuples.

    Returns:
        np.ndarray: Boundary matrix of shape (n_partitions, num_policies),
            where entry (j, i) is the number of boundaries for policy i in partition j.
    """
    num_policies = len(all_policies)
    n_partitions = len(R_set)

    boundary_matrix = np.zeros((n_partitions, num_policies), dtype=int)

    for j, r in enumerate(R_set):
        pi_policies_profiles_r = {}
        for k, profile in enumerate(profiles):
            if len(R_profiles[k]) == 0:
                continue
            sigma_k = R_profiles[k].sigma[r[k]]
            if sigma_k is None:
                continue
            _, pi_policies_k = extract_pools(policies_profiles_masked[k], sigma_k)
            pi_policies_profiles_r[k] = pi_policies_k
        pi_pools_r, pi_policies_r = aggregate_pools(pi_policies_profiles_r, policies_ids_profiles)
        for i in range(num_policies):
            pool_i = pi_policies_r.get(i, -1)
            if pool_i == -1:
                continue
            # Count number of neighbors in a different pool
            count = 0
            for nb in neighbors[i]:
                pool_nb = pi_policies_r.get(nb, -1)
                if pool_nb != -1 and pool_nb != pool_i:
                    count += 1
            boundary_matrix[j, i] = count
    return boundary_matrix