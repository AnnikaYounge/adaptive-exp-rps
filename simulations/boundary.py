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