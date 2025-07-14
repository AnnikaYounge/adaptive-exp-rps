import numpy as np

def phi(policy):
    """
    Returns the ground truth expected outcome for a given policy.
    Edit this function to specify the true underlying causal model.

    Args:
        policy (tuple or array-like): Policy/action vector.

    Returns:
        float: Expected outcome under this policy.
    """
    x = np.array(policy)
    return 2 * x[0] + 0.5 * x[1]**2 + x[0]*x[2] + 0.2 * x[1] * x[2]


def generate_data_from_assignments(D, all_policies, oracle_outcomes, sig=1.0):
    """
    Generates data from oracle outcomes by adding Gaussian noise.

    Args:
        D (np.ndarray): (n, 1) array of assigned policy indices (0 to num_policies-1)
        all_policies (list): List of all policy tuples (in global index order)
        oracle_outcomes (np.ndarray): Vector of oracle outcomes, indexed by policy ID
        sig (float): Standard deviation of outcome noise

    Returns:
        X (np.ndarray): (n, M) array of policy feature vectors
        y (np.ndarray): (n, 1) array of noisy outcomes
    """
    n = D.shape[0]
    M = len(all_policies[0])
    X = np.zeros((n, M))
    y = np.zeros((n, 1))
    for i in range(n):
        policy_idx = D[i, 0]
        policy = all_policies[policy_idx]
        X[i, :] = policy
        mu = oracle_outcomes[policy_idx]
        y[i, 0] = np.random.normal(loc=mu, scale=sig)
    return X, y