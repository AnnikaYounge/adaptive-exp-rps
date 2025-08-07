import numpy as np

# causal function registry

def phi_basic(policy):
    x = np.array(policy)
    return 2 * x[0] + 0.5 * x[1]**2 + x[0]*x[2] + 0.2 * x[1] * x[2] + 0.5*x[1]*x[3]

def phi_linear_interact(policy, R, shift=0.0):
    x = np.array(policy) / (np.array(R) - 1)
    lin = np.dot(x, np.linspace(1, 2, len(x)))  # increasing weights
    pairwise = sum(x[i]*x[i+1] for i in range(len(x)-1))
    return lin + 0.3 * pairwise + shift

def phi_peak(policy, R, center=None, w_scale=1.0):
    x = np.array(policy) / (np.array(R) - 1)
    M = len(x)
    if center is None:
        center = np.ones(M) * 0.5  # center in [0,1]^M
    w = np.linspace(1, 2, M) * w_scale
    value = -np.sum(w * (x - center)**2)  # peak at center
    return float(value)

def phi_grouped_smooth2(policy, R):
    x = np.array(policy) / (np.array(R) - 1)
    centers = [np.full_like(x, val) for val in [0.2, 0.5, 0.8]]
    weights = [1.0, 3.0, 5.0]
    sharpness = 0.5

    group_vals = [
        w * np.exp(-np.sum((x - c)**2) / sharpness)
        for c, w in zip(centers, weights)
    ]
    lin = np.dot(x, np.linspace(0.2, 1.0, len(x)))

    return sum(group_vals) + 0.3 * lin

def phi_grouped_coarse(policy, R, freq=2.0, sharpness=0.15):
    x = np.array(policy) / (np.array(R) - 1)
    x_mean = np.mean(x)
    x_sum = np.sum(x)

    # Broad sinusoidal ridges across average and total activation
    ridge = np.sin(freq * np.pi * x_mean) + np.cos(freq * np.pi * x_sum / len(x))

    # Smooth global interaction bump (moderately centered)
    bump = np.exp(-np.sum((x - 0.5)**2) / sharpness)

    return 2.0 + ridge + 2.0 * bump

def phi_grouped_smooth(policy, R, shift=0.0):
    x = np.array(policy) / (np.array(R) - 1)  # normalize to [0, 1]
    centers = [
        np.full_like(x, 0.2),
        np.full_like(x, 0.5),
        np.full_like(x, 0.8),
    ]
    values = [np.exp(-np.sum((x - c)**2) / 0.05) for c in centers]
    interact = sum(x[i]*x[i+1] for i in range(len(x)-1))
    return 1.0 + 2.0 * max(values) + 0.3 * interact + shift

def phi_5d(policy):
    x = np.array(policy)
    return 2 * x[0] + 0.5 * x[1]**2 + x[0]*x[2] + 0.2 * x[1] * x[2] + 0.5*x[1]*x[3] + 0.1 * x[4]**2 + x[4]*x[0]


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