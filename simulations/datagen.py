import numpy as np

# causal function registry

def phi_linear(policy, alpha=1.0, beta=0.0):
    x = np.array(policy)
    return alpha * np.sum(x) + beta

def phi_quadratic(policy, w=None):
    x = np.array(policy)
    if w is None:
        w = np.ones_like(x)
    return np.sum(w * x**2)

def phi_basic(policy):
    x = np.array(policy)
    return 2 * x[0] + 0.5 * x[1]**2 + x[0]*x[2] + 0.2 * x[1] * x[2] + 0.5*x[1]*x[3]

def phi_5d(policy):
    x = np.array(policy)
    return 2 * x[0] + 0.5 * x[1]**2 + x[0]*x[2] + 0.2 * x[1] * x[2] + 0.5*x[1]*x[3] + 0.1 * x[4]**2 + x[4]*x[0]

def phi_graded_sum(policy, scale=1.0):
    """Smoothly increasing outcome, diminishing returns (e.g. logistic growth)."""
    x = np.array(policy)
    s = np.sum(x)
    return scale * (s / (1 + 0.5 * s))

def phi_linear_combo(policy, weights=None, bias=10.0):
    """
    Weighted additive function with optional bias.
    Produces moderate variation, generalizes to arbitrary dimensions.
    """
    x = np.array(policy)
    if weights is None:
        weights = np.linspace(1.0, 2.0, num=len(x))  # increasing weights
    return float(np.dot(x, weights)) + bias

def phi_graded_sum(policy, scale=20.0):
    """
    Smooth growth with diminishing returns.
    Suitable for moderate pooled variation across profiles.
    """
    x = np.array(policy)
    s = np.sum(x)
    return scale * s / (1 + 0.3 * s)

def phi_profile_bucket(policy, bins=None):
    """
    Discontinuous grouping based on profile structure.
    Groups are controlled by number of active features.
    """
    profile = (np.array(policy) > 0).astype(int)
    active = np.sum(profile)
    if bins is None:
        bins = {0: 10, 1: 30, 2: 60, 3: 90, 4: 100}
    return bins.get(active, 0)

def phi_blended(policy, w=None):
    """
    Blended function with both linear and nonlinear terms.
    Designed to be smooth with meaningful heterogeneity.
    """
    x = np.array(policy)
    if w is None:
        w = np.linspace(0.5, 1.5, len(x))
    lin = np.dot(x, w)
    quad = np.sum((x - 2) ** 2)
    return 80 - 5 * quad + lin  # centered around 80, penalizes being far from (2,2,...)

# --- Registry ---
PHI_REGISTRY = {
    "linear": phi_linear,
    "quadratic": phi_quadratic,
    "linear_combo": phi_linear_combo,
    "graded_sum": phi_graded_sum,
    "profile_bucket": phi_profile_bucket,
    "blended": phi_blended,
    "basic": phi_basic
}

def get_phi_function(name, **kwargs):
    base_fn = PHI_REGISTRY.get(name)
    if base_fn is None:
        raise ValueError(f"Unknown causal function: {name}")
    return lambda policy: base_fn(policy, **kwargs)



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