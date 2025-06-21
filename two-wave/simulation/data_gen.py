import numpy as np

def get_beta_underlying_causal(policies, M, R, kind="poly", sigma=0.2):
    """
    Returns a vector beta of treatment effects for each policy in the lattice.
    The function used depends on the 'kind' argument.

    Args:
      policies (list of tuples): Each policy is a tuple of feature values.
      M (int): Number of features.
      R (int or list[int]): Number of levels per feature.
      kind (str): Which function to use for beta.
      sigma (float): Used for Gaussian-based functions.

    Returns:
      beta (np.ndarray): True outcomes for each policy.
    """

    # Make R an array of length M
    if isinstance(R, int):
        R_arr = np.full(M, R, dtype=float)
    else:
        R_arr = np.array(R, dtype=float)

    # Get normalized coordinates in [0,1]^M
    K = len(policies)
    X = np.zeros((K, M))
    for i, v in enumerate(policies):
        for m in range(M):
            X[i, m] = v[m] / (R_arr[m] - 1)

    # -------------- underlying causal functions --------------
    if kind == "poly":
        # beta(x) = sum of x_m squared over m
        beta = np.sum(X**2, axis=1)

    elif kind == "linear_only": # FOR CHECKING, NOT FOR VALIDITY
        # beta(x) = sum of a_m * x_m over m, with a_m = 1 by default
        a = np.ones(M)
        beta = X.dot(a)

    elif kind == "simple_interaction":
        # beta(x) = x_0 * x_1 (simple interaction between first two features)
        if M < 2:
            raise ValueError("simple_interaction requires at least 2 features")
        beta = X[:, 0] * X[:, 1]

    elif kind == "pairwise_sparse":
        # beta(x) = sum of a_m * x_m over m plus sum of b_ij * x_i * x_j for selected pairs
        # Here a_m = 1 for all m, and only pairs (0,1), (1,2), (2,3), ... with b_ij = 0.7
        a = np.ones(M)
        beta = X.dot(a)
        for i in range(M - 1):
            beta += 0.7 * X[:, i] * X[:, i+1]

    elif kind == "quadratic_form":
        # beta(x) = x^T Q x + c^T x
        # c_m = 1 for all m, Q has diagonal 1 and off-diagonal 0.5 for neighbors
        c = np.ones(M)
        beta = X.dot(c)
        squared = np.sum(X**2, axis=1)
        beta += squared
        for i in range(M - 1):
            beta += 1.0 * X[:, i] * X[:, i+1]

    elif kind == "gauss_sin":
        # beta(x) = exp(-||x-0.5||^2/(2*sigma^2)) * product of sin(pi * x_m) over m
        diff = X - 0.5
        sqnorm = np.sum(diff**2, axis=1)
        gauss = np.exp(-sqnorm / (2 * sigma**2))
        sin_prod = np.prod(np.sin(np.pi * X), axis=1)
        beta = gauss * sin_prod

    elif kind == "gauss":
        # beta(x) = exp(-||x-0.5||^2/(2*sigma^2))
        diff = X - 0.5
        sqnorm = np.sum(diff**2, axis=1)
        beta = np.exp(-sqnorm / (2 * sigma**2))

    elif kind == "sin_prod":
        # beta(x) = product of sin(pi * x_m) over m
        beta = np.prod(np.sin(np.pi * X), axis=1)

    elif kind == "rbf_mixture":
        # beta(x) = sum over k of w_k * exp(-||x-mu_k||^2/(2*rho^2))
        # Uses 3 centers: mu1 = 0.25, mu2 = 0.75, mu3 alternates 0.5 and 0.2, with weights 1.0, 0.8, 1.2
        rho = 0.2
        w = [1.0, 0.8, 1.2]
        mu1 = np.full(M, 0.25)
        mu2 = np.full(M, 0.75)
        mu3 = np.array([(0.5 if (m % 2 == 0) else 0.2) for m in range(M)])
        centers = [mu1, mu2, mu3]
        beta = np.zeros(K)
        for weight, mu in zip(w, centers):
            diff = X - mu
            sqnorm = np.sum(diff**2, axis=1)
            beta += weight * np.exp(-sqnorm / (2 * rho**2))

    elif kind == "neural_net_tanh":
        # beta(x) = sum over h of v_h * tanh(u_h^T x - b_h)
        # Two hidden units: unit1 has all ones, bias M/2, unit2 alternates sign, bias 0.3*M, weights 2.0 and 1.5
        sum_all = np.sum(X, axis=1)
        alt_sign = X * (((-1) ** np.arange(M))[None, :])
        sum_alt = np.sum(alt_sign, axis=1)
        h1 = np.tanh(sum_all - (M / 2.0))
        h2 = np.tanh(sum_alt - (0.3 * M))
        beta = 2.0 * h1 + 1.5 * h2

    elif kind == "poly_degree_3":
        # beta(x) = sum of x_m cubed over m plus 0.5 times sum over m<k of x_m squared times x_k
        cubic = np.sum(X**3, axis=1)
        cross = np.zeros(K)
        for i in range(M):
            for j in range(i+1, M):
                cross += 0.5 * (X[:, i]**2) * X[:, j]
        beta = cubic + cross

    else:
        raise ValueError(
            f"Unknown kind='{kind}'. Valid options:\n"
            "  'bump_sin', 'sin_prod', 'poly', 'gauss',\n"
            "  'linear_only', 'simple_interaction', 'pairwise_sparse', 'quadratic_form',\n"
            "  'rbf_mixture', 'neural_net_tanh', 'poly_degree_3'.")
    return beta

def generate_outcomes(D: np.ndarray,
                      beta: np.ndarray,
                      sigma_noise: float,
                      random_seed: int = None) -> np.ndarray:
    if random_seed is not None:
        np.random.seed(random_seed)
    noise = np.random.normal(loc=0.0, scale=sigma_noise, size=D.shape[0])
    y = beta[D] + noise
    return y


# ------ below is first-pass code to generate a true partition and beta with effects for each pool -----
# ------ not currently in use bc need a smooth function (and would need to workshop computation + heterogeneous R) -----

# def generate_true_partition(policies: list,
#                             R,
#                             p_cut: float = 0.5,
#                             random_seed: int = None) -> tuple[np.ndarray, dict, dict]:
#     """
#     Given a precomputed `policies` list (each an M‐tuple), sample a simple random partition sigma
#     by drawing each boundary sigma[i,j] ~ Bernoulli(p_cut). Return sigma plus the resulting pool mappings.
#
#     Args:
#       policies (list of tuples): All ∏R_i lattice nodes, length K.
#       R (int or list[int]): Levels per feature. If int, every feature has R levels;
#                             if list/array, R[i] is the number of levels for feature i.
#       p_cut (float): Probability to “keep” (pool) each boundary.
#       random_seed (int, optional): Seed to make sigma sampling reproducible.
#
#     Returns:
#       sigma (np.ndarray of shape (M, R_i - 1)): Binary matrix defining the partition.
#       pi_pools_true (dict[int, list[int]]): pool_id → list of policy‐indices.
#       pi_policies_true (dict[int, int]): policy_index → pool_id.
#     """
#     if random_seed is not None:
#         np.random.seed(random_seed)
#
#     # Infer M from the length of any policy tuple
#     M = len(policies[0])
#
#     # Normalize R to array of length M
#     if np.isscalar(R):
#         R_arr = np.array([R] * M, dtype=int)
#     else:
#         R_arr = np.array(R, dtype=int)
#         if R_arr.size != M:
#             raise ValueError(f"R must be int or length‐M list; got length {R_arr.size} for M={M}.")
#
#     # Sample sigma row‐by‐row via Bernoulli(p_cut)
#     sigma_rows = []
#     for i in range(M):
#         num_boundaries = R_arr[i] - 1
#         row = np.random.binomial(1, p_cut, size=(num_boundaries,)).astype(int)
#         sigma_rows.append(row)
#     sigma = np.vstack(sigma_rows)  # shape (M, R_i - 1)
#
#     # extract pools (and the function call also extracts the edges to compute pools+policies)
#     pi_pools_true, pi_policies_true = extract_pools(policies, sigma)
#     return sigma, pi_pools_true, pi_policies_true
# from rashomon.extract_pools import extract_pools


# def get_beta_piecewise(policies: list,
#                        sigma_true: np.ndarray,
#                        pi_pools_true: dict[int, list[int]],
#                        pi_policies_true: dict[int, int],
#                        mu0: float,
#                        tau: float,
#                        random_seed: int = None) -> tuple[np.ndarray, dict, dict]:
#     """
#     From a precomputed lattice `policies` and a binary partition sigma_true,
#     draw one Gaussian mean per pool and assign β[idx] = that pool’s mean.
#
#     Args:
#       policies (list of tuples): All ∏R_i lattice nodes, length K.
#       sigma_true (np.ndarray): Shape (M, R_i-1) binary partition matrix.
#       pi_pools_true (dict[int, list[int]]): pool_id → list of policy‐indices.
#       pi_policies_true (dict[int, int]): policy_index → pool_id.
#       mu0 (float): Prior mean for each pool.
#       tau (float): Prior std for pool‐to‐pool variation.
#       random_seed (int, optional): Seed for reproducible draws.
#
#     Returns:
#       beta (np.ndarray of shape (K,)): True effect β[v] for each policy index.
#     """
#     if random_seed is not None:
#         np.random.seed(random_seed)
#
#     # Draw one gamma_j ~ N(mu0, tau^2) for each true pool
#     unique_pools = set(pi_policies_true.values())
#     pool_gammas = {pid: float(np.random.normal(mu0, tau)) for pid in unique_pools}
#
#     # Assign β for each policy index
#     K = len(policies)
#     beta = np.zeros(K, dtype=float)
#     for idx, pid in pi_policies_true.items():
#         beta[idx] = pool_gammas[pid]
#
#     return beta