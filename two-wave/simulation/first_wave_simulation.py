import numpy as np

def first_wave_simulate_data(
    lattice: np.ndarray,
    beta: np.ndarray,
    allocation: np.ndarray,
    sigma: float,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate wave-1 outcome data given a lattice, treatment effects, and allocation.

    Parameters:
    - lattice: (K, M) array of treatment vectors (each row is a treatment v)
    - beta: (K,) array of true treatment effects β_v = phi(v)
    - allocation: (K,) array with number of units assigned to each treatment v
    - sigma: standard deviation of Gaussian outcome noise
    - seed: optional random seed for reproducibility

    Returns:
    - D: (n_1, K) design matrix, where each row is a one-hot vector encoding treatment assignment
    - y: (n_1,) outcome vector corresponding to D
    """
    K = len(beta) # total number of treatments
    rng = np.random.default_rng(seed)

    rows = [] # will store one-hot rows for each sample
    outcomes = [] # will store outcomes y_i

    for v_idx, count in enumerate(allocation):
        if count == 0:
            continue

        # Draw outcomes y_i = β_v + ε_i
        beta_v = beta[v_idx]
        noise = rng.normal(loc=0.0, scale=sigma, size=count)
        y_v = beta_v + noise

        # Construct one-hot encoding for treatment v_idx
        one_hot = np.zeros(K)
        one_hot[v_idx] = 1
        D_rows = np.tile(one_hot, (count, 1))

        # Store the design matrix rows and corresponding outcomes
        rows.append(D_rows)
        outcomes.append(y_v)

    # Concatenate to form a full design matrix and outcome vector
    D = np.vstack(rows)
    y = np.concatenate(outcomes)

    return D, y