import numpy as np

def boundary_prob(v, R, H: int) -> float:
    """
    Compute P(on boundary | H) for a given policy vector v.

    Args:
        v (tuple or np.ndarray): length‐M tuple (or array) of ints in [0, R_i-1]
        R (int or list/array): number of levels per feature; if int, same R for all M
        H (int): sparsity parameter

    Returns:
        float: boundary probability for node v
    """
    # Ensure R_arr is an array of length M
    M = len(v)
    if np.isscalar(R):
        R_arr = np.array([R] * M, dtype=int)
    else:
        R_arr = np.array(R, dtype=int)
        if R_arr.size != M:
            raise ValueError(f"R must be an int or list/array of length M={M}, instead got size {R_arr.size}.")
    # from formula in paper
    term = 1.0
    for i in range(M):
        R_i = R_arr[i] # TODO check if need type casting for int64
        ratio = 2 * min(int(v[i]), R_i - 1 - int(v[i])) / (R_i - 1)
        term *= (1 - ratio) ** (H - 1)
    return 1 - term


def compute_boundary_probs(policies: list, R, H: int) -> np.ndarray:
    """
    Compute boundary probabilities for each policy

    Args:
        policies (list): list (length K) of tuples (length M)
        R (int or list/array): number of levels per feature; if int, same R for all M
        H (int): sparsity parameter

    Returns:
        np.ndarray: length‐K array of probabilities for each policy index v
    """
    K = len(policies)
    probs = np.zeros(K, dtype=float)
    for idx in range(K):
        v = policies[idx]
        probs[idx] = boundary_prob(v, R, H)
    return probs


def allocate_first_wave(boundary_probs: np.ndarray, n1: int) -> np.ndarray:
    """
    Allocate the first‐wave sample counts n1(v) across all policies

    Args:
        boundary_probs (np.ndarray): length‐K array of boundary probabilities
        n1 (int): total first‐wave sample size

    Returns:
        np.ndarray: length‐K array of integer allocation n₁(v)
    """
    # normalize to get weights, keep proportional to boundary probabilities
    total_prob = boundary_probs.sum()
    if total_prob == 0:
        raise ValueError("Sum of boundary probabilities is zero. Check H or R.")
    normalized = boundary_probs / total_prob

    # get direct allocation (could be fractional)
    dir_alloc = normalized * n1

    # floor to integers
    n1_alloc = np.floor(dir_alloc).astype(int)
    remainder = dir_alloc - n1_alloc
    shortage = n1 - int(n1_alloc.sum())

    # distribute the remaining samples to the largest remainders
    if shortage > 0:
        idx_sorted = np.argsort(remainder)
        top_indices = idx_sorted[-shortage:]
        n1_alloc[top_indices] += 1

    return n1_alloc


def assign_first_wave_treatments(n_alloc: np.ndarray) -> np.ndarray:
    """
    Given an allocation array n_alloc of length K (n_alloc[i] = # draws at policy i),
    create a np.array D of policy‐indices

    Args:
      n_alloc (np.ndarray): Integer array of length K summing to n_total.

    Returns:
      D (np.ndarray): Each policy index repeated n_alloc[i] times.
      TODO check if speed improvements and if need type casting
    """
    total = int(n_alloc.sum())
    D = np.zeros(total, dtype=int)
    pos = 0
    for idx, count in enumerate(n_alloc):
        if count > 0:
            D[pos : pos + count] = idx
            pos += count
    return D