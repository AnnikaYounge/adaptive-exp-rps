import numpy as np

def compute_boundary_probabilities(lattice: np.ndarray, R: int, H: int) -> np.ndarray:
    """
    Compute P(v ∈ ∂Π | H) for each v in the lattice.

    Parameters:
    - lattice: ndarray of shape (R^M, M), treatment vectors with entries in {0, ..., R-1}
    - R: number of levels per dimension
    - H: number of partition parts

    Returns:
    - boundary_probs: ndarray of shape (R^M,), one value per treatment node
    """
    min_terms = np.minimum(lattice, R - 1 - lattice)  # shape (R^M, M)
    frac_terms = 2 * min_terms / (R - 1)              # inside the product
    # Eq: 1 - Π_i [1 - 2 * min(v_i, R-1-v_i)/(R-1)]^{H-1}
    product_term = np.prod(1 - frac_terms, axis=1)
    boundary_probs = 1 - product_term**(H - 1)
    return boundary_probs



def allocate_first_wave(n1: int, boundary_probs: np.ndarray) -> np.ndarray:
    """
    Deterministic allocation: n1(v) ∝ P(v ∈ ∂Π | H)

    Parameters:
    - n1: total number of first-wave samples
    - boundary_probs: unnormalized boundary scores

    Returns:
    - allocation_counts: array of shape (R^M,) with integer sample counts summing to n1
    """
    # Normalize probabilities to sum to 1
    p = boundary_probs / boundary_probs.sum()

    # Compute fractional allocation
    allocation = n1 * p

    # Floor to get initial integer counts
    floored = np.floor(allocation).astype(int)

    # Compute how many samples remain to assign due to flooring
    remainder = n1 - np.sum(floored)

    # Compute fractional parts and assign the remainder to those with largest residuals
    residuals = allocation - floored
    top_up_indices = np.argsort(residuals)[-remainder:]
    floored[top_up_indices] += 1

    return floored