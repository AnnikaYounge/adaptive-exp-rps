import numpy as np
import itertools

def generate_lattice(R: int, M: int) -> np.ndarray:
    """
    Generate the full factorial lattice Λ = R^M.
    Each treatment vector is an M-dimensional vector with entries in {0, ..., R-1}.
    Returns an array of shape (R^M, M) – (each row is a treatment vector, columns for each feature)
    """
    grid = list(itertools.product(range(R), repeat=M))
    return np.array(grid, dtype=int)