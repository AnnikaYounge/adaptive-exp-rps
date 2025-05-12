""""
defines valid mechanistic causal models phi: Λ ⊆ [0,1]^M → ℝ for use in simulation.

constructs functions that satisfy:
- continuity or Lipschitz continuity (nearby treatments in the lattice tend to exhibit correlated effects, 'local' changes)
- non-ephemerality: no measure-zero equivalence artifacts
- non-trivial: not purely additive or constant - increments are generally correlated unless the function is purely additive
"""

import numpy as np

# purely additive model for reference
def phi_additive(v: np.ndarray, weights: np.ndarray = None) -> float:
    if weights is None: # if weights aren't specified, generate
        rng = np.random.default_rng(seed=1)
        weights = rng.uniform(0.5, 1.5, size=v.shape[0])
    return np.dot(weights, v)

# additive with 2-way interaction. satisfies smoothness, interaction, non-ephemerality
def phi_interaction(v: np.ndarray, alpha=0.5) -> float:
    base = phi_additive(v)
    return float(base + alpha * v[0] * v[1])

# polynomial with additive base
def phi_polynomial(v: np.ndarray, degree=2) -> float:
    base = phi_additive(v)
    interaction = float(np.sum(v**degree))
    return base + 0.3 * interaction

# non-linear, smooth, non-ephemeral. sine interactions and bumps so non-additive
def phi_bumpy(v: np.ndarray) -> float:
    base = phi_additive(v)
    bump = np.sin(np.sum(v)) + np.cos(v[0] - v[1])
    return base + 1.2 * bump

PHI_MODELS = {
    "additive": phi_additive,
    "interaction": phi_interaction,
    "polynomial": phi_polynomial,
    "bumpy": phi_bumpy,
}

def compute_true_effects(lattice: np.ndarray, model: str = "interaction") -> np.ndarray:
    """
    Evaluate phi(v) for all v in lattice. returns beta vector.
    """
    phi_fn = PHI_MODELS[model]
    return np.array([phi_fn(v) for v in lattice])