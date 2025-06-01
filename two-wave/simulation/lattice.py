import numpy as np
import itertools
from rashomon.hasse import enumerate_policies, enumerate_profiles # to generate lattice and keep track of profiles

def generate_lattice_hasse(R, M):
    """
    Generate the full factorial lattice over treatments (with M possible features and R[m] levels for each feature)

    Parameters:
    R (int or list[int]): Number of levels per feature, 'int' if uniform levels and 'list' if different levels per feature)
    M (int): Number of features

    Returns:
    lattice (list[tuple[int]]): List of all possible policies, shape (K, M), each row is treatment node with entries in {1,...,R[m]}
    profiles (list): List of all possible profiles
    profile_map (dict[tuple, int]): Mapping from profile to index in profiles
    """

    R = np.array(R)
    # generate lattice and get profiles (generally and for each treatment)
    lattice = enumerate_policies(M, R)
    profiles, profile_map = enumerate_profiles(M)

    return lattice, profiles, profile_map

# Manual lattice function to verify matching with the hasse implementation
# def generate_lattice_manual(R, M):
#     """
#     Generate the full lattice, with manual construction. Used to verify matching with the hasse implementation.
#
#     As above:
#     Parameters: R, M
#     Returns: lattice (np.array), profiles, profile_map
#     """
#
#     if isinstance(R, int): # support both uniform and varying levels
#         R = [R] * M
#
#     # manual construction of the lattice
#     levels = [list(range(1, r + 1)) for r in R]
#     lattice = np.array(list(itertools.product(*levels)), dtype=int)
#     profiles = (lattice > 1).astype(int)
#
#     profile_map = {}
#     for i, p in enumerate(profiles):
#         key = tuple(p)
#         profile_map.setdefault(key, []).append(i)
#
#     return lattice, profiles, profile_map # profiles and map for later RPS construction