from rashomon.extract_pools import extract_pools
from rashomon.loss import compute_pool_means
from itertools import combinations

def plot_volume_perimeter_tradeoff(profile, sigma_idx):
    """
    Scatterplot of (Volume, Perimeter) for all pools in a profile's MAP partition.
    """
    pool_sizes = [len(p) for p in profile.pools[sigma_idx]]
    boundary_mat = profile.boundary[sigma_idx]

    volumes = np.array(pool_sizes)
    perimeter = boundary_mat.sum(axis=1)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=volumes, y=perimeter, s=80, color='darkorange', edgecolor="black")
    for i, (v, p) in enumerate(zip(volumes, perimeter)):
        plt.text(v + 0.3, p + 0.3, str(i), fontsize=8)
    plt.xlabel("Volume (Pool Size)")
    plt.ylabel("Perimeter (Boundary Links)")
    plt.title(f"Volume–Perimeter Diagnostic (Profile {profile.profile_idx})")
    plt.tight_layout()
    plt.show()

def plot_volume_perimeter_ratio_distribution(R_profiles, R_set, map_idx):
    """
    Histogram of volume/perimeter ratios across all pools in the MAP partition.
    """
    all_ratios = []
    for k, rp in enumerate(R_profiles):
        sigma_idx = int(R_set[map_idx][k])
        boundary = rp.boundary[sigma_idx]
        sizes = np.array([len(pool) for pool in rp.pools[sigma_idx]])
        perims = boundary.sum(axis=1)
        ratios = sizes / (perims + 1e-6)
        all_ratios.extend(ratios)

    plt.figure(figsize=(7, 4))
    sns.histplot(all_ratios, bins=20, color="teal", edgecolor="white")
    plt.xlabel("Volume / Perimeter")
    plt.ylabel("Count")
    plt.title("Volume-to-Perimeter Ratios Across MAP Pools")
    plt.tight_layout()
    plt.show()

def plot_volume_perimeter_scatter_all(R_profiles, R_set, map_idx):
    """
    Global scatterplot of (volume, perimeter) for all pools across all profiles.
    """
    all_vols, all_perims, all_profiles = [], [], []
    for k, rp in enumerate(R_profiles):
        sigma_idx = int(R_set[map_idx][k])
        boundary = rp.boundary[sigma_idx]
        pools = rp.pools[sigma_idx]
        sizes = np.array([len(p) for p in pools])
        perims = boundary.sum(axis=1)
        all_vols.extend(sizes)
        all_perims.extend(perims)
        all_profiles.extend([k] * len(pools))

    df = pd.DataFrame({
        "Volume": all_vols,
        "Perimeter": all_perims,
        "Profile": all_profiles
    })

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="Volume", y="Perimeter", hue="Profile", palette="Set2", s=80, edgecolor="black")
    plt.title("Volume vs. Perimeter Across All Profiles (MAP)")
    plt.tight_layout()
    plt.show()


# ----- A FEW MORE NEW ONES -----

def plot_boundary_sensitivity_distribution(R_profiles, R_set, map_idx):
    """
    Show histogram of (Perimeter / Volume) across all MAP pools — a proxy for boundary sensitivity.
    """
    sensitivity_ratios = []
    for k, rp in enumerate(R_profiles):
        sigma_idx = int(R_set[map_idx][k])
        boundary = rp.boundary[sigma_idx]
        pools = rp.pools[sigma_idx]
        volumes = np.array([len(p) for p in pools])
        perims = boundary.sum(axis=1)
        ratios = perims / (volumes + 1e-6)
        sensitivity_ratios.extend(ratios)

    plt.figure(figsize=(7, 4))
    sns.histplot(sensitivity_ratios, bins=25, color="tomato", edgecolor="white")
    plt.xlabel("Perimeter / Volume (Sensitivity)")
    plt.ylabel("Count")
    plt.title("Boundary Sensitivity Across MAP Pools")
    plt.tight_layout()
    plt.show()

def plot_pool_embedding_2d(all_policies, R_profiles, R_set, map_idx):
    """
    Embeds the MAP partition pools into 2D using PCA to show structural partitioning.
    """
    policies_np = np.array(all_policies)
    pca = PCA(n_components=2)
    Z = pca.fit_transform(policies_np)

    pool_labels = -1 * np.ones(len(all_policies), dtype=int)
    label_counter = 0
    for k, rp in enumerate(R_profiles):
        sigma_idx = int(R_set[map_idx][k])
        for pool in rp.pools[sigma_idx]:
            for idx in pool:
                pool_labels[idx] = label_counter
            label_counter += 1

    valid = pool_labels != -1
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=Z[valid, 0], y=Z[valid, 1], hue=pool_labels[valid], palette="tab20", s=50, edgecolor="black")
    plt.title("MAP Partition Pools (PCA projection)")
    plt.legend(title="Pool", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_boundary_density_distribution(R_profiles, R_set, map_idx, policies_profiles, lattice_edges):
    densities = []
    for k, Rk in enumerate(R_profiles):
        sigma = Rk.sigma[R_set[map_idx][k]]
        policies = policies_profiles[k]
        pi_pools, _ = extract_pools(policies, sigma, lattice_edges)
        B = compute_pool_boundary_matrix(pi_pools, lattice_edges)

        sizes = np.array([len(pool) for pool in pi_pools.values()])
        perims = B.sum(axis=1)
        dens = perims / (sizes + 1e-5)
        densities.extend(dens)

    plt.figure(figsize=(7, 4))
    sns.histplot(densities, bins=20, color="slateblue", edgecolor="white")
    plt.xlabel("Boundary Density (Perimeter / Volume)")
    plt.ylabel("Count")
    plt.title("Boundary Density of Pools in MAP Partition")
    plt.tight_layout()
    plt.show()

def plot_pool_clusters_by_geometry(R_profiles, R_set, map_idx, policies_profiles, lattice_edges):
    X, colors = [], []
    for k, Rk in enumerate(R_profiles):
        sigma = Rk.sigma[R_set[map_idx][k]]
        policies = policies_profiles[k]
        pi_pools, _ = extract_pools(policies, sigma, lattice_edges)
        B = compute_pool_boundary_matrix(pi_pools, lattice_edges)

        for v, p in zip([len(pool) for pool in pi_pools.values()], B.sum(axis=1)):
            X.append([v, p])
            colors.append(k)

    X = np.array(X)
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=colors, palette="Set1", edgecolor="black", s=80)
    plt.xlabel("Volume")
    plt.ylabel("Perimeter")
    plt.title("2D Projection of Pool Geometry (Grouped by Profile)")
    plt.tight_layout()
    plt.show()
