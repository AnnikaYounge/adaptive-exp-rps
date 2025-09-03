import numpy as np
import visualizations

from rashomon.hasse import enumerate_policies, enumerate_profiles, policy_to_profile
from rashomon.aggregate import RAggregate
from rashomon.extract_pools import extract_pools, aggregate_pools
from rashomon.loss import compute_pool_means, compute_policy_means
from rashomon.metrics import make_predictions

from allocation import (
    compute_policy_variances,
    allocate_wave1,
    allocate_wave2_pools,
    create_assignments_from_alloc,
)

from datagen import (
    generate_data_from_assignments,
    generate_true_partition_fixed_pools,
    generate_true_partition_from_means,
)

from helpers import (
    pools_to_vector,
    pools_from_partition_map,
    partition_iou,
    _policy_stats,
    _pool_index,
    _pool_counts_and_means_from_stats,
    _pool_prior_from_wave1,
    _ucb_pick,
    _thompson_pick,
)

# --- shared outcomes for fair comparisons between algorithms (CRN) ---
class OutcomeBank:
    """
    Pre-draw epsilon[j, k] so the k-th pull of policy j returns the same outcome
    across all algorithms/branches. Each branch keeps its own per-policy counters,
    but the realizations are identical given (j, k).
    """
    def __init__(self, oracle_outcomes, sig, max_draws_per_policy=200000, seed=0):
        self.mu = np.asarray(oracle_outcomes, dtype=float)
        self.sig = float(sig)
        self.K = len(self.mu)
        rng = np.random.RandomState(seed)
        # eps[j, k] ~ N(0,1); memory footprint is K * max_draws_per_policy
        self.eps = rng.normal(loc=0.0, scale=1.0,
                              size=(self.K, int(max_draws_per_policy))).astype(float)

    def draw(self, indices, counters):
        """
        indices: 1D array of policy ids for this batch
        counters: array[int] of size K (branch-specific draw counts); mutated in place
        returns: column vector y for each index using current counter per policy
        """
        idx = np.asarray(indices, dtype=int).reshape(-1)
        y = np.empty(idx.shape[0], dtype=float)
        for t, j in enumerate(idx):
            k = counters[j]  # current draw number for policy j
            y[t] = self.mu[j] + self.sig * self.eps[j, k]
            counters[j] += 1
        return y.reshape(-1, 1)


# RPS posterior and MAP extraction
def build_rps_and_map(
    M, R, H, D, y, lambda_reg,
    theta_init, theta_init_step, min_rset_size,
    profiles, policies_profiles_masked, policies_ids_profiles,
    num_workers, verbose=False
):
    """
    Coarse-to-fine theta sweep to get a non-empty Rashomon set.
    Returns posterior weights, MAP pools, and MAP loss.
    """
    theta = theta_init
    step = theta_init_step
    R_set, R_profiles = [], None

    max_cycles = 50
    num_sweeps = 3
    for sweep in range(num_sweeps):
        cycles = 0
        if verbose:
            print(f"[theta-sweep] sweep={sweep} start θ={theta:.6f} step={step:.6f} (target |R_set|≥{min_rset_size})")
        while cycles < max_cycles:
            R_set, R_profiles = RAggregate(M, R, H, D, y, theta, reg=lambda_reg, num_workers=num_workers, verbose=False)
            if verbose:
                print(f"  θ={theta:.6f} => |R_set|={len(R_set)}")
            if len(R_set) > 0 and sweep == 0:
                if verbose:
                    print(f" Found initial feasible RPS with |R_set|={len(R_set)} at theta={theta:.4f}")
                theta -= step; step = step / (M * 2); break
            if len(R_set) >= min_rset_size and sweep < num_sweeps - 1:
                theta -= step; step = step / (M * 2); break
            if len(R_set) >= min_rset_size and sweep == num_sweeps - 1:
                step = step * ((M * 2) ** (num_sweeps - 1)); break
            theta += step; cycles += 1

    if len(R_set) == 0:
        print("Warning: No feasible Rashomon set found within range.")
    elif verbose:
        print(f"End theta: {theta:.4f}, RPS size: {len(R_set)}")
    if len(R_set) == 0:
        return dict(theta=theta, R_set=[], R_profiles=None, posterior_weights=None,
                    map_idx=None, map_pools=None, map_pi_policies=None, map_loss=None)

    # Compute pi_pools / pi_policies and losses
    partition_losses = np.zeros(len(R_set))
    pi_pools_list, pi_policies_list = [], []
    for r, part_r in enumerate(R_set):
        pi_policies_profiles_r = {}
        for k, _profile in enumerate(profiles):
            sigma_k = R_profiles[k].sigma[part_r[k]]
            if sigma_k is None:
                n_prof = len(policies_profiles_masked[k])
                pi_policies_r_k = {i: 0 for i in range(n_prof)}
            else:
                _, pi_policies_r_k = extract_pools(policies_profiles_masked[k], sigma_k)
            pi_policies_profiles_r[k] = pi_policies_r_k

        pi_pools_r, pi_policies_r = aggregate_pools(pi_policies_profiles_r, policies_ids_profiles)
        pi_pools_list.append(pi_pools_r)
        pi_policies_list.append(pi_policies_r)
        partition_losses[r] = sum(R_profiles[k].loss[part_r[k]] for k in range(len(part_r)))

    w = np.exp(-partition_losses); w = w / (w.sum() if w.sum() > 0 else 1.0)
    map_idx = int(np.argmin(partition_losses))

    return dict(
        theta=theta, R_set=R_set, R_profiles=R_profiles,
        posterior_weights=w, map_idx=map_idx,
        map_pools=pi_pools_list[map_idx], map_pi_policies=pi_policies_list[map_idx],
        map_loss=float(partition_losses[map_idx])
    )

def run_twowave_experiment(
    R, H, H_true,
    lambda_reg, theta_init, theta_init_step, min_rset_size,
    num_workers,
    allocation_rule_wave1,
    wave2_algorithms,
    within_pool_rule,
    max_alloc, sig, random_seed=0,
    profile_separate_truth=True,
    micro_batch_size=20,
    rps_refresh_every=None,
    verbose=False,
    mean_spec=None,
    force_unique_best=False,
    best_gap=0.0,
    best_pool_choice="smallest",
    mean_order="id_asc",
    true_partition_override=None,   # dict {policy_id -> pool_id} OR None
    true_pool_means_override=None   # dict {pool_id -> mean} OR sequence aligned to order
):

    np.random.seed(random_seed)

    # Policies/profiles
    M = len(R)
    all_policies = enumerate_policies(M, R)
    K = len(all_policies)
    profiles, _ = enumerate_profiles(M)

    # True partition & oracle
    if (true_partition_override is not None) and (true_pool_means_override is not None):
        true_partition_map = {int(k): int(v) for k, v in true_partition_override.items()}
        oracle_outcomes, mu_pool_used = generate_true_partition_from_means(
            all_policies, R, true_partition_map, true_pool_means_override, order=mean_order
        )
    else:
        true_partition_map, oracle_outcomes, mu_pool_used = generate_true_partition_fixed_pools(
            all_policies, R, H_true,
            mu0=0.0, tau=1.0, random_seed=random_seed,
            profile_separate=profile_separate_truth,
            mean_spec=mean_spec,
            force_unique_best=force_unique_best,
            best_gap=best_gap,
            best_pool_choice=best_pool_choice,
            mean_order=mean_order
        )
    true_pools = pools_from_partition_map(true_partition_map)
    pol2pool_true_vec = pools_to_vector(true_pools, K)
    best_idx_true = int(np.argmax(oracle_outcomes))
    # Shared outcome bank for all branches (common random numbers)
    bank = OutcomeBank(oracle_outcomes, sig=sig,
                       max_draws_per_policy=max_alloc * 2,
                       seed=random_seed)
    if verbose:
        print(f"True partition has {len(true_pools)} pools (target was {H_true}).")
        print(f"Oracle best policy: {best_idx_true} with mean {oracle_outcomes[best_idx_true]:.4f}")
        print(f"True pool means: {mu_pool_used}")

    # Wave sizes
    n_wave1 = K
    n_wave2 = max_alloc - n_wave1
    if n_wave2 <= 0:
        raise ValueError("max_alloc must exceed #policies for coverage in Wave-1.")



    # -------- Wave 1: policy-level allocation --------

    D = np.empty((0,1), dtype=int)
    y = np.empty((0,1), dtype=float)

    policy_variances, policy_counts = compute_policy_variances(D, y, K)
    sigmas_est = np.sqrt(policy_variances)
    alloc1 = allocate_wave1(allocation_rule_wave1, sigmas=sigmas_est, counts=policy_counts, N=n_wave1)
    if verbose:
        print(f"Wave-1 allocation rule: {allocation_rule_wave1}")
        print(f"Wave-1 allocation counts (per policy): {alloc1}")
    D_wave = create_assignments_from_alloc(alloc1).reshape(-1, 1)

    # Use CRN so Wave-1 outcomes are identical across branches
    ctr_w1 = np.zeros(K, dtype=int) # local counters during Wave-1
    y_wave = bank.draw(D_wave.flatten(), ctr_w1)
    base_counters = ctr_w1.copy() # persist counters; Wave-2 starts from here
    D = np.vstack([D, D_wave]); y = np.vstack([y, y_wave])

    # RPS on wave-1
    # Build masked per-profile indices
    policies_ids_profiles = {}
    policies_profiles_masked = {}
    for k, profile in enumerate(profiles):
        ids = [i for i, p in enumerate(all_policies) if policy_to_profile(p) == profile]
        policies_ids_profiles[k] = ids
        profile_mask = [bool(v) for v in profile]
        masked_policies = [tuple([all_policies[i][j] for j in range(M) if profile_mask[j]]) for i in ids]
        policies_profiles_masked[k] = masked_policies

    if verbose:
        print(f"Building Wave-1 RPS on {len(D)} samples...")
    # --- build first wave RPS ---
    rps = build_rps_and_map(
        M, R, H, D, y, lambda_reg,
        theta_init, theta_init_step, min_rset_size,
        profiles, policies_profiles_masked, policies_ids_profiles,
        num_workers=num_workers, verbose=verbose
    )
    if verbose:
        print(f"[RPS] theta={rps.get('theta', np.nan):.5f}, |R_set|={len(rps.get('R_set', []))}, "
              f"MAP_loss={rps.get('map_loss', np.nan)}; has_pools={rps.get('map_pools') is not None}")
    # if no RPS, go back to policy-only
    map_pools = rps["map_pools"] if len(rps["R_set"])>0 else None

    # ---------- Wave 2 onwards, with batches for resolution ----------
    n_batches = int(np.ceil(n_wave2 / float(micro_batch_size)))

    def _metrics_snapshot(D, y, label, algo, t_seen, pools=None, pol2pool_vec=None, stats=None):
        """
        Metrics using existing APIs:
          - compute_policy_means(D,y,K) -> [:,0]=sum_y, [:,1]=count
          - compute_pool_means(stats, pools)
          - make_predictions(D, true_partition_map, pool_means_true)
        """
        # per-policy stats (reuse if provided)
        if stats is None:
            stats = compute_policy_means(D, y, K)
        means_policy = stats[:, 0] / np.maximum(stats[:, 1], 1)

        # predictor
        if pools is not None and len(pools) > 0:
            pool_means_est = compute_pool_means(stats, pools)  # vector indexed by pool_id
            if pol2pool_vec is None:
                pol2pool_vec = pools_to_vector(pools, K)
            pred = pool_means_est[pol2pool_vec]  # broadcast pool means to all members
        else:
            pred = means_policy  # per-policy empirical means

        # regret variants
        best_idx_true = int(np.argmax(oracle_outcomes))
        best_idx_pred = int(np.argmax(pred))
        # policy-level true vs. true
        regret_true_policy = float(oracle_outcomes[best_idx_true] - oracle_outcomes[best_idx_pred])
        # pool-aware regrets if this branch is pooled
        regret_true_pool_mean = np.nan
        regret_true_pool_best = np.nan
        if pools is not None and pol2pool_vec is not None:
            chosen_pool = int(pol2pool_vec[best_idx_pred])  # pool we effectively selected
            members = np.array(pools[chosen_pool], dtype=int)
            true_mean_of_pool = float(np.mean(oracle_outcomes[members]))
            true_best_in_pool = float(np.max(oracle_outcomes[members]))
            regret_true_pool_mean = float(oracle_outcomes[best_idx_true] - true_mean_of_pool)
            regret_true_pool_best = float(oracle_outcomes[best_idx_true] - true_best_in_pool)

            # Report pooled version as the main "true-vs-true" regret for RPS branches
            regret_true = regret_true_pool_mean
        else:
            regret_true = regret_true_policy

        # "true-vs-estimated" instant loss
        regret_est = float(oracle_outcomes[best_idx_true] - float(pred[best_idx_pred]))

        # MSE under the true partition
        pool_means_true = compute_pool_means(stats, true_pools)
        yhat_true = np.asarray(make_predictions(D, true_partition_map, pool_means_true)).reshape(-1)
        y_vec = y.reshape(-1)
        mse_oracle_partition = float(np.mean((y_vec - yhat_true) ** 2))

        # pool-parameter estimation error (vs truth)
        true_mu_pool = {pid: float(np.mean(oracle_outcomes[np.array(mems, int)]))
                        for pid, mems in true_pools.items()}
        mse_pool_est = float(np.mean([(pool_means_true[pid] - true_mu_pool[pid]) ** 2
                                      for pid in true_pools.keys()]))

        # top-k IoU
        k = min(10, K)
        topk_pred = set(np.argsort(-pred)[:k])
        topk_true = set(np.argsort(-oracle_outcomes)[:k])
        iou_topk = len(topk_pred & topk_true) / len(topk_pred | topk_true)

        return dict(
            strategy=label, algorithm=algo, t=t_seen,
            regret=regret_est,
            regret_true=regret_true,  # pooled if RPS, policy-level if not
            regret_true_policy=regret_true_policy,  # always policy-level
            regret_true_pool_mean=regret_true_pool_mean,  # only filled for RPS
            regret_true_pool_best=regret_true_pool_best,  # only filled for RPS
            mse_oracle_partition=mse_oracle_partition,
            mse_pool_est=mse_pool_est,
            iou_topk=iou_topk,
            best_idx_pred=best_idx_pred,
            best_idx_true=best_idx_true
        )

    # initialize logs
    metrics = []
    final_snapshots = {}

    rps_size_current = len(rps.get("R_set", []))
    map_npools_current = len(rps["map_pools"]) if rps.get("map_pools") else 0
    # --- POLICY-ONLY baselines ---
    for algo in wave2_algorithms:
        Dp = D.copy(); yp = y.copy()
        t_seen = len(Dp)

        for b in range(n_batches):
            n_this = min(micro_batch_size, n_wave2 - b*micro_batch_size)
            if n_this <= 0: break

            # pick indices according to algorithm (policy-level)
            means, counts, vars_ = _policy_stats(Dp, yp, K)

            if algo == "uniform":
                start = (b * n_this) % K
                alloc_indices = np.array([(start + i) % K for i in range(n_this)], dtype=int)
                D_batch = alloc_indices.reshape(-1, 1)

            elif algo == "neyman":
                stds = np.sqrt(np.maximum(vars_, 1e-12))
                p = stds / (stds.sum() if stds.sum() > 0 else 1.0)
                alloc_indices = np.random.choice(np.arange(K), size=n_this, p=p)
                D_batch = np.asarray(alloc_indices, dtype=int).reshape(-1, 1)

            elif algo == "minimax":
                tmp_counts = counts.copy()
                tmp_vars = vars_.copy()
                picks = []
                for _ in range(n_this):
                    j = int(np.argmax(tmp_vars))
                    picks.append(j)
                    tmp_counts[j] += 1
                    tmp_vars[j] = max(tmp_vars[j] * tmp_counts[j] / (tmp_counts[j] + 1.0), 1e-12)
                D_batch = np.asarray(picks, dtype=int).reshape(-1, 1)

            elif algo == "ucb":
                # sequential UCB with CRN
                picks = []
                if b == 0:
                    ctr_policy_only = base_counters.copy()
                for t_inner in range(n_this):
                    means, counts, vars_ = _policy_stats(Dp, yp, K)
                    j = _ucb_pick(means, np.maximum(counts, 1), t_seen + t_inner + 1, c=2.0)
                    y_one = bank.draw(np.array([j]), ctr_policy_only)
                    Dp = np.vstack([Dp, [[j]]]);
                    yp = np.vstack([yp, y_one])
                    t_seen += 1
                    picks.append(j)
                D_batch = np.asarray(picks, dtype=int).reshape(-1, 1)

            elif algo == "thompson":
                # sequential Gaussian Thompson
                picks = []
                if b == 0:
                    ctr_policy_only = base_counters.copy()
                for t_inner in range(n_this):
                    means, counts, vars_ = _policy_stats(Dp, yp, K)
                    j = _thompson_pick(means, np.maximum(counts, 1), prior_var=1.0, obs_var=1.0)
                    y_one = bank.draw(np.array([j]), ctr_policy_only)
                    Dp = np.vstack([Dp, [[j]]]);
                    yp = np.vstack([yp, y_one])
                    t_seen += 1
                    picks.append(j)
                D_batch = np.asarray(picks, dtype=int).reshape(-1, 1)

            else:
                raise ValueError(f"Unknown algorithm: {algo}")

            # use CRN: draw batched outcomes only if we didn't already draw step-by-step
            if algo in ("uniform", "neyman", "minimax"):
                if b == 0:
                    ctr_policy_only = base_counters.copy()
                y_batch = bank.draw(D_batch.flatten(), ctr_policy_only)
                Dp = np.vstack([Dp, D_batch]);
                yp = np.vstack([yp, y_batch])
                t_seen = len(Dp)
            # (ucb/thompson already updated Dp, yp, t_seen inside their loops)

            stats_batch = compute_policy_means(Dp, yp, K)  # reuse for metrics
            metrics.append(_metrics_snapshot(Dp, yp, "policy_only", algo, t_seen, pools=None, pol2pool_vec=None, stats=stats_batch))
            metrics[-1].update({
                "rps_size": rps_size_current,
                "map_num_pools": map_npools_current
            })
            if verbose and (b % 2 == 0 or b == n_batches - 1):
                last = metrics[-1]
                print(f"[{last['strategy']}/{last['algorithm']}] t={last['t']} "
                      f"regret={last['regret']:.3f} mse*={last['mse_oracle_partition']:.4f} "
                      f"best_pred={last['best_idx_pred']} true_best={last['best_idx_true']}")
            # after finishing all batches for this algo:
            final_snapshots[("policy_only", algo)] = {"D": Dp, "y": yp}
    # --- RPS-ASSISTED baselines (allocate to MAP pools) ---
    for algo in wave2_algorithms:
        Dr = D.copy(); yr = y.copy()
        t_seen = len(Dr)
        current_map_pools = map_pools
        # initialize both mapping vectors for the first batches
        if current_map_pools is not None and len(current_map_pools) > 0:
            map_pol2pool_vec = pools_to_vector(current_map_pools, K)  # used in metrics snapshot
            policy_to_pool = map_pol2pool_vec  # used in UCB/Thompson
        else:
            map_pol2pool_vec = None
            policy_to_pool = None

        for b in range(n_batches):
            n_this = min(micro_batch_size, n_wave2 - b*micro_batch_size)
            if n_this <= 0: break

            # optional refresh of RPS
            if (rps_refresh_every is not None) and (b > 0) and (b % int(rps_refresh_every) == 0):
                stats = compute_policy_means(Dr, yr, K)  # keep cost similar
                rps_new = build_rps_and_map(
                    M, R, H, Dr, yr, lambda_reg,
                    theta_init, theta_init_step, min_rset_size,
                    profiles, policies_profiles_masked, policies_ids_profiles,
                    num_workers=num_workers, verbose=False
                )
                rps_size_current = len(rps_new.get("R_set", []))

                if len(rps_new["R_set"]) > 0:
                    current_map_pools = rps_new["map_pools"]
                    if verbose:
                        print(f"[RPS] refresh IoU(map, truth) = {partition_iou(current_map_pools, true_pools):.3f}")

                # update after possibly changing current_map_pools
                map_npools_current = len(current_map_pools) if current_map_pools else 0
                if current_map_pools is not None and len(current_map_pools) > 0:
                    map_pol2pool_vec = pools_to_vector(current_map_pools, K)
                    policy_to_pool = map_pol2pool_vec # keep UCB/Thompson in sync
                else:
                    map_pol2pool_vec = None
                    policy_to_pool = None

            # if we still don't have pools, fall back to policy-only
            if current_map_pools is None or len(current_map_pools) == 0:
                means, counts, vars_ = _policy_stats(Dr, yr, K)
                alloc_indices = np.random.choice(np.arange(K), size=n_this) # safe fallback
                D_batch = alloc_indices.reshape(-1, 1)
                drew_online = False
            else:
                # convert pool allocation to policy assignments
                stats = compute_policy_means(Dr, yr, K)
                vars_now, counts_now = compute_policy_variances(Dr, yr, K)
                sigmas = np.sqrt(vars_now)

                if algo in ("uniform", "neyman", "minimax", "best_pool"):
                    alloc_policy = allocate_wave2_pools(
                        rule=algo, # normalized names
                        pool_to_policies=current_map_pools,
                        policy_sigmas=sigmas,
                        N=n_this,
                        policy_counts=counts_now,
                        pool_weights=None,
                        pool_gaps=None,
                        within_rule=within_pool_rule
                    )
                    D_batch = create_assignments_from_alloc(alloc_policy).reshape(-1, 1)
                    drew_online = False

                elif algo == "ucb":
                    # --- pool-level UCB ---
                    idx_map, pool_ids = _pool_index(current_map_pools)
                    picks = []
                    # CRN counters for RPS branch
                    if b == 0:
                        ctr_rps = base_counters.copy()

                    for t_inner in range(n_this):
                        stats_now = compute_policy_means(Dr, yr, K)
                        counts_g, means_g = _pool_counts_and_means_from_stats(stats_now, current_map_pools)
                        counts_eff = np.maximum(counts_g, 1)
                        bonus = np.sqrt((2.0 * np.log(max(t_seen + t_inner + 1, 1))) / counts_eff)
                        scores = means_g + 2.0 * bonus  # c=2.0
                        g_idx = int(np.argmax(scores))
                        chosen_pool = int(pool_ids[g_idx])

                        members = list(current_map_pools[chosen_pool])
                        # within-pool choice
                        j = int(np.random.choice(members))

                        # draw one outcome (CRN) and update state
                        y_one = bank.draw(np.array([j]), ctr_rps)
                        Dr = np.vstack([Dr, [[j]]]); yr = np.vstack([yr, y_one])
                        t_seen += 1
                        picks.append(j)

                    D_batch = np.array(picks, dtype=int).reshape(-1, 1)
                    drew_online = True

                elif algo == "thompson":
                    # --- pool-level Thompson with Wave-1 MAP prior, allows refresh ---
                    obs_var = float(sig ** 2)
                    prior_inflation = 1.0
                    pool_ids, mu0, v0, n0 = _pool_prior_from_wave1(D, y, K, current_map_pools, obs_var, prior_inflation)
                    P = len(pool_ids)
                    pool_id_to_idx = {pid: i for i, pid in enumerate(pool_ids)}

                    picks = []
                    if b == 0:
                        ctr_rps = base_counters.copy()

                    for t_inner in range(n_this):
                        # current cumulative stats
                        stats_now = compute_policy_means(Dr, yr, K)
                        stats_w1  = compute_policy_means(D,  y,  K)   # Wave-1 only

                        # aggregate to pools, then take Wave-2 increments = now - w1
                        cnt_now, sum_now = _pool_counts_and_means_from_stats(stats_now, current_map_pools)
                        cnt_w1,  sum_w1  = _pool_counts_and_means_from_stats(stats_w1,  current_map_pools)
                        cnt_w2 = np.maximum(cnt_now - cnt_w1, 0)
                        sum_w2 = np.maximum(sum_now - sum_w1, 0.0)

                        # posterior from (prior + Wave-2 data)
                        prec_post = (1.0 / np.maximum(v0, 1e-12)) + (cnt_w2 / obs_var)
                        var_post  = 1.0 / np.maximum(prec_post, 1e-12)
                        mean_post = ((mu0 / np.maximum(v0, 1e-12)) + (sum_w2 / obs_var)) / np.maximum(prec_post, 1e-12)

                        samples = np.random.normal(mean_post, np.sqrt(np.maximum(var_post, 1e-12)))
                        g_idx = int(np.argmax(samples))
                        chosen_pool = int(pool_ids[g_idx])

                        # within-pool choice: random (or use within_pool_rule variants like above)
                        members = list(current_map_pools[chosen_pool])
                        j = int(np.random.choice(members))

                        # draw one outcome and update state
                        y_one = float(bank.draw(np.array([j]), ctr_rps).ravel()[0])
                        Dr = np.vstack([Dr, [[j]]]); yr = np.vstack([yr, [[y_one]]])
                        t_seen += 1
                        picks.append(j)

                    D_batch = np.array(picks, dtype=int).reshape(-1, 1)
                    drew_online = True

                else:
                    raise ValueError("Unknown algorithm: %s" % algo)

            # use CRN: RPS-assisted branch has its own counters, seeded from Wave-1
            # draw outcomes only if we didn't already draw step-by-step above
            if not drew_online:
                if b == 0:
                    ctr_rps = base_counters.copy()
                y_batch = bank.draw(D_batch.flatten(), ctr_rps)
                Dr = np.vstack([Dr, D_batch]);
                yr = np.vstack([yr, y_batch])
                t_seen = len(Dr)

            stats_batch = compute_policy_means(Dr, yr, K) # reuse for metrics
            metrics.append(_metrics_snapshot(Dr, yr, "rps_assisted", algo, t_seen,
                                             pools=current_map_pools, pol2pool_vec=map_pol2pool_vec,
                                             stats=stats_batch))
            metrics[-1].update({
                "rps_size": rps_size_current,
                "map_num_pools": map_npools_current
            })
            if verbose and (b % 2 == 0 or b == n_batches - 1):
                last = metrics[-1]
                print(f"[{last['strategy']}/{last['algorithm']}] t={last['t']} "
                      f"regret={last['regret']:.3f} mse*={last['mse_oracle_partition']:.4f} "
                      f"best_pred={last['best_idx_pred']} true_best={last['best_idx_true']}")
            # after finishing all batches for this algo:
            final_snapshots[("rps_assisted", algo)] = {
                "D": Dr, "y": yr, "pools": current_map_pools
            }

    return dict(
        metrics=metrics,
        D=D, y=y,
        all_policies=all_policies,
        true_partition_map=true_partition_map,
        true_pools=true_pools,
        oracle_outcomes=oracle_outcomes,
        rps=rps,
        final_snapshots=final_snapshots
    )