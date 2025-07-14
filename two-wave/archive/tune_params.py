# import numpy as np
# from rashomon.aggregate import RAggregate
#
# ## CURRENT DRAFT of potential 'sweep' approach to getting optimal parameters
# UNUSED IN FINAL DRAFT


# def tune_lambda(
#     policies, D1, y1, R, H, lambda_grid=None, theta=10
# ):
#     """
#     Sweep over lambda (regularization) at fixed θ.
#     For each lambda, return:
#       - Q_min(lambda): sum of min-loss per profile
#       - rps_size(lambda): size of RPS
#       - all_losses(lambda): array of total losses for all models in RPS
#       - all_sizes(lambda): array of sizes for all models in RPS
#     """
#     if lambda_grid is None:
#         lambda_grid = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
#     else:
#         lambda_grid = np.array(lambda_grid, dtype=float)
#
#     L = lambda_grid.size
#     Q_min_vals = np.zeros(L, dtype=float)
#     rps_size_vals = np.zeros(L, dtype=int)
#     all_losses = []
#     all_sizes = []
#
#     M = len(policies[0])
#
#     for i, lam in enumerate(lambda_grid):
#         R_set_lam, R_profiles_lam = RAggregate(
#             M=M, R=R, H=H, D=D1, y=y1, theta=theta, reg=lam
#         )
#
#         # Compute total loss and size for each model in R_set_lam
#         model_losses = []
#         model_sizes = []
#         num_profiles = len(R_profiles_lam)
#         for r_set in R_set_lam:
#             loss_r = 0.0
#             size_r = 0
#             for k in range(num_profiles):
#                 idx_subpart = r_set[k]
#                 rp = R_profiles_lam[k]
#                 if rp is None or rp.loss is None or rp.loss.size == 0:
#                     continue
#                 loss_k = rp.loss[idx_subpart]
#                 sigma_k = rp.sigma[idx_subpart]
#                 pools_k = rp.pools[idx_subpart]
#                 loss_r += loss_k
#                 if not (sigma_k is None and loss_k == 0):
#                     size_r += pools_k
#             model_losses.append(loss_r)
#             model_sizes.append(size_r)
#
#         model_losses = np.array(model_losses)
#         model_sizes = np.array(model_sizes)
#         all_losses.append(model_losses)
#         all_sizes.append(model_sizes)
#
#         Q_min_vals[i] = model_losses.min() if model_losses.size > 0 else np.nan
#         rps_size_vals[i] = len(R_set_lam)
#
#     return {
#         'lambda': lambda_grid,
#         'Q_min': Q_min_vals,
#         'rps_size': rps_size_vals,
#         'all_losses': all_losses,
#         'all_sizes': all_sizes,
#         'theta': theta
#     }
#
#
# def tune_theta(
#     policies, D1, y1, R, H, lambda_fixed, theta_grid=None
# ):
#     """
#     Sweep over theta at fixed lambda.
#     For each theta, return:
#       - Q_min(theta): sum of min-loss per profile
#       - rps_size(theta): size of RPS
#       - all_losses(theta): array of total losses for all models in RPS
#       - all_sizes(theta): array of sizes for all models in RPS
#     """
#     if theta_grid is None:
#         theta_grid = np.linspace(0.0, 10.0, 41)
#     else:
#         theta_grid = np.array(theta_grid, dtype=float)
#
#     T = theta_grid.size
#     Q_min_vals = np.zeros(T, dtype=float)
#     rps_size_vals = np.zeros(T, dtype=int)
#     all_losses = []
#     all_sizes = []
#
#     M = len(policies[0])
#
#     for i, theta in enumerate(theta_grid):
#         R_set_th, R_profiles_th = RAggregate(
#             M=M, R=R, H=H, D=D1, y=y1, theta=theta, reg=lambda_fixed
#         )
#
#         # Compute total loss and size for each model in R_set_th
#         model_losses = []
#         model_sizes = []
#         num_profiles = len(R_profiles_th)
#         for r_set in R_set_th:
#             loss_r = 0.0
#             size_r = 0
#             for k in range(num_profiles):
#                 idx_subpart = r_set[k]
#                 rp = R_profiles_th[k]
#                 if rp is None or rp.loss is None or rp.loss.size == 0:
#                     continue
#                 loss_k = rp.loss[idx_subpart]
#                 sigma_k = rp.sigma[idx_subpart]
#                 pools_k = rp.pools[idx_subpart]
#                 loss_r += loss_k
#                 if not (sigma_k is None and loss_k == 0):
#                     size_r += pools_k
#             model_losses.append(loss_r)
#             model_sizes.append(size_r)
#
#         model_losses = np.array(model_losses)
#         model_sizes = np.array(model_sizes)
#         all_losses.append(model_losses)
#         all_sizes.append(model_sizes)
#
#         Q_min_vals[i] = model_losses.min() if model_losses.size > 0 else np.nan
#         rps_size_vals[i] = len(R_set_th)
#
#     return {
#         'theta': theta_grid,
#         'Q_min': Q_min_vals,
#         'rps_size': rps_size_vals,
#         'all_losses': all_losses,
#         'all_sizes': all_sizes,
#         'lambda': lambda_fixed
#     }