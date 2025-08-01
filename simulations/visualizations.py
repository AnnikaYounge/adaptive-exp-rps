import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- MAIN ----
def plot_map_true_vs_predicted_bar_topk(sorted_idx, sorted_means, oracle_beta, oracle_ranks, top_k_indices, N = 20):
    df = pd.DataFrame({
        "MAP_Rank": np.arange(N),
        "Policy_Idx": sorted_idx[:N],
        "MAP_Pred": sorted_means[:N],
        "Oracle_Value": oracle_beta[sorted_idx[:N]],
        "Oracle_Rank": oracle_ranks[sorted_idx[:N]],
        "Is_TopK": [idx in top_k_indices for idx in sorted_idx[:N]]
    })

    plt.figure(figsize=(12, 5))
    sns.barplot(
        x="MAP_Rank", y="Oracle_Value", data=df,
        hue="Is_TopK", dodge=False,
        palette={True: "red", False: "steelblue"},
        alpha=1, edgecolor='white'
    )
    sns.lineplot(
        x="MAP_Rank", y="MAP_Pred", data=df,
        marker="o", color="crimson", label="MAP Predicted Mean"
    )
    plt.ylabel("True Outcome Value")
    plt.xlabel("MAP Policy Ranking")
    plt.title(f"MAP Predicted Policies: Oracle Value and Predicted Mean (Top {N})")
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles = [
        Patch(facecolor='red', edgecolor='white', label='Oracle Top-k'),
        Patch(facecolor='steelblue', edgecolor='white', label='Not Top-k'),
        handles[-1]
    ]
    plt.legend(handles=new_handles, title="")
    plt.xticks(ticks=np.arange(N), labels=sorted_idx[:N])
    plt.tight_layout()
    plt.show()
    return df  # useful for downstream

def plot_map_true_vs_predicted_bar_observed(sorted_idx, sorted_means, oracle_beta, oracle_ranks, observed_policy_indices, N=20):
    observed_set = set(observed_policy_indices)
    df = pd.DataFrame({
        "MAP_Rank": np.arange(N),
        "Policy_Idx": sorted_idx[:N],
        "MAP_Pred": sorted_means[:N],
        "Oracle_Value": oracle_beta[sorted_idx[:N]],
        "Oracle_Rank": oracle_ranks[sorted_idx[:N]],
        "Is_Observed": [idx in observed_set for idx in sorted_idx[:N]]
    })

    plt.figure(figsize=(12, 5))
    sns.barplot(
        x="MAP_Rank", y="Oracle_Value", data=df,
        hue="Is_Observed", dodge=False,
        palette={True: "seagreen", False: "lightcoral"},
        alpha=1, edgecolor='white'
    )
    sns.lineplot(
        x="MAP_Rank", y="MAP_Pred", data=df,
        marker="o", color="crimson", label="MAP Predicted Mean"
    )
    plt.ylabel("True Outcome Value")
    plt.xlabel("MAP Policy Ranking")
    plt.title(f"MAP Policies: Oracle Value by Observed Status (Top {N})")
    handles, labels = plt.gca().get_legend_handles_labels()
    new_handles = [
        Patch(facecolor='seagreen', edgecolor='white', label='Observed'),
        Patch(facecolor='lightcoral', edgecolor='white', label='Unobserved'),
        handles[-1]
    ]
    plt.legend(handles=new_handles, title="")
    plt.xticks([])  # Hide all xticks
    plt.tight_layout()
    plt.show()

    return df

def plot_map_regression_observed(df):
    df_clean = df.dropna(subset=["MAP_Pred", "Oracle_Value"])
    observed_mask = df_clean["Is_Observed"]

    plt.figure(figsize=(12, 5))
    sns.scatterplot(
        data=df_clean,
        x="Oracle_Value", y="MAP_Pred",
        hue="Is_Observed",
        palette={True: "seagreen", False: "lightcoral"},
        alpha=0.8, edgecolor="black", s=70
    )

    # Regression line (fitted on all points)
    model = LinearRegression().fit(
        df_clean["Oracle_Value"].values.reshape(-1, 1),
        df_clean["MAP_Pred"].values
    )
    x_vals = np.linspace(df_clean["Oracle_Value"].min(), df_clean["Oracle_Value"].max(), 100)
    y_pred = model.predict(x_vals.reshape(-1, 1))
    plt.plot(x_vals, y_pred, color="black", linestyle="--", label="Regression Line")

    # Identity line
    plt.plot(x_vals, x_vals, color="grey", linestyle=":", label="Ideal (y = x)")

    r2 = r2_score(df_clean["MAP_Pred"], model.predict(df_clean["Oracle_Value"].values.reshape(-1, 1)))
    plt.title(f"Observed vs Unobserved Regression (R² = {r2:.2f})")
    plt.xlabel("True Oracle Value")
    plt.ylabel("MAP Predicted Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_map_regret_bar(sorted_idx, oracle_beta, top_idx, N=20):
    regrets = oracle_beta[top_idx] - oracle_beta[sorted_idx[:N]]
    labels = [str(idx) for idx in sorted_idx[:N]]

    plt.figure(figsize=(12, 4))
    sns.barplot(x=labels, y=regrets, color="cornflowerblue", edgecolor="black")
    plt.ylabel("Regret vs. Oracle Best")
    plt.xlabel("MAP Policy Index (Top N)")
    plt.title(f"Regret for MAP's Top {N} Policies")
    plt.tight_layout()
    plt.show()

    print(f"MAP top-{N} min regret: {regrets.min():.3f}  max regret: {regrets.max():.3f}")

def plot_map_regression(df):
    mask = ~np.isnan(df.MAP_Pred) & ~np.isnan(df.Oracle_Value)
    X = df.Oracle_Value[mask].values.reshape(-1, 1)
    y = df.MAP_Pred[mask].values

    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    slope = model.coef_[0]
    intercept = model.intercept_

    print(f"Regression results: R2 = {r2:.3f}, Slope = {slope:.2f}, Intercept = {intercept:.2f}")

    plt.figure(figsize=(12, 4))
    sns.scatterplot(x="Oracle_Value", y="MAP_Pred", data=df, color="dodgerblue", s=80, edgecolor="black")
    sns.regplot(x="Oracle_Value", y="MAP_Pred", data=df, scatter=False, color="crimson", line_kws={'label':"Regression"})
    plt.plot(
        [df.Oracle_Value.min(), df.Oracle_Value.max()],
        [df.Oracle_Value.min(), df.Oracle_Value.max()],
        'k--', alpha=0.7, label="Ideal"
    )
    plt.xlabel("True Oracle Value")
    plt.ylabel("MAP Predicted Mean")
    plt.title(f"MAP's Top Policies: Regression on True vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_oracle_ordered_bar(df, top_k_indices, oracle_beta, all_policies, N=20):
    oracle_sorted = np.argsort(-oracle_beta)
    indices = oracle_sorted[:N]
    labels = ['-'.join(map(str, all_policies[i])) for i in indices]
    true_vals = oracle_beta[indices]
    map_preds = df.set_index("Policy_Idx").reindex(indices)["MAP_Pred"]

    df2 = pd.DataFrame({
        "Oracle_Rank": np.arange(N),
        "Policy_Label": labels,
        "Oracle_Value": true_vals,
        "MAP_Pred": map_preds.values
    })

    plt.figure(figsize=(12, 4))
    sns.barplot(x="Oracle_Rank", y="Oracle_Value", data=df2, color="orange", edgecolor="black", label="Oracle Value")
    sns.lineplot(x="Oracle_Rank", y="MAP_Pred", data=df2, marker="o", color="crimson", label="MAP Prediction")
    plt.ylabel("Value")
    plt.xlabel("Oracle Policy Rank")
    plt.title(f"Oracle Top {N} Policies: True vs MAP Predicted Mean")
    plt.legend()
    plt.xticks(np.arange(-1, N, 1)+1)
    plt.tight_layout()
    plt.show()

    diffs = df2["Oracle_Value"] - df2["MAP_Pred"]
    print(f"Oracle top-{N} min/max regret (MAP predicted minus true): {diffs.min():.3f} / {diffs.max():.3f}")

def plot_minimax_risk_matrix(profile_losses, map_idx=None):
    """
    Heatmap showing loss per profile × partition, with MAP index marked.
    Handles ragged loss vectors using NaN padding.
    """
    max_len = max(len(losses) for losses in profile_losses)
    loss_matrix = np.full((len(profile_losses), max_len), np.nan)
    for i, losses in enumerate(profile_losses):
        loss_matrix[i, :len(losses)] = losses

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(loss_matrix, cmap="YlOrBr", cbar_kws={'label': 'Loss'}, mask=np.isnan(loss_matrix))
    ax.set_xlabel("Partition Index")
    ax.set_ylabel("Profile Index")
    ax.set_title("Loss Landscape: Profile × Partition")
    if map_idx is not None:
        ax.axvline(map_idx, color='crimson', linestyle='--', label="MAP")
        ax.legend()
    plt.tight_layout()
    plt.show()

# ---- extra
from sklearn.decomposition import PCA

def plot_policy_coverage_map(all_policies, observed_indices, rps_indices):
    """
    Projects policies to 2D using PCA, colors by status: observed, RPS, both, unobserved.
    """
    from sklearn.decomposition import PCA
    policies_np = np.array(all_policies)
    Z = PCA(n_components=2).fit_transform(policies_np)

    n = len(all_policies)
    labels = np.zeros(n, dtype=int)  # default: unobserved

    labels[np.array(list(rps_indices))] = 1
    labels[np.array(list(observed_indices))] += 2

    # 0: unobserved, 1: RPS only, 2: observed only, 3: both
    label_names = {0: "Unobserved", 1: "In RPS", 2: "Observed", 3: "Observed & In RPS"}
    colors = {
        "Unobserved": "lightgrey",
        "In RPS": "skyblue",
        "Observed": "orange",
        "Observed & In RPS": "crimson"
    }
    label_vec = [label_names[l] for l in labels]

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=label_vec, palette=colors, s=80, edgecolor="black")
    plt.title("Policy Coverage Map (2D Projection)")
    plt.legend(title="Policy Status", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# --------- RPS Diagnostics ---------
def plot_rps_loss_histogram(partition_losses, map_idx=None, bins=30):
    """
    Plot histogram of partition losses in the Rashomon set.
    Optionally highlight MAP loss.
    """
    plt.figure(figsize=(8, 4))
    sns.histplot(partition_losses, bins=bins, kde=False, color="skyblue", edgecolor="white")
    if map_idx is not None:
        plt.axvline(partition_losses[map_idx], color='red', linestyle='--', label='MAP Loss')
        plt.legend()
    plt.xlabel("Partition Loss")
    plt.ylabel("Count")
    plt.title("Distribution of Losses in Rashomon Set")
    plt.tight_layout()
    plt.show()

def plot_rps_loss_histogram_v2(partition_losses, map_idx=None, bins=30, show_kde=False):
    """
    Plot histogram of partition losses in the Rashomon set, optionally showing the MAP loss.

    Args:
        partition_losses (np.ndarray): loss values for each partition
        map_idx (int, optional): index of MAP partition to highlight
        bins (int): number of histogram bins
        show_kde (bool): whether to overlay a KDE
    """
    plt.figure(figsize=(8, 4))
    sns.histplot(partition_losses, bins=bins, kde=show_kde, color="steelblue", edgecolor="white", alpha=0.85)
    if map_idx is not None:
        plt.axvline(partition_losses[map_idx], color='crimson', linestyle='--', linewidth=2, label='MAP Loss')
        plt.legend()
    plt.xlabel("Partition Loss")
    plt.ylabel("Count")
    plt.title("Distribution of Losses in Rashomon Set")
    plt.tight_layout()
    plt.show()

def plot_rps_loss_curve(partition_losses, map_idx=None):
    """
    Line plot of partition losses sorted from smallest to largest.
    Useful to visualize curvature and concentration of low-loss regions.
    """
    sorted_losses = np.sort(partition_losses)
    plt.figure(figsize=(8, 4))
    sns.lineplot(x=np.arange(len(sorted_losses)), y=sorted_losses, color="dodgerblue")
    if map_idx is not None:
        map_loss = partition_losses[map_idx]
        map_rank = np.where(sorted_losses == map_loss)[0][0]
        plt.axvline(map_rank, color='crimson', linestyle='--', label='MAP Rank')
        plt.legend()
    plt.xlabel("Sorted Partition Index")
    plt.ylabel("Loss")
    plt.title("Sorted Partition Losses")
    plt.tight_layout()
    plt.show()


# ---- for RPS construction -----
def plot_rps_size_vs_epsilon(epsilons, rps_sizes):
    """
    Plot Rashomon set size as a function of epsilon (slack over minimum loss).
    """
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=epsilons, y=rps_sizes, marker="o")
    plt.xlabel("Epsilon (Slack)")
    plt.ylabel("Rashomon Set Size")
    plt.title("RPS Size vs. Tolerance")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_rps_size_vs_theta(thetas, rps_sizes, log_scale=False):
    """
    Plot Rashomon set size vs. theta threshold.
    """
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=thetas, y=rps_sizes, marker="o")
    plt.xlabel("Theta")
    plt.ylabel("Rashomon Set Size")
    plt.title("RPS Size vs. Theta Threshold")
    if log_scale:
        plt.xscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()