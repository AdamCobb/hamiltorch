import math
import os
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import numpy as np
import torch
import pandas as pd
from typing import Dict


def plot_ess_vs_training_size(csv_path: str, output_path: str = "experiments/ess_vs_training_size.png"):
    """Plot ESS/s vs fraction of burn-in used for training, one panel per distribution."""
    df = pd.read_csv(csv_path)

    # ESS/s = ess / total_time
    df["ess_per_sec"] = df["ess"] / df["time"].clip(lower=1e-6)

    distributions = df["distribution"].unique()
    n_dist = len(distributions)
    # a 1xN strip is unreadable once placed in a two-column float; lay the
    # panels out in a grid with an aspect close to the page-width float
    ncols = 2 if n_dist > 2 else n_dist
    nrows = int(np.ceil(n_dist / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.4 * nrows), sharey=False)
    axes = np.atleast_1d(axes).ravel()
    for extra in range(n_dist, len(axes)):
        axes[extra].set_visible(False)

    model_order = ["HMC", "NNgHMC", "NNODEgHMC", "Explicit NNODEgHMC", "SymplecticNNgHMC", "GSymplecticNNgHMC"]
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(model_order)))

    for ax, dist in zip(axes, distributions):
        for model, color in zip(model_order, colors):
            subset = df[(df["distribution"] == dist) & (df["model"] == model)]
            if subset.empty:
                continue
            subset = subset.sort_values("training_size")
            label = model if model != "Explicit NNODEgHMC" else "HNNODEgHMC"
            # HMC doesn't vary with training size — draw as a horizontal dashed reference
            if model == "HMC":
                ax.axhline(subset["ess_per_sec"].mean(), color=color, linestyle="--",
                           linewidth=1.5, label=label, alpha=0.8)
            else:
                ax.plot(subset["training_size"] * 100, subset["ess_per_sec"],
                        marker="o", markersize=4, color=color, label=label)
        ax.set_title(dist.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Training samples (% of burn-in)")
        ax.set_ylabel("ESS / second")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Sample Efficiency: ESS/s vs Training Budget", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.clf()
    print(f"Saved ESS vs training size figure to {output_path}")


def plot_results(anchor_points, gradient_field_func, samples, trajectory, t, model_name = "", solver = "", sensitivity = "", distribution=""):
    _, ax = plt.subplots()

    # Add vectors V and W to the plot
    gradient_field = gradient_field_func(anchor_points)
    ax.quiver(anchor_points[:,0],  anchor_points[:,1], gradient_field[:,0], gradient_field[:,1], angles='xy', scale_units='xy', scale=.3, color='r')
    ax.scatter(samples[:, 0], samples[:, 1], alpha = .3, color = "blue")
    
    # 300 represents number of points to make between T.min and T.max
    xnew = np.linspace(t.min(), t.max(), 300)  
    power_smooth = CubicSpline(t, trajectory)(xnew)
    ax.plot(xnew, power_smooth)
    plt.grid()
    plt.title(f"Model: {model_name}, Solver: {solver}, Sensitivity: {sensitivity}")
    plt.savefig(f"../experiments/{model_name}_{solver}_{sensitivity}_{distribution}_full.png")
    # plt.show()


def plot_samples(sample_dict: Dict, mean, distribution_name=""):
    """
    dictionary of model name to samples

    """
    if torch.is_tensor(mean):
        mean = mean.detach().cpu().numpy()
    n = len(sample_dict)
    ncols = 3
    nrows = math.ceil(n / ncols)
    # sized for a page-width float: a 15in-wide source scaled into the column
    # renders its internal labels at a few points
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.9 * nrows),
                            sharex=True, sharey=True)
    for index, label in enumerate(sample_dict):
        samples = sample_dict[label]["samples"]
        axs.flat[index].scatter(samples[:,0].cpu(),samples[:,1].cpu(), s=4,alpha=0.3,label=label)
        axs.flat[index].scatter(mean[0],mean[1],marker = '*',color='C3',s=100,label='True Mean')
        axs.flat[index].set_title(f"{label}", fontsize=9)
    for index in range(n, nrows * ncols):
        axs.flat[index].set_visible(False)
    fig.suptitle(f"Samples from {distribution_name} Distribution", y=1.005)
    plt.tight_layout()
    suffix = "_smoke" if os.environ.get("HAMILTORCH_SMOKE", "0") == "1" else ""
    plt.savefig(f'experiments/{distribution_name}_samples{suffix}.png', bbox_inches='tight')
    # plt.show()



def plot_reversibility(sample_dict: Dict, samples, distribution = ""):
    n = len(sample_dict)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.9 * nrows),
                            sharex=True, sharey=True)

    # Add samples
    for index, label in enumerate(sample_dict):
        samples = sample_dict[label]["samples"]
        axs.flat[index].scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), alpha = .3, color = "green")
        forward_trajectories = sample_dict[label]["forward"]
        backward_trajectories = sample_dict[label]["backward"]
        if torch.is_tensor(forward_trajectories):
            forward_trajectories = forward_trajectories.detach().cpu().numpy()
        if torch.is_tensor(backward_trajectories):
            backward_trajectories = backward_trajectories.detach().cpu().numpy()
        num_samples = forward_trajectories.shape[0]
        # 300 represents number of points to make between T.min and T.max
        start = 0
        end = forward_trajectories.shape[1]
        t = np.linspace(start,end,num=end)
        xnew = np.linspace(t.min(),t.max() ,300)
        for i in range(num_samples):
            try:
                power_smooth_forward = CubicSpline(t, forward_trajectories[i])(xnew)
                power_smooth_backward = CubicSpline(t, backward_trajectories[i])(xnew)
                axs.flat[index].plot(power_smooth_forward[:,0], power_smooth_forward[:,1] ,label = "Forward Trajectory", color = "blue")
                axs.flat[index].plot(power_smooth_backward[:,0], power_smooth_backward[:,1], label = "Backward Trajectory", color = "red")
            except Exception:
                continue
        axs.flat[index].set_title(f"{label}", fontsize=9)
    for index in range(n, nrows * ncols):
        axs.flat[index].set_visible(False)
    # A single diverged trajectory otherwise sets the axis range and collapses
    # every other panel to a point: frame on the samples, not the trajectories.
    all_samples = np.concatenate(
        [sample_dict[k]["samples"][:, :2].detach().cpu().numpy() for k in sample_dict], axis=0)
    lo, hi = np.percentile(all_samples, [0.5, 99.5], axis=0)
    pad = 0.35 * (hi - lo + 1e-12)
    axs.flat[0].set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    axs.flat[0].set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    fig.suptitle(f"Reversibility of {distribution}", y=1.005)
    plt.tight_layout()
    suffix = "_smoke" if os.environ.get("HAMILTORCH_SMOKE", "0") == "1" else ""
    plt.savefig(f"experiments/{distribution}_reversibility{suffix}.png")
    # plt.show()
    plt.clf()







