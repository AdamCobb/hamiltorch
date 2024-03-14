import matplotlib.pyplot as plt 
from scipy.interpolate import CubicSpline
import numpy as np
import torch
from typing import Dict


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
    fs=16
    fig, axs = plt.subplots(2, 2, figsize=(15,15), sharex=True, sharey=True)
    for index, label in enumerate(sample_dict):
        samples = torch.stack(sample_dict[label]["samples"],0)
        axs.flat[index].scatter(samples[:,0],samples[:,1], s=5,alpha=0.3,label=label)
        axs.flat[index].scatter(mean[0],mean[1],marker = '*',color='C3',s=100,label='True Mean')
        axs.flat[index].set_title(f"Model: {label}")
    fig.suptitle(f"Samples from {distribution_name} Distribution")
    plt.tight_layout()
    plt.savefig(f'../experiments/{distribution_name}_samples.png',bbox_inches='tight')
    # plt.show()



def plot_reversibility(sample_dict: Dict, samples, distribution = ""):
    fig, axs = plt.subplots(2, 2, figsize=(15,15), sharex=True, sharey=True)

    
    # Add samples 
    for index, label in enumerate(sample_dict):
        samples = torch.stack(sample_dict[label]["samples"],0)
        axs.flat[index].scatter(samples[:, 0], samples[:, 1], alpha = .3, color = "blue")
        forward_trajectories = sample_dict[label]["forward"]
        backward_trajectories = sample_dict[label]["backward"]
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
                
            except:
                continue
        axs.flat[index].set_title(f"Model: {label}")
    fig.suptitle(f"Reversibility of {distribution}")
    plt.savefig(f"../experiments/{distribution}_reversibility.png")
    # plt.show()
    plt.clf()







