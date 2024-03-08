import matplotlib.pyplot as plt 
from scipy.interpolate import CubicSpline
import numpy as np
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
    fig, axs = plt.subplots(1, 1, figsize=(15,15))
    for label in sample_dict:
        samples = sample_dict[label]
        axs.scatter(samples[:,0],samples[:,1], s=5,alpha=0.3,label=label)

    axs.scatter(mean[0],mean[1],marker = '*',color='C3',s=100,label='True Mean')
    axs.grid()
    plt.tight_layout()
    plt.savefig(f'../experiments/{distribution_name}_samples.png',bbox_inches='tight')
    # plt.show()



def plot_reversibility(forward_trajectories, backward_trajectories, samples, model_name ="", solver="", sensitivity="", distribution = ""):
    _, ax = plt.subplots()

    
    # Add samples 
    ax.scatter(samples[:, 0], samples[:, 1], alpha = .3, color = "blue")
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
            ax.plot(power_smooth_forward[:,0], power_smooth_forward[:,1] ,label = "Forward Trajectory", color = "blue")
            ax.plot(power_smooth_backward[:,0], power_smooth_backward[:,1], label = "Backward Trajectory", color = "red")
        except:
            continue
    plt.grid()
    plt.title(f"Model: {model_name}, Solver: {solver}, Sensitivity: {sensitivity}")
    plt.savefig(f"../experiments/{model_name}_{solver}_{sensitivity}_{distribution}_reversibility.png")
    # plt.show()
    plt.clf()







