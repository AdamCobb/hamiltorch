import matplotlib.pyplot as plt 
from scipy.interpolate import CubicSpline
import numpy as np


def plot_results(anchor_points, gradient_field, samples, trajectory, t):
    _, ax = plt.subplots()

    # Add vectors V and W to the plot
    ax.quiver(anchor_points[:,0],  anchor_points[:,1], gradient_field[:,0], gradient_field[:,1], angles='xy', scale_units='xy', scale=.3, color='r')
    ax.scatter(samples[:, 0], samples[:, 1], alpha = .3, color = "blue")
    
    # 300 represents number of points to make between T.min and T.max
    xnew = np.linspace(t.min(), t.max(), 300)  
    power_smooth = CubicSpline(t, trajectory)(xnew)
    ax.plot(xnew, power_smooth)
    plt.grid()
    plt.show()











