import torch
import torch.nn as nn
import hamiltorch

def gaussian_log_prob(omega):
    mean = torch.tensor([0.,0.,0.])
    stddev = torch.tensor([.5,1.,2.]) 
    return torch.distributions.MultivariateNormal(mean, torch.diag(stddev**2)).log_prob(omega).sum()

def banana_log_prob(w, a = 1, b = 1, c = 1):
    ll = -(1/200) * torch.square(a * w[0]) - .5 * torch.square(c*w[1] + b * torch.square(a * w[0]) - 100 * b)
    return ll.sum()


def high_dimensional_gaussian_log_prob(w, D):
    ll = torch.distributions.MultivariateNormal(torch.zeros(D), covariance_matrix=torch.diag(torch.ones(D))).log_prob(w).sum()
    return ll

def normal_normal_conjugate(w):
    mu0 = 0.0
    tau = 1.5 
    sigma = torch.exp(w[1]) + .001
    ll = torch.distributions.Normal(mu0 , tau).log_prob(w[0]).sum()
    ll += torch.distributions.InverseGamma(2, 3).log_prob(sigma).sum()
    ll += torch.distributions.Normal(1.7, sigma).log_prob(w[0]).sum()
    return ll
    


def compute_reversibility_error(model, test_initial_conditions, t):
    D = test_initial_conditions.shape[-1] // 2
    _, forward_trajectories = model(test_initial_conditions, t)
    forward_trajectories = torch.swapaxes(forward_trajectories, 0, 1)
    end_positions = forward_trajectories[:,-1,:]
    backward_conditions = torch.matmul(end_positions, torch.block_diag(torch.eye(D), -1*torch.eye(D)))
    _, backward_trajectories = model(backward_conditions , t)
    backward_trajectories = torch.swapaxes(backward_trajectories, 0, 1)
    loss = nn.MSELoss()(backward_trajectories[:, -1, :D].detach(), test_initial_conditions[..., :D].detach())
    return loss, forward_trajectories[..., :D].detach(), backward_trajectories[..., :D].detach()



def params_grad(p, log_prob_func):
    p = p.requires_grad_(True)
    grad = grad(log_prob_func(p), p, create_graph=False)[0]
    return grad