import torch
import torch.nn as nn
import hamiltorch

def gaussian_log_prob(omega):
    mean = torch.tensor([0.,0.,0.])
    stddev = torch.tensor([.5,1.,2.]) 
    return torch.distributions.MultivariateNormal(mean, torch.diag(stddev**2)).log_prob(omega).sum()

def banana_log_prob(w, a = 1, b = 1, c = 1):
    ll = -(1/200) * torch.square(a * w[0]) - .5 * torch.square(c*w[1] + b * torch.square(a * w[0]) - 100 * b)
    return ll


def ill_conditioned_gaussian_log_prob(w, D):
    hamiltorch.set_random_seed(123)
    diagonal = torch.distributions.Uniform(0, 100).sample(sample_shape=(D,))
    diagonal[0] = .1
    diagonal[-1] = 1000
    ll = torch.distributions.MultivariateNormal(torch.zeros(D), covariance_matrix=torch.diag(diagonal)).log_prob(w).sum()
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