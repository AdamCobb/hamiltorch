import torch 
import torch.nn as nn
from torch.autograd import grad
from torchdyn.core import NeuralODE

class NNgHMC(nn.Module):
    """
    simple model which aims to model the gradient of the log likelihood directly 
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super(NNgHMC, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.layer_1 = nn.Linear(in_features=self.input_dim, out_features = self.hidden_dim)
        self.layer_2 = nn.Linear(in_features=self.hidden_dim, out_features = self.output_dim)

    def forward(self, x):
        return self.layer_2(nn.Tanh()(self.layer_1(x)))
    

class NNmRHMC(nn.Module):
    """
    simple model which aims to model the mass matrix (fisher curvature) directly
    by using a cholesky decomposition
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(NNmRHMC, self).__init__()
        self.manifold_dim = input_dim
        self.parameter_dim = self.manifold_dim*(self.manifold_dim + 1)/2
        self.hidden_dim = hidden_dim

    def forward(self, x):
        untransformed_output = nn.Linear(in_features=self.hidden_dim, out_features = self.parameter_dim)(nn.Tanh()(
            nn.Linear(in_features=self.manifold_dim, out_features = self.hidden_dim)(x)
            ))
        m = torch.zeros((self.manifold_dim, self.manifold_dim))
        tril_indices = torch.tril_indices(row=self.manifold_dim, col=self.manifold_dim, offset=0)
        m[:,tril_indices[0], tril_indices[1]] = untransformed_output
        v = nn.Softplus()(torch.diag(m))
        mask = torch.diag(torch.ones_like(v))
        lower_tri = mask*torch.diag(v) + (1. - mask)*m
        return torch.einsum("...ij,...jk -> ...ik", lower_tri, lower_tri)



class NNEnergy(nn.Module):
    """
    simple neural network that models the hamiltonian energy,
    need it to always be positive
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(NNEnergy, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 1
        self.hidden_dim = hidden_dim

        self.layer_1 = nn.Linear(in_features=self.input_dim, out_features = self.hidden_dim)
        self.layer_2 = nn.Linear(in_features=self.hidden_dim, out_features = self.output_dim)


    def forward(self, x, *args, **kwargs):
        return nn.Softplus()(self.layer_2(nn.Tanh()(self.layer_1(x))))


class NNEnergyExplicit(nn.Module):
    """
    simple neural network that models the hamiltonian energy Explicitly,
    H(q,p) = U(q) + .5*p^TM(q)p
    here we are enforcing that the mass matrix is diagonal 
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(NNEnergyExplicit, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 1 + input_dim // 2
        self.hidden_dim = hidden_dim
        self.layer_1 = nn.Linear(in_features=self.input_dim // 2, out_features = self.hidden_dim)
        self.layer_2 = nn.Linear(in_features=hidden_dim, out_features = self.output_dim)



    def forward(self, x, *args, **kwargs):
        n = self.input_dim // 2
        q, p = torch.split(x, n, 1)
        output = nn.Softplus()(self.layer_2(nn.Tanh()(self.layer_1(q))))
        potential = output[:,0]
        mass = output[:, 1:]
        return  potential + .5 * torch.sum(mass * torch.pow(p, 2), dim = 1)



    
class HNN(nn.Module):
    def __init__(self, Hamiltonian: nn.Module) -> None:
        super(HNN, self).__init__()
        self.H = Hamiltonian
    def forward(self, t, x, *args, **kwargs):
        with torch.set_grad_enabled(True):
            n = x.shape[1] // 2
            x = x.requires_grad_(True)
            gradH = grad(self.H(x).sum(), x, create_graph=True)[0]
        return torch.cat([gradH[:, n:2*n], -gradH[:, :n]], 1).to(x)

class HNNODE(nn.Module):
    def __init__(self, odefunc: HNN, sensitivity="adjoint", solver="dopri5", atol=1e-3, rtol=1e-3) -> None:
        super(HNNODE, self).__init__()
        self.odefunc = odefunc
        self.neural_ode_layer = NeuralODE(self.odefunc, solver = solver, sensitivity=sensitivity, atol=atol, rtol=rtol)
    def forward(self, x, t, *args, **kwargs):
        return self.neural_ode_layer.forward(x, t)





def train(model: nn.Module, X, y, epochs = 10, lr = .01, loss_type = "l2"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("Training Surrogate Model")
     # Compute and print loss.
    if loss_type == "l2":
        loss_func = nn.MSELoss()
    else:
        raise ValueError
    for _ in range(epochs):

        y_pred = model(X)
        loss = loss_func(y_pred, y)
       
        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
    return model


def train_ode(model: nn.Module, X, y, t,  epochs = 10, lr = .01, loss_type = "l2"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("Training Surrogate ODE Model")
     # Compute and print loss.
    if loss_type == "l2":
        loss_func = nn.MSELoss()
    else:
        raise ValueError
    for i in range(epochs):
        _, y_pred = model(X, t)
        loss = loss_func(torch.swapaxes(y_pred, 0, 1), y)
       
        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
    return model
