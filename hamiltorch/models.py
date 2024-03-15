import torch 
import numpy as np
import torch.nn as nn
from typing import Union
from torch.autograd import grad
from torchdyn.core import NeuralODE


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class NNgHMC(nn.Module):
    """
    simple model which aims to model the gradient of the Hamiltonian directly 
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


class HNNEnergyDeriv(nn.Module):
    """
    simple neural network that models the derivative of the hamiltonian energy. Explicitly,
    H(q,p) = U(q) + .5*p^Tp
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(HNNEnergyDeriv, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.potential_deriv = NNgHMC(input_dim = self.input_dim, output_dim=self.input_dim, hidden_dim=self.hidden_dim)
    def forward(self, x, *args, **kwargs):
        n = self.input_dim 
        q, p = x[..., :n], x[..., n:]
        dHdq = self.potential_deriv(q)
        return  torch.cat([1*p, -dHdq], -1)

class RMHNNEnergyDeriv(nn.Module):
    """
    simple neural network that models the derivative of the hamiltonian energy. Explicitly,
    H(q,p) = U(q) + .5*p^M(q)Tp
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(RMHNNEnergyDeriv, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.potential_deriv = NNgHMC(input_dim = self.input_dim, output_dim=self.input_dim, hidden_dim=self.hidden_dim)
    def forward(self, x, *args, **kwargs):
        n = self.input_dim  ### here it is both p, q concatenated
        state_space = x[..., :n] 
        dH = self.potential_deriv(state_space)
        dHdq, dHdp = dH[..., : n // 2], dH[..., n // 2: ]
        return  torch.cat([dHdp, -dHdq], - 1)


class PotentialFunction(nn.Module):
    """
    simple neural network that models the potential function U(q)
    since this is -log(p(q)) >= 0 we us the softplus

    """
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(PotentialFunction, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 1 
        self.hidden_dim = hidden_dim
        self.layer_1 = nn.Linear(in_features=self.input_dim, out_features = self.hidden_dim)
        self.layer_2 = nn.Linear(in_features=hidden_dim, out_features = self.output_dim)



    def forward(self, x, *args, **kwargs):
        return nn.Softplus()(self.layer_2(nn.Tanh()(self.layer_1(x))))





class PSD(nn.Module):
    '''A Neural Net which outputs a positive semi-definite matrix'''
    def __init__(self, input_dim, hidden_dim, diag_dim):
        super(PSD, self).__init__()
        self.diag_dim = diag_dim
        self.off_diag_dim = int(diag_dim * (diag_dim - 1) / 2)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, self.diag_dim + self.off_diag_dim)

        for l in [self.linear1, self.linear2]:
            nn.init.orthogonal_(l.weight) # use a principled initialization
        
        self.nonlinearity = nn.Tanh()

    def forward(self, q):

        bs = q.shape[0]
        h = self.nonlinearity( self.linear1(q) )
        diag, off_diag = torch.split(self.linear2(h), [self.diag_dim, self.off_diag_dim], dim=1)
        # diag = nn.functional.relu( self.linear4(h) )

        L = torch.diag_embed(nn.Softplus()(diag))

        ind = np.tril_indices(self.diag_dim, k=-1)
        flat_ind = np.ravel_multi_index(ind, (self.diag_dim, self.diag_dim))
        L = torch.flatten(L, start_dim=1)
        L[:, flat_ind] = off_diag
        L = torch.reshape(L, (bs, self.diag_dim, self.diag_dim))

        D = torch.bmm(L, L.permute(0, 2, 1))
        return D


class PSDPotential(nn.Module):
    '''A Neural Net which outputs a positive semi-definite matrix and potential'''
    def __init__(self, input_dim, hidden_dim, diag_dim):
        super(PSDPotential, self).__init__()
        self.diag_dim = diag_dim
 

        self.diag_dim = diag_dim
        self.off_diag_dim = int(diag_dim * (diag_dim - 1) / 2)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, self.diag_dim + self.off_diag_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)   
        self.nonlinearity = nn.Tanh()

    def forward(self, q):
        bs = q.shape[0]
        h = self.nonlinearity( self.linear1(q) )
        diag, off_diag = torch.split(self.linear2(h), [self.diag_dim, self.off_diag_dim], dim=-1)
        # diag = nn.functional.relu( self.linear4(h) )

        L = torch.diag_embed(nn.Softplus()(diag))

        ind = np.tril_indices(self.diag_dim, k=-1)
        flat_ind = np.ravel_multi_index(ind, (self.diag_dim, self.diag_dim))
        L = torch.flatten(L, start_dim=1)
        L[:, flat_ind] = off_diag
        L = torch.reshape(L, (bs, self.diag_dim, self.diag_dim))

        D = torch.bmm(L, L.permute(0, 2, 1))
        return D, nn.Softplus()(self.nonlinearity(self.linear3(h)))



class HNNEnergyExplicit(nn.Module):
    """
    simple neural network that models the hamiltonian energy Explicitly,
    H(q,p) 

    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(HNNEnergyExplicit, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 1 
        self.hidden_dim = hidden_dim
        self.layer_1 = nn.Linear(in_features=self.input_dim, out_features = self.hidden_dim)
        self.layer_2 = nn.Linear(in_features=hidden_dim, out_features = self.output_dim)



    def forward(self, x, *args, **kwargs):
        n = self.input_dim
        q, p = x[..., :n], x[..., n:]
        return nn.Softplus()(self.layer_2(nn.Tanh()(self.layer_1(q)))) + .5 * torch.square(p).sum(axis = -1)
    
class RMHNNEnergyExplicit(nn.Module):
    """
    simple neural network that models the hamiltonian energy Explicitly,
    H(q,p)  = U(q) + .5 * p^TM(q)p


    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super(RMHNNEnergyExplicit, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hamiltonian_components = PSDPotential(input_dim = input_dim, hidden_dim=hidden_dim, diag_dim=input_dim)


    def forward(self, x, *args, **kwargs):
        q, p = x[..., :self.input_dim], x[..., self.input_dim:]

        mass_matrix, potential = self.hamiltonian_components(q)
        kinetic = .5 * torch.bmm(p[:, None, :], torch.bmm(mass_matrix, p[:, :, None]))
        return potential + kinetic


    
class HNN(nn.Module):
    """
    for the very simple case of HMC
    """
    def __init__(self, Hamiltonian: HNNEnergyExplicit) -> None:
        super(HNN, self).__init__()
        self.H = Hamiltonian
    def forward(self, x, *args, **kwargs):
        n = self.H.input_dim 
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            gradH = grad(self.H(x).sum(), x, create_graph=True)[0]
        return torch.cat([gradH[..., n:], -1*gradH[..., :n]], -1).to(x)
    

class RMHNN(nn.Module):
    """
    for the very general case of riemannian manifold
    """
    def __init__(self, Hamiltonian: RMHNNEnergyExplicit) -> None:
        super(RMHNN, self).__init__()
        self.H = Hamiltonian
    def forward(self, x, *args, **kwargs):
        n = self.H.input_dim // 2 ### here the hamiltonian is expected to take in both q,p
        with torch.set_grad_enabled(True): 
            x = x.requires_grad_(True)
            gradH = grad(self.H(x).sum(), x, create_graph=True)[0]
        return torch.cat([gradH[..., n:2*n], -gradH[..., :n]], -1).to(x)

class HNNODE(nn.Module):
    def __init__(self, odefunc: Union[HNN,HNNEnergyDeriv], sensitivity="adjoint", solver="dopri5", atol=1e-3, rtol=1e-3) -> None:
        super(HNNODE, self).__init__()
        self.odefunc = odefunc
        self.neural_ode_layer = NeuralODE(self.odefunc, solver = solver, sensitivity=sensitivity, atol=atol, rtol=rtol)
    def forward(self, x, t, *args, **kwargs):
        return self.neural_ode_layer.forward(x, t)
    

class RMHNNODE(nn.Module):
    def __init__(self, odefunc: RMHNN, sensitivity="adjoint", solver="dopri5", atol=1e-3, rtol=1e-3) -> None:
        super(RMHNNODE, self).__init__()
        self.odefunc = odefunc
        self.neural_ode_layer = NeuralODE(self.odefunc, solver = solver, sensitivity=sensitivity, atol=atol, rtol=rtol)
    def forward(self, x, t, *args, **kwargs):
        return self.neural_ode_layer.forward(x, t)

class NNODEgHMC(nn.Module):
    def __init__(self, odefunc: HNNEnergyDeriv, sensitivity="adjoint", solver = "dopri5", atol=1e-3, rtol=1e-3) -> None:
        super(NNODEgHMC, self).__init__()
        self.odefunc = odefunc
        self.neural_ode_layer = NeuralODE(self.odefunc, solver = solver, sensitivity=sensitivity, atol=atol, rtol=rtol)
    def forward(self, x, t, *args, **kwargs):
        return self.neural_ode_layer.forward(x, t)


class NNODEgRMHMC(nn.Module):
    def __init__(self, odefunc: RMHNNEnergyDeriv, sensitivity="adjoint", solver = "dopri5", atol=1e-3, rtol=1e-3) -> None:
        super(NNODEgRMHMC, self).__init__()
        self.odefunc = odefunc
        self.neural_ode_layer = NeuralODE(self.odefunc, solver = solver, sensitivity=sensitivity, atol=atol, rtol=rtol)
    def forward(self, x, t, *args, **kwargs):
        return self.neural_ode_layer.forward(x, t)



def train(model: nn.Module, X, y, epochs = 10, lr = .01, loss_type = "l2"):
    # early_stopper = EarlyStopper(patience = 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("Training Surrogate Model")
     # Compute and print loss.
    if loss_type == "l2":
        loss_func = nn.MSELoss()
    else:
        raise ValueError
    for epoch in range(epochs):

        y_pred = model(X)
        loss = loss_func(y_pred, y)
        # if early_stopper.early_stop(loss):             
        #     break
       
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
    return model, epoch


def train_ode(model: nn.Module, X, y, t,  epochs = 10, lr = .01, loss_type = "l2", gradient_traj = None):
    # early_stopper = EarlyStopper(patience = 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print("Training Surrogate ODE Model")
     # Compute and print loss.
    dims = y.shape[-1]
    if loss_type == "l2":
        loss_func = nn.MSELoss()
        
    else:
        raise ValueError
    for epoch in range(epochs):
        _, y_pred = model(X, t)
        loss = loss_func(torch.swapaxes(y_pred, 0, 1)[..., :dims], y)
        if gradient_traj is not None:
            observed_flattened = torch.flatten(gradient_traj, end_dim = -2)
            input_flattened = torch.flatten(y, end_dim = -2)
            gradient_loss = loss_func(model.odefunc(input_flattened)[..., dims // 2 : ], observed_flattened)
        else:
            gradient_loss = 0.0


        total_loss = gradient_loss + loss
        # if early_stopper.early_stop(total_loss):             
        #         break
        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        total_loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()
    return model, epoch
