import torch
import numpy as np
import torch.nn as nn
from typing import Union, Tuple
from torch.autograd import grad
from torch.func import jacfwd
from torchdyn.core import NeuralODE
from .symplectic import SymplecticNeuralNetwork, GSymplecticNeuralNetwork


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
        dH = self.potential_deriv(x)
        dHdq, dHdp = dH[..., : n // 2], dH[..., n // 2: ]
        return  torch.cat([1*dHdp, -dHdq], - 1)


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
        return self.layer_2(nn.Tanh()(self.layer_1(x)))





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
        # potential head is linear: -log p(q) is unbounded, so no squashing
        return D, self.linear3(h)



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
        # squeeze the trailing potential dim: (N,1) + (N,) would broadcast to
        # (N,N), scaling the whole learned vector field by the batch size
        potential = self.layer_2(nn.Tanh()(self.layer_1(q))).squeeze(-1)
        return potential + .5 * torch.square(p).sum(axis = -1)
    
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
        kinetic = .5 * torch.bmm(p[:, None, :], torch.bmm(mass_matrix, p[:, :, None])).squeeze(-1)
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
        # RMHNNEnergyExplicit.input_dim is the position dimension D; x is (q, p)
        n = self.H.input_dim
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            gradH = grad(self.H(x).sum(), x, create_graph=True)[0]
        return torch.cat([gradH[..., n:], -gradH[..., :n]], -1).to(x)

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



def _make_loss(loss_type):
    if loss_type == "l2":
        return nn.MSELoss()
    raise ValueError(f"Unknown loss type: {loss_type}")


def _restore_best(model, best_state):
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train(model: nn.Module, X, y, epochs = 100, lr = .01, loss_type = "l2", patience = 25):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=max(2, patience // 4))
    print("Training Surrogate Model")
    loss_func = _make_loss(loss_type)
    early_stopper = EarlyStopper(patience=patience)
    best_loss, best_state = float("inf"), None
    for epoch in range(epochs):
        y_pred = model(X)
        loss = loss_func(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        loss_val = float(loss.detach())
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if early_stopper.early_stop(loss_val):
            break
    return _restore_best(model, best_state), epoch


def train_ode(model: nn.Module, X, y, t,  epochs = 100, lr = .01, loss_type = "l2", gradient_traj = None, patience = 25, gradient_mode = "momentum"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=max(2, patience // 4))
    print("Training Surrogate ODE Model")
    dims = y.shape[-1]
    loss_func = _make_loss(loss_type)
    early_stopper = EarlyStopper(patience=patience)
    best_loss, best_state = float("inf"), None
    for epoch in range(epochs):
        _, y_pred = model(X, t)
        loss = loss_func(torch.swapaxes(y_pred, 0, 1)[..., :dims], y)
        if gradient_traj is not None:
            observed_flattened = torch.flatten(gradient_traj, end_dim = -2)
            input_flattened = torch.flatten(y, end_dim = -2)
            field = model.odefunc(input_flattened)
            if gradient_mode == "full":
                # observed is the complete (dq/dt, dp/dt) field (non-separable H)
                gradient_loss = loss_func(field[..., :dims], observed_flattened)
            else:
                # observed is grad log p = dp/dt (separable H)
                gradient_loss = loss_func(field[..., dims // 2 : ], observed_flattened)
        else:
            gradient_loss = 0.0

        total_loss = gradient_loss + loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: trajectory loss {float(loss):.6f}")
        loss_val = float(total_loss.detach())
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if early_stopper.early_stop(loss_val):
            break
    return _restore_best(model, best_state), epoch



def train_symplectic(model: Union[SymplecticNeuralNetwork,GSymplecticNeuralNetwork], X, y, t, epochs = 300, lr = .01,
                     loss_type = "l2", gradient_traj = None, batch_size = 4096, patience = 20,
                     gradient_weight = "auto"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=max(2, patience // 4))
    print("Training Surrogate Symplectic Model")
    dims = y.shape[-1]
    D = dims // 2
    loss_func = _make_loss(loss_type)
    early_stopper = EarlyStopper(patience=patience)
    best_loss, best_state = float("inf"), None
    N = X.shape[0]
    dt0 = torch.zeros(1, device=X.device)
    # The gradient term is typically 3-35x the trajectory term at init, so an
    # unweighted sum lets it dominate the objective that actually determines
    # proposal quality. Rescale once so both start at comparable magnitude.
    grad_w = 1.0
    if gradient_traj is not None and gradient_weight == "auto":
        with torch.no_grad():
            n0 = min(N, batch_size)
            y0, g0 = y[:n0], gradient_traj[:n0]
            v0 = jacfwd(lambda dt: model.step(y0, dt))(dt0).squeeze(-1)
            if g0.shape[-1] == dims:
                gl0 = loss_func(v0, g0)
            else:
                gl0 = loss_func(v0[:, D:], g0) + loss_func(v0[:, :D], y0[:, D:])
            tl0 = loss_func(model.step(X[:n0], t[:n0]), y[:n0])
            # a near-identity init can make gl0 vanish; clamp so the ratio
            # stays a balancing factor rather than an explosion
            grad_w = float((tl0 / gl0.clamp(min=1e-12)).clamp(1e-3, 1e3))
        print(f"Gradient loss weight (auto-balanced): {grad_w:.4g}")
    elif gradient_weight != "auto":
        grad_w = float(gradient_weight)
    for epoch in range(epochs):
        perm = torch.randperm(N, device=X.device)
        epoch_loss, num_batches = 0.0, 0
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            X_batch, y_batch, t_batch = X[idx], y[idx], t[idx]
            y_pred = model.step(X_batch, t_batch)
            loss = loss_func(y_pred, y_batch)
            if gradient_traj is not None:
                # gradient_traj: (N, D) — observed dp/dt = grad log p at output points.
                # d(phi(x, t))/dt |_{t=0} gives the learned vector field at each
                # output point. dt is a scalar, so forward-mode (one JVP pass)
                # computes the full (B, 2D, 1) Jacobian; reverse-mode would
                # need a backward per output element and OOMs on large batches.
                velocity = jacfwd(lambda dt: model.step(y_batch, dt))(dt0).squeeze(-1)
                g_batch = gradient_traj[idx]
                if g_batch.shape[-1] == dims:
                    # full (dq/dt, dp/dt) field observed (non-separable RMHMC)
                    gradient_loss = loss_func(velocity, g_batch)
                else:
                    # separable case: dp/dt = grad log p is stored, and
                    # dq/dt = p comes free from the trajectory itself
                    gradient_loss = loss_func(velocity[:, D:], g_batch) \
                        + loss_func(velocity[:, :D], y_batch[:, D:])
            else:
                gradient_loss = 0.0
            total_loss = grad_w * gradient_loss + loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += float(total_loss.detach())
            num_batches += 1

        epoch_loss /= max(num_batches, 1)
        scheduler.step(epoch_loss)
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss {epoch_loss:.6f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if early_stopper.early_stop(epoch_loss):
            break
    return _restore_best(model, best_state), epoch



def _symplectic_pair_indices(T, pair_mode):
    """Index pairs (i, j), i < j, used to build flow-map training data.

    The sampler only ever queries phi(., L*eps), so the full O(L^2) set of
    offsets is not required to fit the map it actually uses.

      "all"        — every (i, j): O(L^2) pairs, learns phi at all offsets
      "from_start" — (0, j) for all j: O(L) pairs, all offsets from x_0
      "endpoint"   — (0, L) only: O(1) pairs, exactly the sampled offset
    """
    if pair_mode == "all":
        return torch.triu_indices(T, T, offset=1)
    if pair_mode == "from_start":
        return torch.zeros(T - 1, dtype=torch.long), torch.arange(1, T)
    if pair_mode == "endpoint":
        return torch.zeros(1, dtype=torch.long), torch.tensor([T - 1])
    raise ValueError(f"Unknown pair_mode: {pair_mode}")


def create_training_set_symplectic_with_gradients(
    X: torch.Tensor, G: torch.Tensor, pair_mode: str = "all"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Like create_training_set_symplectic but also returns the gradient at the output point j.

    G : (N, T, D//2) — gradient of U w.r.t. q at each trajectory step (dU/dq = -dp/dt).
    Returns input, output, time, and gradient tensors aligned by pair (i, j).
    """
    N, T, D = X.shape
    i, j = _symplectic_pair_indices(T, pair_mode)
    K = i.shape[0]

    input_tensor = X[:, i, :].reshape(N * K, D)
    output_tensor = X[:, j, :].reshape(N * K, D)
    grad_tensor = G[:, j, :].reshape(N * K, G.shape[-1])

    time = j - i
    time_tensor = torch.unsqueeze(torch.tile(time, dims=(N,)), dim=-1)
    return input_tensor, output_tensor, time_tensor, grad_tensor


def create_training_set_symplectic(X: torch.Tensor, pair_mode: str = "all") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    N, T, D = X.shape

    i, j = _symplectic_pair_indices(T, pair_mode)
    # Calculate the number of pairs
    K = i.shape[0]

    # Expand the tensor to match the shape of the index pairs


    # Use index pairs to select elements from the tensor
    input_tensor = X[:, i, :]
    output_tensor = X[:, j, :]

    # Reshape to flatten the first two dimensions
    input_tensor = input_tensor.reshape(N * K, D)
    output_tensor = output_tensor.reshape(N * K, D)

    ### get time lengths
    time = j - i 
    time_tensor = torch.unsqueeze(torch.tile(time, dims=(N,)), dim = -1)
    return input_tensor, output_tensor, time_tensor