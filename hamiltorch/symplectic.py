import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
### all implementation from https://arxiv.org/pdf/2001.03750.pdf SympNets

class Symmetric(nn.Module):
    def forward(self, X):
        return X.triu() + X.triu(1).transpose(-1, -2)

class SymplecticLinearBlock(nn.Module):

    def __init__(self, dim, channels: int) -> None:
        #### dim is the size of the space of the input space 2*D D = param space
        super(SymplecticLinearBlock, self).__init__()
        assert (dim % 2) == 0
        assert (channels % 2) == 0
        self.dim = dim
        self.param_dim = dim // 2

        self.channels = channels

        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(self.dim)) for _ in range(self.channels)])
        self.A = nn.ModuleList([nn.Linear(self.param_dim,self.param_dim, bias = False) for _ in range(self.channels)])
        for layer in self.A:
            parametrize.register_parametrization(layer, "weight", parametrization=Symmetric())
            # init the underlying (unparametrized) tensor — initializing
            # layer.weight would only touch the symmetrized temporary
            nn.init.orthogonal_(layer.parametrizations["weight"].original)
    def forward(self, z, dt) -> torch.Tensor:
        ### assume the first block is up, the second is down and they alternate
        #### dt is the step size
        mode = "up"

        final_result = z
        for bias, layer in zip(self.bias,self.A):
            q, p = torch.hsplit(final_result, 2)
            if mode == "up":
                final_result = torch.cat([q + layer(p)*dt, p], -1) + bias*dt
                mode = "down"
            elif mode == "down":
                final_result = torch.cat([q, p + layer(q)*dt], -1) + bias * dt
                mode = "up" ### alternate modes
        return final_result

    def inverse(self, z, dt) -> torch.Tensor:
        # exact inverse: undo the channel shears in reverse order
        modes = ["up", "down"] * (self.channels // 2)
        final_result = z
        for bias, layer, mode in reversed(list(zip(self.bias, self.A, modes))):
            final_result = final_result - bias * dt
            q, p = torch.hsplit(final_result, 2)
            if mode == "up":
                final_result = torch.cat([q - layer(p) * dt, p], -1)
            else:
                final_result = torch.cat([q, p - layer(q) * dt], -1)
        return final_result

class SymplecticActivation(nn.Module):
    def __init__(self, dim: int, mode: str) -> None:
        super(SymplecticActivation, self).__init__()
        assert (dim % 2) == 0
        assert (mode in ["up", "down"])
        self.dim = dim
        self.mode = mode
        self.param_dim = dim // 2
        self.activation = nn.SiLU()
        self.a = nn.Parameter(torch.ones(self.param_dim))
        # small init keeps the map near the identity at initialization
        nn.init.normal_(self.a, std=0.1)
    def forward(self, z, dt) -> torch.Tensor:
        q, p = torch.hsplit(z, 2)
        if self.mode == "up":
            return torch.cat([q, dt*self.activation(q) * self.a + p], -1)
        elif self.mode == "down":
            return torch.cat([dt*self.activation(p)*self.a + q, p], -1)

        else:
            return z

    def inverse(self, z, dt) -> torch.Tensor:
        q, p = torch.hsplit(z, 2)
        if self.mode == "up":
            return torch.cat([q, p - dt * self.activation(q) * self.a], -1)
        elif self.mode == "down":
            return torch.cat([q - dt * self.activation(p) * self.a, p], -1)
        else:
            return z


class LASymplecticBlock(nn.Module):
    def __init__(self, dim, activation_mode: str = "up", channels:int = 4) -> None:
        super(LASymplecticBlock, self).__init__()
        self.linear_block = SymplecticLinearBlock(dim = dim, channels=channels)
        self.activation_block = SymplecticActivation(dim = dim, mode = activation_mode)

    def forward(self, z, dt) -> torch.Tensor:
        return self.activation_block(self.linear_block(z, dt), dt)

    def inverse(self, z, dt) -> torch.Tensor:
        return self.linear_block.inverse(self.activation_block.inverse(z, dt), dt)


class SymplecticNeuralNetwork(nn.Module):
    def __init__(self, dim, activation_modes: list[str], channels: list[int]) -> None:
        super(SymplecticNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList([LASymplecticBlock(dim, activation_mode, channel) for (activation_mode, channel) in zip (activation_modes, channels)])
        # Each shear multiplies the state by roughly (1 + ||A||*dt); with
        # orthogonal (unit-norm) init the full composition amplifies by
        # (1 + dt)^n_shears — 7e11 at dt=1.5 for the default 64 shears, which
        # destroys reversibility on targets with larger step sizes. Rescale so
        # the composition starts near the identity regardless of depth.
        n_shears = sum(len(block.linear_block.A) for block in self.layers)
        if n_shears > 0:
            with torch.no_grad():
                for block in self.layers:
                    for layer in block.linear_block.A:
                        layer.parametrizations["weight"].original.mul_(1.0 / n_shears)
                    for bias in block.linear_block.bias:
                        bias.mul_(1.0 / n_shears)
    def step(self, z, dt) -> torch.Tensor:
        for layer in self.layers:
            z = layer(z, dt)
        return z

    def inverse(self, z, dt) -> torch.Tensor:
        for layer in reversed(self.layers):
            z = layer.inverse(z, dt)
        return z

    def forward(self, z, t):
        """Trajectory at times t, each evaluated directly from the initial z.

        The network is trained on single applications (x_i, dt) -> x_j and the
        sampler proposes with a single step(z, L*eps); nothing enforces the
        composition property phi(.,2dt) = phi(.,dt)^2. Composing small steps
        here would measure an operator that is neither trained nor used for
        proposals, so the diagnostic would not describe the sampler.
        """
        preds = [self.step(z, dt) for dt in t]
        return t, torch.stack(preds, axis=0)



class GSymplecticBlock(nn.Module):
    def __init__(self, dim: int, width: int, mode: str) -> None:
        super(GSymplecticBlock, self).__init__()
        assert (dim % 2) == 0
        assert (mode in ["up", "down"])
        self.dim = dim
        self.mode = mode
        self.param_dim = dim // 2
        self.n = width
        self.K = nn.Parameter(torch.zeros(self.n, self.param_dim))
        self.a = nn.Parameter(torch.zeros(self.n))
        self.activation = nn.SiLU()
        self.bias = nn.Parameter(torch.zeros(self.n))

        ### initialize
        # small outer coefficients keep the map near the identity at init;
        # normal bias inside the activation still breaks symmetry
        nn.init.normal_(self.a, std=0.1)
        nn.init.normal_(self.bias)
        nn.init.orthogonal_(self.K)
    
    def forward(self, z, dt):
        q, p = torch.hsplit(z, 2)
        pre_activation_term = torch.transpose(self.K, 0, 1) * self.a
        if self.mode == "up":
            post_activation_term = self.activation(torch.einsum("ik,...k->...i",self.K, q) + self.bias)
            multiplier = torch.einsum("ik,...k->...i",pre_activation_term, post_activation_term)
            return torch.cat([q, dt*multiplier + p], -1)
        elif self.mode == "down":
            post_activation_term = self.activation(torch.einsum("ik,...k->...i",self.K, p) + self.bias)
            multiplier = torch.einsum("ik,...k->...i",pre_activation_term, post_activation_term)
            return torch.cat([q + dt * multiplier , p], -1)

        else:
            return z

    def inverse(self, z, dt):
        q, p = torch.hsplit(z, 2)
        pre_activation_term = torch.transpose(self.K, 0, 1) * self.a
        if self.mode == "up":
            post_activation_term = self.activation(torch.einsum("ik,...k->...i", self.K, q) + self.bias)
            multiplier = torch.einsum("ik,...k->...i", pre_activation_term, post_activation_term)
            return torch.cat([q, p - dt * multiplier], -1)
        elif self.mode == "down":
            post_activation_term = self.activation(torch.einsum("ik,...k->...i", self.K, p) + self.bias)
            multiplier = torch.einsum("ik,...k->...i", pre_activation_term, post_activation_term)
            return torch.cat([q - dt * multiplier, p], -1)
        else:
            return z

class GSymplecticNeuralNetwork(nn.Module):
    def __init__(self, dim, activation_modes: list[str], widths: list[int]) -> None:
        super(GSymplecticNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList([GSymplecticBlock(dim, width, activation_mode ) for (width, activation_mode) in zip (widths, activation_modes)])
    
    def step(self, z, dt) -> torch.Tensor:
        for layer in self.layers:
            z = layer(z, dt)
        return z

    def inverse(self, z, dt) -> torch.Tensor:
        for layer in reversed(self.layers):
            z = layer.inverse(z, dt)
        return z

    def forward(self, z, t):
        """Trajectory at times t, each evaluated directly from the initial z.

        The network is trained on single applications (x_i, dt) -> x_j and the
        sampler proposes with a single step(z, L*eps); nothing enforces the
        composition property phi(.,2dt) = phi(.,dt)^2. Composing small steps
        here would measure an operator that is neither trained nor used for
        proposals, so the diagnostic would not describe the sampler.
        """
        preds = [self.step(z, dt) for dt in t]
        return t, torch.stack(preds, axis=0)

    





class TimeSymmetricSymplectic(nn.Module):
    """Exactly momentum-reversible proposal built from an invertible symplectic net.

    Psi_t = (R . Phi_{t/2}^{-1} . R) . Phi_{t/2} with R the momentum flip.
    Then Psi_t^{-1} = R . Psi_t . R exactly, so an MH chain using Psi as its
    proposal satisfies detailed balance regardless of how well Phi is trained.
    If Phi equals the true (momentum-even) Hamiltonian flow, Psi_t = Phi_t.
    """

    def __init__(self, net) -> None:
        super(TimeSymmetricSymplectic, self).__init__()
        self.net = net

    @staticmethod
    def _flip(z):
        D = z.shape[-1] // 2
        return torch.cat([z[..., :D], -z[..., D:]], -1)

    def step(self, z, dt) -> torch.Tensor:
        half = dt * 0.5
        z = self.net.step(z, half)
        z = self._flip(z)
        z = self.net.inverse(z, half)
        return self._flip(z)

    def forward(self, z, t):
        """Trajectory at times t, each evaluated directly from the initial z.

        The network is trained on single applications (x_i, dt) -> x_j and the
        sampler proposes with a single step(z, L*eps); nothing enforces the
        composition property phi(.,2dt) = phi(.,dt)^2. Composing small steps
        here would measure an operator that is neither trained nor used for
        proposals, so the diagnostic would not describe the sampler.
        """
        preds = [self.step(z, dt) for dt in t]
        return t, torch.stack(preds, axis=0)
