import torch
from torchdyn.numerics.solvers.templates import DiffEqSolver
from torchdyn.numerics.solvers.ode import SolverTemplate
from torchdyn.numerics.solvers._constants import construct_rk4
from .models import HNN


class SynchronousLeapfrog(DiffEqSolver):
    def __init__(self, channel_index:int=-1, stepping_class:str='fixed', dtype=torch.float32):
        """Explicit Leapfrog symplectic ODE stepper.
        Can return local error estimates if adaptive stepping is required"""
        super().__init__(order=2)
        self.dtype = dtype
        self.channel_index = channel_index
        self.stepping_class = stepping_class
        self.const = 1
        self.tableau = construct_rk4(self.dtype)
        # an additional overhead, necessary to preserve a certain degree of sanity
        # in the implementation and to avoid API bloating.
        self.x_shape = None


    def step(self, f:HNN , xv, t, dt, k1=None, args=None):
        half_state_dim = xv.shape[-1] // 2
        q, p = xv[..., :half_state_dim], xv[..., half_state_dim:]
        dH = f(t, xv)
        dq, dp = dH[..., :half_state_dim], dH[..., half_state_dim:]
        q_new = q + dt * p  + .5 * torch.square(dt) * dp 

        dH_new = f(t, torch.cat([q_new, p], -1))
        dq_new, dp_new = dH_new[..., :half_state_dim], dH_new[..., half_state_dim:]
        p_new = p  + .5 * dt * (dp + dp_new)

        x_sol = torch.cat([q_new, p_new], -1)
        if self.stepping_class == 'adaptive':
            xv_err = torch.cat([torch.zeros_like(q), p], -1)
        else:
            xv_err = None
        return None, x_sol, xv_err