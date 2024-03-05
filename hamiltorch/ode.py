import torch
from torchdyn.numerics.solvers.templates import DiffEqSolver
from torchdyn.numerics.solvers._constants import construct_rk4
from .models import HNN, RMHNN


class SynchronousLeapfrog(DiffEqSolver):
    def __init__(self, channel_index:int=-1, stepping_class:str='fixed', dtype=torch.float32, ):
        """Explicit Leapfrog symplectic ODE stepper for separable Hamiltonian
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
        q, p = xv[..., :half_state_dim], xv[..., half_state_dim:2*half_state_dim]
        dH = f(t, xv)
        dq, dp = dH[..., :half_state_dim], dH[..., half_state_dim:2*half_state_dim]
        q_new = q + dt * p  + .5 * torch.square(dt) * dp 

        dH_new = f(t, torch.cat([q_new, p], -1))
        dq_new, dp_new = dH_new[..., :half_state_dim], dH_new[..., half_state_dim:2*half_state_dim]
        p_new = p  + .5 * dt * (dp + dp_new)

        x_sol = torch.cat([q_new, p_new], -1)
        if self.stepping_class == 'adaptive':
            xv_err = torch.cat([torch.zeros_like(q), p], -1)
        else:
            xv_err = None
        return None, x_sol, xv_err





class NonSeparableSynchronousLeapfrog(DiffEqSolver):
    def __init__(self, channel_index:int=-1, stepping_class:str='fixed', dtype=torch.float32, binding_const = 100.):
        """Explicit Leapfrog symplectic ODE stepper for non-separable hamiltonians
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
        self.binding_const = binding_const


    def step(self, f:RMHNN , xv, t, dt, k1=None, args=None):
        """
        xv is a state vector of concatenated states, q,p, q_cop, p_cop
        """
        c = torch.cos(torch.FloatTensor([2* self.binding_const * dt]))
        s = torch.sin(torch.FloatTensor([2* self.binding_const * dt]))

        half_state_dim = xv.shape[-1] // 4

        q,p,q_cop,p_cop = torch.split(xv, half_state_dim, dim = -1)

        gradH = f(t, torch.cat([q,p_cop], -1))
        dq, dp = gradH[..., : half_state_dim], gradH[..., half_state_dim: 2*half_state_dim]

        p_new = p + .5 * dt * dp
        q_cop_new = q_cop + .5 * dt * dq

        gradH_new = f(t, torch.cat([q_cop_new, p_new], -1))
        dq_new, dp_new = gradH_new[..., : half_state_dim], gradH_new[..., half_state_dim:2*half_state_dim]

        q_new = q + .5 * dt * dq_new
        p_cop_new = p_cop + .5 * dt * dp_new

        q_avg = .5 * ((q_new + q_cop_new) + c*(q_new - q_cop_new) + s*(p_new - p_cop_new))
        p_avg = .5*( (p_new + p_cop_new)  - s* (q_new - q_cop_new) + c*(p_new - p_cop_new) )
        q_cop_avg = .5*((q_new + q_cop_new) - c* (q_new - q_cop_new) - s*(p_new - p_cop_new))
        p_cop_avg = .5*((p_cop_new + p_new) + s*(q_new - q_cop_new) -c * (p_new - p_cop_new))

        gradH_avg = f(t, torch.cat([q_avg, p_cop_avg], -1))
        dq_avg, dp_avg = gradH_avg[..., : half_state_dim], gradH_avg[..., half_state_dim: 2*half_state_dim]


        q_final = q_avg + .5 * dt * dq_avg
        p_cop_final = p_cop_avg + .5 * dt * dp_avg

        gradH_avg_new = f(t, torch.cat([q_final, p_cop_final], -1))
        dq_avg_new, dp_avg_new = gradH_avg_new[..., : half_state_dim], gradH_avg_new[..., half_state_dim: 2* half_state_dim]

        p_final = p_avg + .5 * dt * dp_avg_new
        q_cop_final = q_cop_avg + .5 * dt * dq_avg_new

        x_sol = torch.cat([q_final, p_final, q_cop_final, p_cop_final], -1)

        return None, x_sol, None