
import gym
import torch 
import torch.nn as nn
from torch.distributions import Normal
from .samplers import approximate_leapfrog_hmc, hamiltonian, acceptance, gibbs
from .models import HNNODE
from gym import spaces


class HMCEnv(gym.Env):

    def __init__(self, dims, priors, log_prob_func, steps, step_size) -> None:
        self.dims = dims

        self.action_space = spaces.Box(low = -1*torch.ones(2 * self.dims)*torch.inf,
                                       high = torch.ones(self.dims * 2) * torch.inf)
        self.observation_space = spaces.Box(low = -1*torch.ones(2 * self.dims)*torch.inf,
                                       high = torch.ones(self.dims * 2) * torch.inf)
        self.priors = priors
        self.log_prob_func = log_prob_func
        self.steps = steps
        self.step_size = step_size
        

    def _get_obs(self):
        return {"position": self.p, "momentum": self.q}
    
    def _get_info(self):
        return {"hamiltonian": hamiltonian(self.p, self.q, self.log_prob_func)}
    
    def sample_momentum(self):
        return Normal().sample_n(self.dims)

    def sample_position(self):
        return torch.cat([prior.sample() for prior in self.priors]) if self.priors else self.sample_momentum()

    def reset(self, seed=None, options=None):
        super().reset(seed = seed)

        self.q = self.sample_position()
        self.p = self.sample_momentum()

        return self.q , self.p
    
    def step(self, action_func):
        #### my action here is the gradient of p, q
        ### a step would involve moving forward using ODE dynamics with a leapfrog integrator
        _, dp  = action_func(torch.cat[self.q, self.p])
        v1 = self.p + dp * self.step_size * .5
        x1 = self.q + v1*self.step_size
        v2 = v1 + self.step_size * .5 * action_func(torch.cat[x1, v1])[1]
        self.q = x1
        self.p = v2






    