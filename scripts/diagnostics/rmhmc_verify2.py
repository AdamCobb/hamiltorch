import torch, numpy as np
torch.set_default_dtype(torch.float64)
import hamiltorch
from hamiltorch.hmc import RMHMC, HMC
from hamiltorch.experiment_utils import normal_normal_conjugate
hamiltorch.set_random_seed(0)
D = 2
r = RMHMC(step_size=0.05, L=5, log_prob_func=normal_normal_conjugate, dim=D, softabs_const=10.)
q0 = torch.ones(D); p0, pb0 = r.gibbs(q0, q0)

def flat(z):
    q,p,qb,pb = z[:D], z[D:2*D], z[2*D:3*D], z[3*D:]
    tq,tp,_,qbn,pbn = r.step(q,p,qb,pb)
    return torch.cat([tq[-1], tp[-1], qbn, pbn])

# _dH detaches, so autograd reports a singular Jacobian; use finite differences
z = torch.cat([q0,p0,q0,pb0]); h = 1e-6
J = torch.zeros(4*D, 4*D)
for i in range(4*D):
    e = torch.zeros(4*D); e[i] = h
    J[:, i] = (flat(z+e) - flat(z-e)) / (2*h)
print(f"(V) |det J| - 1  (finite differences) : {abs(float(torch.det(J).abs())-1):.2e}")

# marginal correctness: compare against a long reference HMC chain
hamiltorch.set_random_seed(1)
tr,_,_,acc = r.sample(torch.ones(D), num_samples=1200)
s_rm = tr[:,-1,:].detach()[200:]
hamiltorch.set_random_seed(1)
h_s = HMC(step_size=0.1, L=20, log_prob_func=normal_normal_conjugate, dim=D)
tr2,_,_,acc2 = h_s.sample(torch.ones(D), num_samples=3000)
s_hmc = tr2[:,-1,:].detach()[400:]
print(f"\n{'':>22}{'mean':>22}{'sd':>22}")
print(f"{'extended RMHMC':>22}{str(s_rm.mean(0).numpy().round(3)):>22}{str(s_rm.std(0).numpy().round(3)):>22}   accept {float(acc.mean()):.2f}")
print(f"{'reference HMC':>22}{str(s_hmc.mean(0).numpy().round(3)):>22}{str(s_hmc.std(0).numpy().round(3)):>22}   accept {float(acc2.mean()):.2f}")
d_mean = float((s_rm.mean(0)-s_hmc.mean(0)).abs().max())
d_sd = float((s_rm.std(0)-s_hmc.std(0)).abs().max())
print(f"\nmax |mean diff| = {d_mean:.3f}   max |sd diff| = {d_sd:.3f}")
