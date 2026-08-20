import torch, numpy as np
torch.set_default_dtype(torch.float64)
import hamiltorch
from hamiltorch.hmc import RMHMC
# 3-D Gaussian: the metric is CONSTANT (= Sigma^-1), so RMHMC reduces to HMC
# with a fixed mass matrix and the answer is known exactly.
TRUE_SD = torch.tensor([0.5, 1.0, 2.0])
def gauss(w):
    return torch.distributions.MultivariateNormal(
        torch.zeros(3), torch.diag(TRUE_SD**2)).log_prob(w).sum()

print(f"truth: mean [0. 0. 0.]  sd {TRUE_SD.numpy()}")
print(f"{'variant':>26}{'accept':>9}{'mean':>26}{'sd':>26}{'max sd err':>12}")
import hamiltorch.hmc as H
orig = H.RMHMC.extended_hamiltonian
def make(half):
    def eh(self,q,p,qb,pb):
        b = 0.5*self.binding_const*(torch.square(q-qb).sum()+torch.square(p-pb).sum())
        tot = self.hamiltonian(q,pb)+self.hamiltonian(qb,p)+b
        return 0.5*tot if half else tot
    return eh
for label, half in [("consistent Hbar/2", True)]:
    H.RMHMC.extended_hamiltonian = make(half)
    hamiltorch.set_random_seed(2)
    r = H.RMHMC(step_size=0.15, L=10, log_prob_func=gauss, dim=3, softabs_const=1e6)
    tr,_,_,acc = r.sample(torch.zeros(3), num_samples=3000)
    s = tr[:,-1,:].detach()[500:]
    sd = s.std(0)
    print(f"{label:>26}{float(acc.mean()):>9.2f}{str(s.mean(0).numpy().round(3)):>26}"
          f"{str(sd.numpy().round(3)):>26}{float((sd-TRUE_SD).abs().max()):>12.3f}")
H.RMHMC.extended_hamiltonian = orig
