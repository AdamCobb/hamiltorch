import torch
torch.set_default_dtype(torch.float64)
import hamiltorch
from hamiltorch.hmc import RMHMC
from hamiltorch.experiment_utils import banana_log_prob
hamiltorch.set_random_seed(0)
q0 = torch.tensor([0., 100.]); p0 = torch.tensor([0.35, -0.20])
print("fixed T=0.25, banana, omega=100")
print(f"{'eps':>8}{'L':>4}{'|dH|':>14}{'ratio vs prev':>16}")
prev=None
for eps,L in [(0.1,3),(0.05,5),(0.025,10),(0.0125,20),(0.00625,40)]:
    r = RMHMC(step_size=eps, L=L, log_prob_func=banana_log_prob, dim=2, softabs_const=10.)
    qs,ps,_,_,_ = r.step(q0.clone(), p0.clone())
    d = abs(float(r.hamiltonian(qs[-1],ps[-1]).detach()) - float(r.hamiltonian(qs[0],ps[0]).detach()))
    print(f"{eps:>8.5f}{L:>4}{d:>14.3e}{'' if prev is None else f'{prev/d:>16.2f}x'}")
    prev=d
