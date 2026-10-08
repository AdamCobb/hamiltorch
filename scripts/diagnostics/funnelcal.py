import torch, hamiltorch
from hamiltorch.hmc import RMHMC
from hamiltorch.experiment_utils import funnel_log_prob
print("RMHMC on Neal's funnel — acceptance across (eps, softabs)")
print(f"{'eps':>7}{'L':>4}{'softabs':>9}{'accept':>9}{'v range':>22}")
for eps, L in [(0.1,5),(0.05,5),(0.02,10),(0.01,10)]:
    for sa in [1e1, 1e3]:
        hamiltorch.set_random_seed(0)
        r = RMHMC(step_size=eps, L=L, log_prob_func=funnel_log_prob, dim=2, softabs_const=sa)
        try:
            q,_,_,acc = r.sample(torch.Tensor([0., 1.]), num_samples=60)
            s = q[:,-1,:].detach()
            vr = f"[{float(s[:,0].min()):.1f},{float(s[:,0].max()):.1f}]" if torch.isfinite(s).all() else "nonfinite"
            print(f"{eps:>7.3f}{L:>4}{sa:>9.0e}{float(acc.mean()):>9.2f}{vr:>22}")
        except Exception as e:
            print(f"{eps:>7.3f}{L:>4}{sa:>9.0e}   {type(e).__name__}: {str(e)[:40]}")
