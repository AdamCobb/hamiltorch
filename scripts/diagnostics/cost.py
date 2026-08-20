import torch, time
torch.set_default_dtype(torch.float64)
import hamiltorch
from hamiltorch.models import HNNEnergyDeriv, HNNODE, NNgHMC
from hamiltorch.symplectic import GSymplecticNeuralNetwork, TimeSymmetricSymplectic
from hamiltorch.ode import SynchronousLeapfrog
hamiltorch.set_random_seed(0)
D = 3
z = torch.randn(1, 2*D)

def bench(fn, n=300):
    for _ in range(20): fn()          # warm up
    t0 = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter()-t0)/n * 1e3   # ms per proposal

print("Per-proposal cost: network evaluations needed to advance L leapfrog steps")
print(f"{'L':>4}{'NNODE (integrator)':>22}{'SympNet (flow map)':>22}{'RevSympNet':>14}{'speedup':>10}")
for L in [5, 10, 25, 50, 100]:
    eps = 0.1
    ode = HNNODE(HNNEnergyDeriv(input_dim=D, hidden_dim=100*D), solver=SynchronousLeapfrog(), sensitivity="autograd")
    g  = GSymplecticNeuralNetwork(dim=2*D, activation_modes=["up","down"]*4, widths=[D*100]*8)
    rev = TimeSymmetricSymplectic(g)
    t = torch.linspace(0, L*eps, L+1)
    T = torch.tensor(L*eps)
    f_ode = lambda: ode.forward(z, t)                 # L sequential integrator steps
    f_snn = lambda: g.step(z, T)                      # ONE evaluation
    f_rev = lambda: rev.step(z, T)                    # TWO evaluations
    with torch.no_grad():
        a, b, c = bench(f_ode), bench(f_snn), bench(f_rev)
    print(f"{L:>4}{a:>19.3f}ms{b:>19.3f}ms{c:>11.3f}ms{a/c:>9.1f}x")
print("\n(NNODE cost grows with L; flow-map cost is constant in L by construction)")
