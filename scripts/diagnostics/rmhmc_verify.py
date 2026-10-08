import torch
torch.set_default_dtype(torch.float64)
import hamiltorch
from hamiltorch.hmc import RMHMC
from hamiltorch.experiment_utils import banana_log_prob, normal_normal_conjugate
hamiltorch.set_random_seed(0)
D = 2
r = RMHMC(step_size=0.05, L=5, log_prob_func=normal_normal_conjugate, dim=D, softabs_const=10.)
q0 = torch.ones(D)
p0, pb0 = r.gibbs(q0, q0)

def T(q,p,qb,pb):
    tq,tp,_,qbn,pbn = r.step(q,p,qb,pb)
    return tq[-1], tp[-1], qbn, pbn
Rbar = lambda q,p,qb,pb: (q, -p, qb, -pb)

# (I) involutivity of Rbar o T on the EXTENDED state
q1,p1,qb1,pb1 = T(q0,p0,q0,pb0)
q2,p2,qb2,pb2 = T(*Rbar(q1,p1,qb1,pb1))
back = Rbar(q2,p2,qb2,pb2)
err = max(float((back[0]-q0).abs().max()), float((back[1]-p0).abs().max()),
          float((back[2]-q0).abs().max()), float((back[3]-pb0).abs().max()))
print(f"(I) involution error on extended state : {err:.2e}")

# compare against the OLD behaviour: project to (q,p), reset copies each time
def T_reset(q,p):
    tq,tp,_,_,_ = r.step(q,p,q.clone(),p.clone())
    return tq[-1], tp[-1]
a,b = T_reset(q0,p0); c,d = T_reset(a,-b)
print(f"    same, with copy-reset (old code)   : {float((c-q0).abs().max()):.2e}")

# (V) volume preservation of the extended map
z = torch.cat([q0,p0,q0,pb0])
def flat(z):
    q,p,qb,pb = z[:D], z[D:2*D], z[2*D:3*D], z[3*D:]
    a,b,c,d = T(q,p,qb,pb)
    return torch.cat([a,b,c,d])
J = torch.autograd.functional.jacobian(flat, z)
print(f"(V) |det J| - 1                        : {abs(float(torch.det(J).abs())-1):.2e}")

# does the chain recover the right marginal?
hamiltorch.set_random_seed(1)
traj,_,_,acc = r.sample(torch.ones(D), num_samples=1500)
s = traj[:,-1,:].detach()
print(f"\nacceptance {float(acc.mean()):.2f}; posterior mean {s.mean(0).numpy().round(3)}, sd {s.std(0).numpy().round(3)}")
