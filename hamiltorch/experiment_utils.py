import math
import torch
import torch.nn as nn
from . import util
from torch.autograd import grad as autograd_grad


def gaussian_log_prob(omega):
    mean = torch.tensor([0.,0.,0.])
    stddev = torch.tensor([.5,1.,2.])
    ll = torch.distributions.MultivariateNormal(mean, torch.diag(stddev**2)).log_prob(omega)
    return ll.sum()

def banana_log_prob(w, a = 1, b = 1, c = 1):
    ll = -(1/200) * torch.square(a * w[..., 0]) - .5 * torch.square(c*w[..., 1] + b * torch.square(a * w[..., 0]) - 100 * b)
    return ll.sum()

def high_dimensional_gaussian_log_prob(w, D):
    ll = torch.distributions.MultivariateNormal(torch.zeros(D), covariance_matrix=torch.diag(torch.ones(D))).log_prob(w)

    return ll.sum()

def normal_normal_conjugate(w):
    mu0 = 0.0
    tau = 1.5
    sigma = torch.exp(w[..., 1]) + .001
    ll = torch.distributions.Normal(mu0, tau).log_prob(w[..., 0])
    ll += torch.distributions.InverseGamma(2, 3).log_prob(sigma)
    ll += torch.distributions.Normal(1.7, sigma).log_prob(w[..., 0])
    return ll.sum()

def high_dimensional_warped_gaussian_log_prob(w, D, scales):
    mean = torch.zeros(D)
    cov = torch.diag(scales)
    ll = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov).log_prob(w)
    return ll.sum()
    


def compute_reversibility_error(model, test_initial_conditions, t):
    D = test_initial_conditions.shape[-1] // 2
    _, forward_trajectories = model(test_initial_conditions, t)
    forward_trajectories = torch.swapaxes(forward_trajectories, 0, 1)
    end_positions = forward_trajectories[:,-1,:]
    backward_conditions = torch.matmul(end_positions, torch.block_diag(torch.eye(D), -1*torch.eye(D)))
    _, backward_trajectories = model(backward_conditions , t)
    backward_trajectories = torch.swapaxes(backward_trajectories, 0, 1)
    loss = nn.MSELoss()(backward_trajectories[:, -1, :D].detach(), test_initial_conditions[..., :D].detach())
    return loss, forward_trajectories[..., :D].detach(), backward_trajectories[..., :D].detach()

def compute_hamiltonian_error(model, test_initial_conditions, t, log_prob_func):
    D = test_initial_conditions.shape[-1] // 2
    hamiltonian = lambda x: -1*log_prob_func(x[..., :D]) + .5 * torch.sum(torch.square(x[..., D:]),dim=-1)
    initial_hamiltonian_values = hamiltonian(test_initial_conditions)
    _, trajectories = model(test_initial_conditions, t)
    forward_trajectories = torch.swapaxes(trajectories.detach(), 0, 1)
    batched_hamiltonian = torch.vmap(hamiltonian)
    try:
        forward_hamiltonians = batched_hamiltonian(forward_trajectories)
    except:
        hamiltonians_list = []
        bad_index = []
        for i in range(test_initial_conditions.shape[0]):
            try:
                hamiltonians_list.append(hamiltonian(trajectories[i]))
            except:
                bad_index.append(i)
        forward_hamiltonians = torch.stack(hamiltonians_list, axis = 0)

        index = torch.ones(test_initial_conditions.shape[0], dtype=bool)
        index[bad_index] = False
        initial_hamiltonian_values = initial_hamiltonian_values[index]

    delta_hamiltonian = torch.abs(forward_hamiltonians - initial_hamiltonian_values[:, None]) / (initial_hamiltonian_values[:, None])
    return torch.mean(delta_hamiltonian, dim = -1)


def params_grad(p, log_prob_func):
    p = p.detach().requires_grad_(True)
    return autograd_grad(log_prob_func(p), p, create_graph=False)[0]






def funnel_log_prob(w):
    """Neal's funnel (2-D): v ~ N(0, 9), x ~ N(0, e^v).

    The canonical target where position-dependent curvature defeats plain HMC
    and Riemannian methods shine. The densities are written analytically rather
    than via torch.distributions: the conditional scale exp(v/2) underflows and
    v itself can leave the reals under an aggressive proposal, and the
    distribution classes validate their arguments and *raise* on both, which
    kills the chain instead of rejecting the draw. Non-finite values are
    surfaced as LogProbError so the sampler rejects them.
    """
    if not torch.isfinite(w).all():
        raise util.LogProbError()
    v, x = w[..., 0], w[..., 1]
    half_log_2pi = 0.5 * math.log(2.0 * math.pi)
    # log N(v; 0, 3)
    ll = -0.5 * (v / 3.0) ** 2 - math.log(3.0) - half_log_2pi
    # log N(x; 0, exp(v/2)) = -x^2 e^{-v} / 2 - v/2 - log sqrt(2 pi)
    ll = ll - 0.5 * torch.square(x) * torch.exp(-v) - 0.5 * v - half_log_2pi
    if not torch.isfinite(ll).all():
        raise util.LogProbError()
    return ll.sum()


def make_gp_regression_log_prob(num_data=500, num_features=4, seed=0):
    """Log posterior of GP regression hyperparameters, matching the benchmark
    of Li et al. (2019) so results are directly comparable to their reported
    speedups: n = 500 observations with 4 standard-normal features, a Matern
    kernel with smoothness nu = 3/2 fixed, and two sampled hyperparameters
    (log lengthscale, log noise variance).

    Each evaluation costs an O(n^3) Cholesky, which is the regime that makes a
    surrogate worthwhile; at smaller n the gradient is cheap enough that plain
    HMC wins outright.
    """
    # build the dataset on CPU with an explicit generator (reproducible and
    # device-independent), then move it to the ambient default device
    g = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.randn(num_data, num_features, generator=g, device="cpu")
    # squared euclidean distances -> pairwise distance matrix
    sq = torch.cdist(X, X, p=2.0) ** 2
    dist = torch.sqrt(torch.clamp(sq, min=1e-12))
    # draw y from a GP with known hyperparameters so the posterior is well posed
    root3 = torch.sqrt(torch.tensor(3.0, device="cpu"))
    l_true, noise_true = 1.0, 0.1
    K_true = (1 + root3 * dist / l_true) * torch.exp(-root3 * dist / l_true)
    K_true = K_true + noise_true * torch.eye(num_data, device="cpu")
    y = torch.linalg.cholesky(K_true) @ torch.randn(num_data, generator=g, device="cpu")
    dist = dist.to(torch.get_default_device())
    y = y.to(torch.get_default_device())

    def log_prob(w):
        log_l, log_noise = w[..., 0], w[..., 1]
        # trailing singleton dims so a batch of hyperparameters broadcasts
        # against the (n, n) distance matrix -> (B, n, n)
        l = torch.exp(log_l)[..., None, None]
        noise = torch.exp(log_noise)[..., None, None]
        r = root3.to(dist.device) * dist / l
        # jitter floor: as log_noise drifts down the Matern gram matrix becomes
        # numerically singular, and a hard failure would crash the chain rather
        # than being rejected
        K = (1 + r) * torch.exp(-r) + (noise + 1e-4) * torch.eye(num_data)
        try:
            chol = torch.linalg.cholesky(K)
        except Exception:
            raise util.LogProbError()
        if not torch.isfinite(chol).all():
            raise util.LogProbError()
        # MVN log density from the factor directly, skipping the distribution's
        # strict positive-definite validation
        y_b = y.expand(chol.shape[:-1])
        sol = torch.cholesky_solve(y_b.unsqueeze(-1), chol)
        quad = (y_b.unsqueeze(-1) * sol).sum(dim=(-2, -1))
        logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
        ll = -0.5 * (quad + logdet + num_data * math.log(2.0 * math.pi))
        prior = torch.distributions.Normal(0., 1.).log_prob(w).sum()
        return ll.sum() + prior

    return log_prob


def compute_rm_hamiltonian_error(model, test_initial_conditions, t, rm_hamiltonian_func):
    """Mean relative drift of a *Riemannian* Hamiltonian along model
    trajectories. rm_hamiltonian_func(q, p) evaluates the exact non-separable
    Hamiltonian for a single (unbatched) phase-space point; rows whose exact
    Hamiltonian cannot be evaluated (diverged trajectories) are skipped."""
    D = test_initial_conditions.shape[-1] // 2
    _, trajectories = model(test_initial_conditions, t)
    traj = torch.swapaxes(trajectories.detach(), 0, 1)  # (B, T, 2D)
    errors = []
    for b in range(traj.shape[0]):
        try:
            h = torch.stack([
                rm_hamiltonian_func(traj[b, i, :D], traj[b, i, D:]).reshape(())
                for i in range(traj.shape[1])
            ])
            errors.append(torch.mean(torch.abs((h - h[0]) / h[0])))
        except Exception:
            continue
    if not errors:
        return torch.tensor(float("nan"))
    return torch.stack(errors).mean()


def energy_distance(x, y, max_n=600, seed=0):
    """Two-sample energy distance between sample sets x and y.

    E = 2 E|X-Y| - E|X-X'| - E|Y-Y'|, which is zero if and only if the two
    distributions coincide. It is parameter-free (no kernel bandwidth), works
    in any dimension, and needs only pairwise distances.

    This exists because effective sample size cannot certify a sampler. A map
    with a large involution defect produces weakly autocorrelated paths --- ESS
    reads high --- while targeting the wrong distribution; we measured a
    surrogate posting 158 ESS/s against exact HMC's 48 at a reversibility error
    of 7e8. ESS measures autocorrelation, not correctness, so any reported ESS
    for a learned proposal needs a distributional check beside it.

    Both sample sets are subsampled to at most max_n rows to keep the O(n^2)
    distance matrices cheap; the estimator is unbiased under subsampling.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    def _prep(a):
        a = a.detach().cpu()
        if a.shape[0] > max_n:
            idx = torch.randperm(a.shape[0], generator=g)[:max_n]
            a = a[idx]
        return a.double()
    x, y = _prep(x), _prep(y)
    if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
        return float("nan")
    d_xy = torch.cdist(x, y).mean()
    d_xx = torch.cdist(x, x).mean()
    d_yy = torch.cdist(y, y).mean()
    return float(2 * d_xy - d_xx - d_yy)


def normalised_energy_distance(x, y, **kw):
    """Energy distance scaled by the reference set's own spread.

    The raw statistic carries the units of the target, so it is not comparable
    across the distributions in a sweep. Dividing by E|Y-Y'| gives a
    dimensionless number: 0 means the samples are indistinguishable from the
    reference, and order 1 means they are as far from it as two independent
    draws from the reference are from each other.
    """
    raw = energy_distance(x, y, **kw)
    if raw != raw:
        return float("nan")
    y_ = y.detach().cpu().double()
    scale = float(torch.cdist(y_, y_).mean())
    return raw / scale if scale > 0 else float("nan")
