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

def make_multichain_warmup(base_sampler, init_sampler, n_chains, seed=0,
                           functional_trajectories=False, verbose=True):
    """Warm-up data pooled from many short chains, each started from a fresh
    draw of `init_sampler()`.

    Returns a callable(burn) -> (params, momenta, grads, accept) with exactly
    the shape a single `sampler.sample(...)` returns, so it can be assigned to
    a surrogate's `warmup_source`.

    This mirrors the training set of Glatt-Holtz et al. (2024), who fit NNgHMC
    to the advection-diffusion posterior using ~10k draws from many short
    chains rather than one long run. The distinction matters whenever the
    posterior is multimodal: a single chain started at one point may never
    leave its own mode -- and this target is bimodal by construction, with v*
    and -v* fitting the data identically -- so a single-chain surrogate is
    asked at sampling time to propose into regions it was never shown.

    `burn` is the total number of draws; it is divided as evenly as possible
    across `n_chains`, so chain length falls as the chain count rises.
    """
    def _source(burn):
        per = max(1, burn // n_chains)
        chains = max(1, min(n_chains, burn))
        P, M, G = [], [], []
        kw = {"functional_trajectories": True} if functional_trajectories else {}
        got = 0
        for c in range(chains):
            want = per if c < chains - 1 else max(1, burn - got)
            torch.manual_seed(seed + c)
            q0 = init_sampler()
            try:
                p, m, g, _ = base_sampler.sample(q0, num_samples=want, **kw)
            except Exception as exc:
                if verbose:
                    print(f"   multichain: chain {c} failed ({type(exc).__name__}), skipped",
                          flush=True)
                continue
            P.append(p.detach()); M.append(m.detach()); G.append(g.detach())
            got += want
        if not P:
            raise RuntimeError("every warm-up chain failed")
        params = torch.cat(P, 0); momenta = torch.cat(M, 0); grads = torch.cat(G, 0)
        if verbose:
            print(f"   multichain warm-up: {len(P)} chains x ~{per} draws "
                  f"-> {params.shape[0]} trajectories", flush=True)
        return params, momenta, grads, None
    return _source


def compute_hamiltonian_error(model, test_initial_conditions, t, log_prob_func):
    """Per-sample mean relative drift of H(q, p) = -log p(q) + |p|^2 / 2 along
    the trajectories the model produces from each initial condition.

    Every phase-space point is evaluated on its own, as in
    compute_rm_hamiltonian_error. That is forced by the log_prob convention in
    this codebase: each target sums over whatever batch it is handed, so a
    batched call returns one number. The previous implementation handed it the
    whole batch of initial conditions (one sum over N samples) and, through
    vmap, each trajectory (one sum over its L+1 points), then compared the
    two. The "error" that produced was the mismatch between those sums,
    (N - (L+1)) / N ~ 0.94 -- reported identically for exact leapfrog and for
    an identity map whose true drift is zero. Its vmap fallback also indexed
    the time-first trajectory array by sample, so on targets vmap cannot
    trace it compared time i of every sample against sample i.

    Rows whose trajectory cannot be evaluated (a LogProbError, or a diverged
    map returning non-finite values) are dropped, not misaligned. Returns the
    per-sample values, or NaN if none could be evaluated.
    """
    D = test_initial_conditions.shape[-1] // 2
    _, trajectories = model(test_initial_conditions, t)
    traj = torch.swapaxes(trajectories.detach(), 0, 1)  # (B, T, 2D)

    def H(x):
        return -log_prob_func(x[:D]).reshape(()) + 0.5 * (x[D:] ** 2).sum()

    errors = []
    with torch.no_grad():
        for b in range(traj.shape[0]):
            try:
                h = torch.stack([H(traj[b, i]) for i in range(traj.shape[1])])
            except Exception:
                continue
            if not torch.isfinite(h).all():
                continue
            errors.append(torch.mean(torch.abs(h - h[0]) / torch.clamp(h[0].abs(), min=1e-8)))
    if not errors:
        return torch.tensor(float("nan"))
    return torch.stack(errors)


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


def divergence_free_modes(k_max):
    """Half-plane enumeration of Fourier wavevectors with 0 < ||k||_2 <= k_max.

    One representative is kept per +/-k pair, since a real field determines
    v_{-k} from v_k; see the reality condition in Borggaard et al. (2020),
    eq. (2.2).
    """
    kc = int(math.floor(k_max))
    ks = [(kx, ky)
          for kx in range(-kc, kc + 1)
          for ky in range(-kc, kc + 1)
          if (kx, ky) != (0, 0)
          and kx * kx + ky * ky <= k_max * k_max
          and (kx > 0 or (kx == 0 and ky > 0))]
    ks.sort(key=lambda k: (k[0] ** 2 + k[1] ** 2, k[0], k[1]))
    return ks


def _kraichnan_spectrum(k, E0, xi, n_subfields):
    """Kraichnan energy spectrum, Borggaard et al. (2020) eq. (4.1).

    E(k) = E0 sum_i (k/k_i)^4 exp(-(3/2)(k/k_i)^2) k_i^{-xi},  k_i = 2^{i/2}.
    """
    total = 0.0
    for i in range(n_subfields + 1):
        ki = 2.0 ** (i / 2.0)
        r = k / ki
        total += (r ** 4) * math.exp(-1.5 * r * r) * ki ** (-xi)
    return E0 * total


def advection_diffusion_prior_std(k_max=8, xi=1.5, prior_scale=8.0, E0=None):
    """Per-coefficient prior standard deviations for the passive-scalar target.

    Pure arithmetic on the Kraichnan spectrum -- no PDE solve, no basis -- so
    callers can size and initialise a chain without paying to build the target.
    """
    two_pi = 2.0 * math.pi
    n_subfields = max(1, int(math.ceil(2.0 * math.log2(max(k_max, 2)))))
    if E0 is None:
        E0 = (prior_scale ** 2) / (_kraichnan_spectrum(1.0, 1.0, xi, n_subfields) / two_pi)
    var = []
    for (kx, ky) in divergence_free_modes(k_max):
        knorm = math.sqrt(kx * kx + ky * ky)
        v = E0 * _kraichnan_spectrum(knorm, 1.0, xi, n_subfields) / (two_pi * knorm)
        var.extend([v, v])                       # cos and sin share a wavenumber
    return torch.tensor(var).sqrt()


def make_advection_diffusion_log_prob(
        k_max=8, grid=32, kappa=3e-5, sigma_obs=0.125, n_sub=1,
        xi=1.5, prior_scale=8.0, E0=None, seed=0, dealias=True,
        dtype=torch.float32, cfl_target=1.2, max_sub=64):
    """Log posterior of the passive-scalar Bayesian inverse problem.

    Reproduces Example 2 of Borggaard, Glatt-Holtz and Krometis (2020), "A
    Bayesian Approach to Estimating Background Flows from a Passive Scalar"
    (SIAM/ASA JUQ 8(3)), which is the PDE inverse problem used as the
    surrogate-HMC benchmark in Glatt-Holtz, Holbrook, Krometis, Mondaini and
    Sheth (2024), "Sacred and Profane" (arXiv:2410.17398, Section 3.3).

    A passive solute obeys the advection-diffusion equation on the periodic
    torus T^2 = [0,1]^2,

        d_t theta + v . grad theta = kappa laplacian theta,   div v = 0,

    with known initial condition theta_0 = 1/2 - cos(2 pi x)/4 - cos(2 pi y)/4
    and known diffusivity kappa. The unknown is the background flow v, observed
    only through noisy point measurements of theta. The true flow is
    v* = [8 cos 2 pi y, 8 cos 2 pi x], observed at x1 = (0,0) and
    x2 = (1/2,1/2) at the 50 times t = 0.001, 0.002, ..., 0.050.

    The point of this target is its symmetry: at those two locations
    theta(v*, t, x_i) = theta(-v*, t, x_i), so v* and -v* explain the data
    identically, and the mean-zero prior gives them equal mass. The posterior
    is therefore multimodal by construction rather than by accident, which is
    what makes it hard to sample.

    Each log-density evaluation integrates the PDE over the full observation
    window, and the gradient differentiates through that solve --- the autograd
    analogue of the adjoint method of the source paper. This places the target
    firmly in the expensive-gradient regime that motivates surrogates: unlike
    the GP benchmark, where the cost is a single O(n^3) Cholesky, here it is a
    sequential time integration that no amount of parallelism collapses.

    Parameters follow the source where it reports them. Two do not appear
    there and are ours:

    ``prior_scale`` sets the overall prior energy E0 by calibration, so that a
    unit-wavenumber component has prior standard deviation ``prior_scale``.
    The source specifies the spectrum shape (4.1) but never reports E0, and the
    shape alone does not pin down the scale; the default makes the true flow's
    amplitude of 8 a one-sigma draw rather than an absurd one.

    ``grid``, ``n_sub`` and ``dealias`` are discretization choices. The source
    used a Julia spectral solver whose resolution it does not state.

    Returns (log_prob, info) where info carries the basis, the true
    coefficients and the simulated data, for tests and diagnostics.
    """
    device = torch.get_default_device()
    two_pi = 2.0 * math.pi
    # float64 throughout costs roughly 2x for no statistical gain: the
    # discretization error at this resolution is far below the observation
    # noise, and the +v*/-v* symmetry is exact by construction rather than
    # accumulated, so it survives single precision to ~1e-6.
    rdt, cdt = dtype, (torch.complex128 if dtype == torch.float64 else torch.complex64)

    # ---- divergence-free Fourier basis -------------------------------------
    # For a wavevector k the field (k_perp/||k||^2) * {cos,sin}(2 pi k.x) is
    # divergence free, since div(a f(k.x)) = (a.k) f'(k.x) and k_perp . k = 0.
    # The 1/||k||^2 normalisation is eq. (2.2) of the source; it makes the
    # ||k||=1 basis fields have unit amplitude, so the true flow's coefficients
    # are literally 8.
    modes = divergence_free_modes(k_max)
    xs = torch.arange(grid, dtype=torch.float64, device=device) / grid
    gx = xs.view(-1, 1).expand(grid, grid)
    gy = xs.view(1, -1).expand(grid, grid)

    basis, prior_var, labels = [], [], []
    n_subfields = max(1, int(math.ceil(2.0 * math.log2(max(k_max, 2)))))
    for (kx, ky) in modes:
        k2 = float(kx * kx + ky * ky)
        knorm = math.sqrt(k2)
        arg = two_pi * (kx * gx + ky * gy)
        # k_perp = [-ky, kx]
        perp = torch.tensor([-float(ky), float(kx)], dtype=torch.float64, device=device) / k2
        for phase, fn in (("cos", torch.cos), ("sin", torch.sin)):
            basis.append(perp.view(2, 1, 1) * fn(arg).unsqueeze(0))
            # eq. (4.2): diagonal covariance E(||k||)/(2 pi ||k||)
            prior_var.append(_kraichnan_spectrum(knorm, 1.0, xi, n_subfields)
                             / (two_pi * knorm))
            labels.append((kx, ky, phase))
    basis = torch.stack(basis).to(rdt)               # (M, 2, grid, grid)
    prior_var = torch.tensor(prior_var, dtype=torch.float64, device=device)

    # calibrate E0 so a unit-wavenumber component has std == prior_scale
    if E0 is None:
        unit_var = _kraichnan_spectrum(1.0, 1.0, xi, n_subfields) / two_pi
        E0 = (prior_scale ** 2) / unit_var
    prior_var = (prior_var * E0).to(rdt)
    n_params = basis.shape[0]

    # ---- true flow: v* = [8 cos 2 pi y, 8 cos 2 pi x] ----------------------
    # k=(1,0) has k_perp/||k||^2 = [0,1], giving [0, cos 2 pi x]; k=(0,1) has
    # [-1,0], giving [-cos 2 pi y, 0]. Hence coefficients +8 and -8.
    w_true = torch.zeros(n_params, dtype=rdt, device=device)
    w_true[labels.index((1, 0, "cos"))] = 8.0
    w_true[labels.index((0, 1, "cos"))] = -8.0

    # ---- spectral operators -------------------------------------------------
    freqs = torch.fft.fftfreq(grid, d=1.0 / grid).to(device=device, dtype=torch.float64)
    KX = freqs.view(-1, 1).expand(grid, grid)
    KY = freqs.view(1, -1).expand(grid, grid)
    lap = -((two_pi) ** 2) * (KX ** 2 + KY ** 2)
    # 2/3 rule: the advective product spreads energy to wavenumbers the grid
    # cannot represent, which folds back as aliasing error
    cutoff = grid / 3.0
    mask = ((KX.abs() < cutoff) & (KY.abs() < cutoff)).to(cdt) if dealias else None
    # pre-cast the spectral multipliers so the inner loop does no conversions
    ikx, iky, lapc = (1j * two_pi * KX).to(cdt), (1j * two_pi * KY).to(cdt), lap.to(cdt)

    theta0 = (0.5 - 0.25 * torch.cos(two_pi * gx) - 0.25 * torch.cos(two_pi * gy)).to(rdt)

    # observation design: 50 times, 2 locations that are exact grid nodes
    dt_out = 0.001
    t_obs = [dt_out * (j + 1) for j in range(50)]
    obs_idx = [(0, 0), (grid // 2, grid // 2)]

    def _velocity(w):
        # w: (..., M) -> (..., 2, grid, grid)
        return torch.einsum("...m,mcij->...cij", w, basis)

    def _rhs(theta, vx, vy):
        th = torch.fft.fft2(theta)
        # d/dx, d/dy and the Laplacian share one batched inverse transform
        d = torch.fft.ifft2(torch.stack((ikx * th, iky * th, lapc * th), dim=0)).real
        adv = vx * d[0] + vy * d[1]
        if mask is not None:
            adv = torch.fft.ifft2(torch.fft.fft2(adv) * mask).real
        return kappa * d[2] - adv

    def _solve(w):
        """Integrate the PDE and return observations, shape (..., 100)."""
        v = _velocity(w)
        vx, vy = v[..., 0, :, :], v[..., 1, :, :]
        # Explicit RK4 is stable only while the Courant number |v| dt / dx stays
        # below roughly 2.8. The prior is unbounded and a typical draw reaches
        # |v| ~ 50, not the true flow's 8, so a fixed step would diverge on such
        # draws and be rejected as a non-finite density -- silently truncating
        # the prior instead of sampling it. Choose the substep count from the
        # realised velocity instead. The count is a step function of w, so it is
        # constant in a neighbourhood of almost every point and does not disturb
        # the gradient.
        vmax = float(torch.maximum(vx.abs().max(), vy.abs().max()).detach())
        sub = max(n_sub, int(math.ceil(vmax * dt_out * grid / cfl_target)))
        if sub > max_sub:
            # only reachable in the far prior tail, where the posterior has no
            # mass; rejecting is then the honest response
            raise util.LogProbError()
        dt = dt_out / sub
        theta = theta0.expand(vx.shape).clone()
        out = []
        for _ in range(len(t_obs)):
            for _ in range(sub):
                # classical RK4; kappa is small enough at this resolution that
                # the diffusive term is not stiff and no integrating factor is
                # needed, while the CFL limit is set by |v| ~ 8
                k1 = _rhs(theta, vx, vy)
                k2 = _rhs(theta + 0.5 * dt * k1, vx, vy)
                k3 = _rhs(theta + 0.5 * dt * k2, vx, vy)
                k4 = _rhs(theta + dt * k3, vx, vy)
                theta = theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            out.append(torch.stack([theta[..., i, j] for (i, j) in obs_idx], dim=-1))
        return torch.cat(out, dim=-1)

    # ---- synthetic data -----------------------------------------------------
    with torch.no_grad():
        clean = _solve(w_true)
    g = torch.Generator(device="cpu").manual_seed(seed)
    # drawn on CPU with the explicit generator, as the GP target does, so the
    # data are identical whatever the ambient default device
    noise = torch.randn(clean.shape, generator=g, dtype=torch.float64,
                        device="cpu").to(device=device, dtype=rdt)
    Y = clean + sigma_obs * noise

    def log_prob(w):
        wc = w.to(rdt)
        try:
            pred = _solve(wc)
        except Exception:
            raise util.LogProbError()
        if not torch.isfinite(pred).all():
            raise util.LogProbError()
        resid = Y - pred
        ll = -0.5 * (resid ** 2).sum(-1) / (sigma_obs ** 2)
        prior = -0.5 * (wc ** 2 / prior_var).sum(-1)
        return (ll + prior).sum().to(w.dtype)

    info = {
        "n_params": n_params, "labels": labels, "w_true": w_true,
        "prior_var": prior_var, "Y": Y, "clean": clean, "basis": basis,
        "solve": _solve, "velocity": _velocity, "theta0": theta0,
        "E0": E0, "t_obs": t_obs, "obs_idx": obs_idx,
    }
    return log_prob, info


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
            # device pinned: with a CUDA default device, randperm would try to
            # draw on the GPU from this CPU generator and raise. Smoke runs
            # never reach this branch (their chains are shorter than max_n),
            # which is how it survived to kill two hour-old GPU runs.
            idx = torch.randperm(a.shape[0], generator=g, device="cpu")[:max_n]
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
