import os
import torch
import numpy as np
import hamiltorch
import arviz as az
from hamiltorch.hmc import (
    HMC, HMCGaussianAnalytic, SymplecticHMC, SurrogateGradientHMC, SurrogateNeuralODEHMC,
    RMHMC, SurrogateNeuralODERMHMC, SymplecticRMHMC,
)
from hamiltorch.ode import SynchronousLeapfrog
from hamiltorch.plot_utils import plot_reversibility, plot_samples
from hamiltorch.experiment_utils import (
    high_dimensional_warped_gaussian_log_prob, banana_log_prob, gaussian_log_prob,
    high_dimensional_gaussian_log_prob, compute_reversibility_error, params_grad,
    normal_normal_conjugate, compute_hamiltonian_error,
    funnel_log_prob, make_gp_regression_log_prob, compute_rm_hamiltonian_error,
    normalised_energy_distance, make_advection_diffusion_log_prob,
    make_multichain_warmup,
    advection_diffusion_prior_std,
)
from arviz import ess
import pandas as pd
import time


def _compute_ess(samples_tensor):
    """Compute mean ESS across parameters. Handles arviz >= 1.0 (DataTree API)."""
    arr = samples_tensor.cpu().numpy()[None, :, :]  # (1, draws, params)
    idata = az.from_dict({"posterior": {"x": arr}})
    return float(ess(idata).x.mean())


# Smoke mode (HAMILTORCH_SMOKE=1) shrinks chains/epochs so the full pipeline
# can be validated end-to-end in minutes rather than hours.
SMOKE = os.environ.get("HAMILTORCH_SMOKE", "0") == "1"
ODE_EPOCHS = 5 if SMOKE else 100
SNN_EPOCHS = 5 if SMOKE else 300
NN_EPOCHS = 5 if SMOKE else 100


def _out(path):
    """Route smoke-mode output to a separate file.

    Smoke runs previously wrote to the same paths as real runs, so validating
    the pipeline silently destroyed real results (this is how a genuine
    rmhmc_results.csv was replaced by 15 rows of 1-second chains).
    """
    if not SMOKE:
        return path
    stem, dot, ext = path.rpartition(".")
    return f"{stem}_smoke{dot}{ext}" if dot else path + "_smoke"


def _chain_lengths(experiment_params):
    """(burn, N) for the current mode."""
    if SMOKE:
        return 20, 40
    return experiment_params["burn"], experiment_params["N"]


# Glatt-Holtz et al. (2024) replication knobs. Their NNgHMC on this same
# advection-diffusion posterior used ~10k draws pooled from many short chains
# and 3-7 hidden layers; ours used one chain and one layer. Set by
# advection_diffusion_gh_experiment, read by run_experiment; empty means the
# original behaviour, so every other experiment is untouched.
GH_CONFIG = {"arch": None, "n_chains": 0, "train_kwargs": None,
             "snn_train_kwargs": None, "seed": 0}


def _gh_arch():
    """(hidden_dim, activations) for the configured architecture, or (None, None)."""
    from hamiltorch.models import GH_ARCHITECTURES
    if not GH_CONFIG["arch"]:
        return None, None
    return GH_ARCHITECTURES[GH_CONFIG["arch"]]


_GH_WARMUP_CACHE = {}


def _configure_gh(sampler, experiment_params, base_sampler):
    """Point a surrogate at pooled multi-chain warm-up data, when configured.

    The pooled draw is cached per (burn, n_chains, seed): generating it costs
    roughly as long as a full chain, and regenerating it for each of the 28
    (model, budget) blocks would dominate the run. Sharing one draw across
    models also makes the comparison cleaner -- every surrogate then sees
    identical training data, so differences are architecture, not draw noise.
    """
    if not (GH_CONFIG["n_chains"] and experiment_params.get("prior_draw") is not None):
        return

    def cached(burn):
        key = (burn, GH_CONFIG["n_chains"], GH_CONFIG["seed"])
        if key not in _GH_WARMUP_CACHE:
            print(f"   building pooled warm-up {key} (cached for later models)", flush=True)
            _GH_WARMUP_CACHE[key] = make_multichain_warmup(
                base_sampler, experiment_params["prior_draw"],
                n_chains=GH_CONFIG["n_chains"], seed=GH_CONFIG["seed"])(burn)
        return _GH_WARMUP_CACHE[key]

    sampler.warmup_source = cached


def _safe_plots(model_dict, initial_positions, distribution, mean):
    """Figures are diagnostics. A plotting failure -- a model whose
    reversibility trajectories could not be computed, a matplotlib error --
    must not abort a run whose results are already on disk."""
    try:
        plot_samples(model_dict, mean=mean, distribution_name=distribution)
    except Exception as exc:
        print(f"!! PLOT FAILED  samples  distribution={distribution}: "
              f"{type(exc).__name__}: {exc}", flush=True)
    try:
        # models whose reversibility diagnostic failed carry no trajectories
        plottable = {k: v for k, v in model_dict.items() if v.get("forward") is not None}
        if plottable:
            plot_reversibility(plottable, initial_positions, distribution=distribution)
    except Exception as exc:
        print(f"!! PLOT FAILED  reversibility  distribution={distribution}: "
              f"{type(exc).__name__}: {exc}", flush=True)


hamiltorch.set_random_seed(13)
scales = 100 * torch.rand(30)
_gp_log_prob = make_gp_regression_log_prob(num_data=500, num_features=4)

# The passive-scalar target builds a Fourier basis and runs a PDE solve to
# generate its data, so it is constructed on first use rather than at import,
# which every other experiment would otherwise pay for.
_advdiff_cache = {}


def _advdiff_prior_draw(k_max, scale=0.5):
    """A fresh prior draw, used as a start point for each pooled warm-up chain.
    Independent draws are the whole point -- they are what let the pool reach
    both +v* and -v* (measured mode balance 0.62 across 50 chains, against 0.28
    for a single chain)."""
    std = advection_diffusion_prior_std(k_max)
    return lambda: scale * std * torch.randn(std.shape, device=std.device)


def _advdiff_log_prob(k_max):
    def _lp(w):
        if k_max not in _advdiff_cache:
            _advdiff_cache[k_max] = make_advection_diffusion_log_prob(k_max=k_max)
        return _advdiff_cache[k_max][0](w)
    return _lp


def _advdiff_init(k_max, scale=0.5, seed=7):
    """A modest prior draw. Starting from v = 0 leaves the chain far outside the
    posterior with a large initial gradient; a half-scale prior draw is inside
    the basin of one of the two modes."""
    std = advection_diffusion_prior_std(k_max)
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(std.shape, generator=g, device="cpu").to(std.device)
    return scale * std * z
experiment_hyperparams = {
    "banana": {
        "step_size": .1, "L": 5, "burn": 3000, "N": 6000,
        "params_init": torch.Tensor([0., 100.]),
        "log_prob": banana_log_prob,
        "grad_func": lambda p: params_grad(p, banana_log_prob),
    },
    "gaussian": {
        "step_size": .3, "L": 5, "burn": 1000, "N": 2000,
        "params_init": torch.zeros(3),
        "log_prob": gaussian_log_prob,
        "grad_func": lambda p: params_grad(p, gaussian_log_prob),
    },
    "high_dimensional_gaussian": {
        "step_size": .1, "L": 5, "burn": 3000, "N": 6000, "D": 30,
        "params_init": torch.randn(30),
        "log_prob": lambda omega: high_dimensional_gaussian_log_prob(omega, D=30),
        "grad_func": lambda p: params_grad(p, high_dimensional_gaussian_log_prob),
    },
    "normal_normal": {
        "step_size": .1, "L": 5, "burn": 3000, "N": 6000,
        "params_init": torch.ones(2),
        "log_prob": lambda omega: normal_normal_conjugate(omega),
        "grad_func": lambda p: params_grad(p, normal_normal_conjugate),
    },
    # matches Li et al. (2019): eps = .05, L = 20, 10k samples, n = 500
    "gp_regression": {
        "step_size": .05, "L": 20, "burn": 2000, "N": 10000,
        # start near the posterior mode (log l = 0, log noise = log 0.1); from
        # the origin the initial gradient is ~140 and the chain diverges
        "params_init": torch.Tensor([0., -2.3]),
        "log_prob": _gp_log_prob,
        "grad_func": lambda p: params_grad(p, _gp_log_prob),
    },
    # Passive-scalar PDE inversion, Borggaard et al. (2020) Example 4.2 -- the
    # benchmark used for surrogate HMC in Glatt-Holtz et al. (2024). The
    # posterior is bimodal by construction: v* and -v* fit the data identically.
    # eps = 0.2 is the smallest step at which the chain crosses between the two
    # modes; at eps <= 0.05 it stays in whichever mode it started in.
    "advection_diffusion": {
        "step_size": .2, "L": 10, "burn": 500, "N": 1000,
        "params_init": _advdiff_init(2),
        "log_prob": _advdiff_log_prob(2),
        "grad_func": lambda p: params_grad(p, _advdiff_log_prob(2)),
        "prior_draw": _advdiff_prior_draw(2),
    },
    # The full truncation of the source, ||k||_2 <= 8. 196 parameters against
    # the 30 of the largest target here, and a phase space of 392 dimensions for
    # a flow-map surrogate to represent; expect this to be hard.
    "advection_diffusion_full": {
        "step_size": .05, "L": 10, "burn": 500, "N": 1000,
        "params_init": _advdiff_init(8),
        "log_prob": _advdiff_log_prob(8),
        "grad_func": lambda p: params_grad(p, _advdiff_log_prob(8)),
        "prior_draw": _advdiff_prior_draw(8),
    },
    "high_dimensional_warped_gaussian": {
        "step_size": .1, "L": 5, "burn": 3000, "N": 6000, "D": 30,
        "params_init": torch.randn(30),
        "log_prob": lambda omega: high_dimensional_warped_gaussian_log_prob(omega, D=30, scales=scales),
        "grad_func": lambda p: params_grad(p, high_dimensional_warped_gaussian_log_prob),
    },
}


def run_experiment(model_type, sensitivity, distribution, solver, percent=1,
                   is_analytic=False, a=None, device="cuda", pair_mode="all"):
    hamiltorch.set_random_seed(123)
    print(f"Running experiment for: solver: {solver}, sensitivity: {sensitivity}, "
          f"distribution: {distribution}, model: {model_type}")
    experiment_params = experiment_hyperparams[distribution]
    log_prob = experiment_params["log_prob"]
    params_init = experiment_params["params_init"].to(device)

    dim = params_init.shape[0]
    step_size = experiment_params["step_size"]
    L = experiment_params["L"]
    burn, N = _chain_lengths(experiment_params)
    if solver == "SynchronousLeapfrog":
        solver = SynchronousLeapfrog()
    if model_type == "HMC":
        sampler = (HMC(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim)
                   if not is_analytic
                   else HMCGaussianAnalytic(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim, a=a))
        params_hmc, _, _, _ = sampler.sample(q_init=params_init, grad_func=None,
                                              num_samples=int(burn * percent))
        params_hmc, _, _, _ = sampler.sample(q_init=params_hmc[-1, -1, :], grad_func=None,
                                              num_samples=N - int(burn * percent))

        def model_func(x, t):
            step_results = sampler.step(x[..., :dim], x[..., dim:])
            return (None, torch.cat([step_results[0], step_results[1]], -1))

        return params_hmc, model_func, experiment_params["grad_func"]

    elif model_type == "NNgHMC":
        base_sampler = (HMC(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim)
                        if not is_analytic
                        else HMCGaussianAnalytic(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim, a=a))
        sampler = SurrogateGradientHMC(step_size=step_size, L=L, log_prob_func=log_prob,
                                        base_sampler=base_sampler, dim=dim)
        _configure_gh(sampler, experiment_params, base_sampler)
        sampler.create_surrogate(q_init=params_init, burn=int(burn * percent), epochs=NN_EPOCHS,
                                 hidden_dim=_gh_arch()[0], activations=_gh_arch()[1],
                                 train_kwargs=GH_CONFIG["train_kwargs"])
        params_out, _, _, _ = sampler.sample(q_init=None, num_samples=N - int(burn * percent))

        def model_func(x, t):
            step_results = base_sampler.step(x[..., :dim], x[..., dim:], sampler.model)
            return (None, torch.cat([step_results[0], step_results[1]], -1))

        return params_out, model_func, sampler.model

    elif model_type == "NNODEgHMC":
        base_sampler = (HMC(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim)
                        if not is_analytic
                        else HMCGaussianAnalytic(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim, a=a))
        sampler = SurrogateNeuralODEHMC(step_size=step_size, L=L, log_prob_func=log_prob,
                                         dim=dim, base_sampler=base_sampler, model_type="")
        _configure_gh(sampler, experiment_params, base_sampler)
        sampler.create_surrogate(q_init=params_init, burn=int(burn * percent), epochs=ODE_EPOCHS,
                                  solver=solver, sensitivity=sensitivity,
                                  hidden_dim=_gh_arch()[0], activations=_gh_arch()[1],
                                  train_kwargs=GH_CONFIG["train_kwargs"])
        params_out, _, _, _ = sampler.sample(q_init=None, num_samples=N - int(burn * percent))
        return params_out, sampler.model, sampler.model.odefunc

    elif model_type == "Explicit NNODEgHMC":
        base_sampler = (HMC(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim)
                        if not is_analytic
                        else HMCGaussianAnalytic(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim, a=a))
        sampler = SurrogateNeuralODEHMC(step_size=step_size, L=L, log_prob_func=log_prob,
                                         dim=dim, base_sampler=base_sampler,
                                         model_type="explicit_hamiltonian")
        _configure_gh(sampler, experiment_params, base_sampler)
        sampler.create_surrogate(q_init=params_init, burn=int(burn * percent), epochs=ODE_EPOCHS,
                                  solver=solver, sensitivity=sensitivity,
                                  hidden_dim=_gh_arch()[0], activations=_gh_arch()[1],
                                  train_kwargs=GH_CONFIG["train_kwargs"])
        params_out, _, _, _ = sampler.sample(q_init=None, num_samples=N - int(burn * percent))
        return params_out, sampler.model, sampler.model.odefunc

    elif model_type == "SymplecticNNgHMC":
        base_sampler = (HMC(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim)
                        if not is_analytic
                        else HMCGaussianAnalytic(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim, a=a))
        sampler = SymplecticHMC(step_size=step_size, L=L, log_prob_func=log_prob,
                                 dim=dim, base_sampler=base_sampler, model_type="LA")
        _configure_gh(sampler, experiment_params, base_sampler)
        sampler.create_surrogate(q_init=params_init, burn=int(burn * percent), epochs=SNN_EPOCHS,
                                 pair_mode=pair_mode,
                                 train_kwargs=GH_CONFIG["snn_train_kwargs"])
        params_out, _, _, _ = sampler.sample(num_samples=N - int(burn * percent), q_init=None)
        return params_out, sampler.model, None

    elif model_type == "GSymplecticNNgHMC":
        base_sampler = (HMC(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim)
                        if not is_analytic
                        else HMCGaussianAnalytic(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim, a=a))
        sampler = SymplecticHMC(step_size=step_size, L=L, log_prob_func=log_prob,
                                 dim=dim, base_sampler=base_sampler, model_type="GSymp")
        _configure_gh(sampler, experiment_params, base_sampler)
        sampler.create_surrogate(q_init=params_init, burn=int(burn * percent), epochs=SNN_EPOCHS,
                                 pair_mode=pair_mode,
                                 train_kwargs=GH_CONFIG["snn_train_kwargs"])
        params_out, _, _, _ = sampler.sample(num_samples=N - int(burn * percent), q_init=None)
        return params_out, sampler.model, None

    elif model_type == "GradSymplecticNNgHMC":
        base_sampler = (HMC(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim)
                        if not is_analytic
                        else HMCGaussianAnalytic(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim, a=a))
        sampler = SymplecticHMC(step_size=step_size, L=L, log_prob_func=log_prob,
                                 dim=dim, base_sampler=base_sampler, model_type="LA")
        _configure_gh(sampler, experiment_params, base_sampler)
        sampler.create_surrogate(q_init=params_init, burn=int(burn * percent), epochs=SNN_EPOCHS,
                                  use_gradient=True, pair_mode=pair_mode,
                                 train_kwargs=GH_CONFIG["snn_train_kwargs"])
        params_out, _, _, _ = sampler.sample(num_samples=N - int(burn * percent), q_init=None)
        return params_out, sampler.model, None

    elif model_type == "GradGSymplecticNNgHMC":
        base_sampler = (HMC(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim)
                        if not is_analytic
                        else HMCGaussianAnalytic(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim, a=a))
        sampler = SymplecticHMC(step_size=step_size, L=L, log_prob_func=log_prob,
                                 dim=dim, base_sampler=base_sampler, model_type="GSymp")
        _configure_gh(sampler, experiment_params, base_sampler)
        sampler.create_surrogate(q_init=params_init, burn=int(burn * percent), epochs=SNN_EPOCHS,
                                  use_gradient=True, pair_mode=pair_mode,
                                 train_kwargs=GH_CONFIG["snn_train_kwargs"])
        params_out, _, _, _ = sampler.sample(num_samples=N - int(burn * percent), q_init=None)
        return params_out, sampler.model, None

    elif model_type in ("RevGSymplecticNNgHMC", "RevGradGSymplecticNNgHMC"):
        # exactly momentum-reversible SNN proposal (time-symmetric composition)
        base_sampler = (HMC(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim)
                        if not is_analytic
                        else HMCGaussianAnalytic(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim, a=a))
        sampler = SymplecticHMC(step_size=step_size, L=L, log_prob_func=log_prob,
                                 dim=dim, base_sampler=base_sampler, model_type="RevGSymp")
        _configure_gh(sampler, experiment_params, base_sampler)
        sampler.create_surrogate(q_init=params_init, burn=int(burn * percent), epochs=SNN_EPOCHS,
                                  use_gradient=model_type.startswith("RevGrad"), pair_mode=pair_mode,
                                 train_kwargs=GH_CONFIG["snn_train_kwargs"])
        params_out, _, _, _ = sampler.sample(num_samples=N - int(burn * percent), q_init=None)
        return params_out, sampler.model, None


def snn_gradient_ablation_experiment(device: str = "cuda"):
    """Compare LA/G-Symplectic SNNs with and without gradient supervision."""
    distributions = ["banana", "gaussian", "high_dimensional_gaussian", "normal_normal"]
    models = [
        "HMC", "SymplecticNNgHMC", "GSymplecticNNgHMC",
        "GradSymplecticNNgHMC", "GradGSymplecticNNgHMC",
        "RevGSymplecticNNgHMC", "RevGradGSymplecticNNgHMC",
    ]
    percent_of_warmup = [1.0] if SMOKE else np.linspace(0.1, 1, 5)
    sensitivity = "autograd"
    solver = "SynchronousLeapfrog"
    error_list = []

    for percent in percent_of_warmup:
        for distribution in distributions:
            model_dict = {}
            for model_type in models:
                start = time.time()
                experiment_samples, experiment_model, _ = run_experiment(
                    model_type, sensitivity, distribution, solver, percent, device=device
                )
                model_dict[model_type] = {
                    "samples": experiment_samples[:, -1, :].detach(),
                    "model": experiment_model,
                    "time": time.time() - start,
                }

            true_samples = model_dict["HMC"]["samples"]
            hamiltorch.set_random_seed(1)
            num_eval = min(100, true_samples.shape[0])
            initial_momentum = torch.distributions.Normal(0, 1).sample(
                sample_shape=(num_eval, true_samples.shape[-1])
            )
            initial_positions = true_samples[
                torch.multinomial(torch.ones(true_samples.shape[0]),
                                  num_samples=num_eval, replacement=False), :
            ]
            initial_conditions = torch.cat([initial_positions, initial_momentum], -1)

            for model_type in model_dict:
                step_size = experiment_hyperparams[distribution]["step_size"]
                L = experiment_hyperparams[distribution]["L"]
                t_span = torch.linspace(0, L * step_size, L + 1)
                error, _, _ = compute_reversibility_error(
                    model_dict[model_type]["model"], initial_conditions, t=t_span
                )
                hamiltonian_error = compute_hamiltonian_error(
                    model_dict[model_type]["model"], initial_conditions, t=t_span,
                    log_prob_func=experiment_hyperparams[distribution]["log_prob"]
                )
                error_list.append({
                    "model": model_type,
                    "training_size": percent,
                    "distribution": distribution,
                    "reversibility_error": error.detach().cpu().numpy(),
                    "hamiltonian_error": hamiltonian_error.detach().cpu().numpy(),
                    "distribution_error": normalised_energy_distance(
                        model_dict[model_type]["samples"], true_samples),
                    "time": model_dict[model_type]["time"],
                    "ess": _compute_ess(model_dict[model_type]["samples"]),
                })

    pd.DataFrame(error_list).to_csv(_out("experiments/snn_gradient_ablation.csv"), index=False)
    print("SNN gradient ablation results saved to experiments/snn_gradient_ablation.csv")


def surrogate_neural_ode_hmc_sample_size_experiment(device="cuda", distributions=None,
                                                    output_csv="experiments/diagnostic_results.csv",
                                                    percents=None, models=None,
                                                    pair_mode="all"):
    if distributions is None:
        distributions = ["banana", "gaussian", "high_dimensional_gaussian", "normal_normal"]
    sensitivities = ["autograd"]
    solvers = ["SynchronousLeapfrog"]
    if models is None:
        models = [
            "HMC", "NNgHMC", "NNODEgHMC", "Explicit NNODEgHMC",
            "SymplecticNNgHMC", "GSymplecticNNgHMC",
            "GradSymplecticNNgHMC", "GradGSymplecticNNgHMC",
        ]
    percent_of_warmup = ([1.0] if SMOKE
                         else (percents if percents is not None else np.linspace(0.1, 1, 10)))
    error_list = []

    for percent in percent_of_warmup:
        for distribution in distributions:
            for sensitivity in sensitivities:
                for solver in solvers:
                    model_dict = {}
                    for model in models:
                        start = time.time()
                        experiment_samples, experiment_model, _ = run_experiment(
                            model, sensitivity, distribution, solver, percent, device=device,
                            pair_mode=pair_mode
                        )
                        model_dict[model] = {
                            "samples": experiment_samples[:, -1, :].detach(),
                            "model": experiment_model,
                            "time": time.time() - start,
                        }

                    true_samples = model_dict["HMC"]["samples"]
                    hamiltorch.set_random_seed(1)
                    num_samples = min(100, true_samples.shape[0])
                    initial_momentum = torch.distributions.Normal(0, 1).sample(
                        sample_shape=(num_samples, true_samples.shape[-1])
                    )
                    initial_positions = true_samples[
                        torch.multinomial(torch.ones(true_samples.shape[0]),
                                          num_samples=num_samples, replacement=False), :
                    ]
                    initial_conditions = torch.cat([initial_positions, initial_momentum], -1)

                    for model in model_dict:
                        step_size = experiment_hyperparams[distribution]["step_size"]
                        L = experiment_hyperparams[distribution]["L"]
                        t_span = torch.linspace(0, L * step_size, L + 1)
                        # The samples above are the expensive part; nothing below
                        # may prevent them reaching the checkpoint. Each diagnostic
                        # is isolated: a failure records NaN and says so loudly
                        # rather than discarding hours of sampling.
                        row = {
                            "model": model,
                            "training_size": percent,
                            "sensitivity": sensitivity,
                            "distribution": distribution,
                            "solver": solver,
                            "step_size": step_size,
                            "time": model_dict[model]["time"],
                        }
                        model_dict[model]["forward"] = model_dict[model]["backward"] = None
                        for key, fn in (
                            ("reversibility_error", lambda: compute_reversibility_error(
                                model_dict[model]["model"], initial_conditions, t=t_span)),
                            ("hamiltonian_error", lambda: compute_hamiltonian_error(
                                model_dict[model]["model"], initial_conditions, t=t_span,
                                log_prob_func=experiment_hyperparams[distribution]["log_prob"])),
                            # ESS cannot certify a sampler; this compares the
                            # sampled distribution against the exact chain
                            ("distribution_error", lambda: normalised_energy_distance(
                                model_dict[model]["samples"], true_samples)),
                            ("ess", lambda: _compute_ess(model_dict[model]["samples"])),
                        ):
                            try:
                                val = fn()
                                if key == "reversibility_error":
                                    val, fwd, bwd = val
                                    model_dict[model]["forward"] = fwd[:5]
                                    model_dict[model]["backward"] = bwd[:5]
                                row[key] = (val.detach().cpu().numpy()
                                            if torch.is_tensor(val) else val)
                            except Exception as exc:
                                print(f"!! DIAGNOSTIC FAILED  {key}  model={model}  "
                                      f"distribution={distribution}  percent={percent}\n"
                                      f"   {type(exc).__name__}: {exc}", flush=True)
                                row[key] = float("nan")
                        error_list.append(row)

                    # checkpoint after every block, and before plotting: a
                    # failure hours in should cost one block, not the run, and
                    # a figure must never stand between the data and the disk
                    pd.DataFrame(error_list).to_csv(_out(output_csv), index=False)
                    _safe_plots(model_dict, initial_positions, distribution,
                                experiment_hyperparams[distribution]["params_init"])

    pd.DataFrame(error_list).to_csv(_out(output_csv), index=False)


def gp_sample_size_experiment(device="cuda", percents=None):
    """GP benchmark. Includes the symmetrized flow map trained with endpoint
    pairs, which is the configuration trajectory_length identifies as the one
    that actually wins; the plain SympNets are retained only to show that their
    high ESS coexists with a reversibility error of order 1e9."""
    """Sample-size sweep on an expensive-likelihood target (GP regression
    hyperparameters, O(N^3) per gradient): the regime surrogate HMC targets."""
    surrogate_neural_ode_hmc_sample_size_experiment(
        device=device, distributions=["gp_regression"],
        output_csv="experiments/diagnostic_results_gp.csv",
        percents=percents if percents is not None else [0.1, 0.4, 0.7, 1.0],
        models=["HMC", "NNgHMC", "NNODEgHMC", "Explicit NNODEgHMC",
                "GSymplecticNNgHMC", "RevGSymplecticNNgHMC",
                "RevGradGSymplecticNNgHMC"],
        pair_mode="endpoint")


def advection_diffusion_experiment(device="cuda", percents=None, full=False):
    """Passive-scalar PDE inversion (Borggaard et al. 2020, Example 4.2), the
    surrogate-HMC benchmark of Glatt-Holtz et al. (2024). Same model list,
    training-budget sweep and pair construction as the GP benchmark, so the two
    expensive-likelihood targets are directly comparable.

    Where the GP's cost is one O(n^3) Cholesky per gradient, here it is a
    sequential time integration of the PDE, and the posterior is bimodal by
    construction (v* and -v* fit the data identically). Whether a surrogate
    trained on one mode's warm-up can propose into the other is the question
    this target asks that the GP cannot.

    full=True uses the source's ||k||_2 <= 8 truncation (196 parameters);
    the default ||k||_2 <= 2 (12 parameters) keeps both modes and is the size
    at which the surrogates have been shown to work elsewhere in this suite.
    """
    name = "advection_diffusion_full" if full else "advection_diffusion"
    surrogate_neural_ode_hmc_sample_size_experiment(
        device=device, distributions=[name],
        output_csv=f"experiments/diagnostic_results_{name}.csv",
        percents=percents if percents is not None else [0.1, 0.4, 0.7, 1.0],
        models=["HMC", "NNgHMC", "NNODEgHMC", "Explicit NNODEgHMC",
                "GSymplecticNNgHMC", "RevGSymplecticNNgHMC",
                "RevGradGSymplecticNNgHMC"],
        pair_mode="endpoint")


def advection_diffusion_gh_experiment(device="cuda", full=False, arch="medium",
                                      n_chains=50, percents=None):
    """The advection-diffusion benchmark under the conditions of Glatt-Holtz et
    al. (2024), run across our full surrogate suite.

    Three changes from `advection_diffusion_experiment`, each measured to matter
    on the held-out gradient fit (relative error, 12-parameter target):

      single chain, 1x100D, full-batch lr 1e-2   0.940   (the original run)
      multi-chain, 1x100D, full-batch            0.864
      multi-chain, 1x100D, minibatch + val-stop  0.603
      multi-chain, medium (5.3M), same protocol  0.492

    The training protocol matters more than the architecture: the same 30k
    network improves 0.864 -> 0.603 on minibatching alone, while depth under
    the old full-batch protocol was catastrophic (small: 1.000, i.e. no better
    than predicting zero) because 100 full-batch steps cannot fit 1M weights.
    """
    name = "advection_diffusion_full" if full else "advection_diffusion"
    GH_CONFIG.update({
        "arch": arch, "n_chains": n_chains, "seed": 7,
        "train_kwargs": dict(lr=1e-3, batch_size=512, val_frac=0.2, patience=100),
        # SympNets already minibatch at 4096 over the pair construction, a size
        # tuned for it; only the stopping rule changes, for protocol parity.
        "snn_train_kwargs": dict(val_frac=0.2, patience=100),
    })
    try:
        surrogate_neural_ode_hmc_sample_size_experiment(
            device=device, distributions=[name],
            output_csv=f"experiments/diagnostic_results_{name}_gh.csv",
            percents=percents if percents is not None else [0.1, 0.4, 0.7, 1.0],
            models=["HMC", "NNgHMC", "NNODEgHMC", "Explicit NNODEgHMC",
                    "GSymplecticNNgHMC", "RevGSymplecticNNgHMC",
                    "RevGradGSymplecticNNgHMC"],
            pair_mode="endpoint")
    finally:
        GH_CONFIG.update({"arch": None, "n_chains": 0, "train_kwargs": None,
                          "snn_train_kwargs": None})
        _GH_WARMUP_CACHE.clear()


def surrogate_neural_ode_hmc_sample_size_experiment_analytic():
    distributions = ["high_dimensional_gaussian", "high_dimensional_warped_gaussian"]
    sensitivities = ["autograd"]
    solvers = ["SynchronousLeapfrog"]
    models = [
        "HMC", "NNgHMC", "NNODEgHMC", "Explicit NNODEgHMC",
        "SymplecticNNgHMC", "GSymplecticNNgHMC",
    ]
    percent_of_warmup = [1.0] if SMOKE else np.linspace(0.1, 1, 10)
    error_list = []

    for percent in percent_of_warmup:
        for distribution in distributions:
            a = (torch.ones(experiment_hyperparams[distribution]["D"])
                 if distribution == "high_dimensional_gaussian"
                 else scales)
            for sensitivity in sensitivities:
                for solver in solvers:
                    model_dict = {}
                    for model in models:
                        start = time.time()
                        experiment_samples, experiment_model, _ = run_experiment(
                            model, sensitivity, distribution, solver, percent,
                            is_analytic=True, a=a,
                        )
                        model_dict[model] = {
                            "samples": experiment_samples[:, -1, :].detach(),
                            "model": experiment_model,
                            "time": time.time() - start,
                        }

                    true_samples = model_dict["HMC"]["samples"]
                    hamiltorch.set_random_seed(1)
                    num_samples = min(100, true_samples.shape[0])
                    initial_momentum = torch.distributions.Normal(0, 1).sample(
                        sample_shape=(num_samples, true_samples.shape[-1])
                    )
                    initial_positions = true_samples[
                        torch.multinomial(torch.ones(true_samples.shape[0]),
                                          num_samples=num_samples, replacement=False), :
                    ]
                    initial_conditions = torch.cat([initial_positions, initial_momentum], -1)

                    for model in model_dict:
                        step_size = experiment_hyperparams[distribution]["step_size"]
                        L = experiment_hyperparams[distribution]["L"]
                        t_span = torch.linspace(0, L * step_size, L + 1)
                        # The samples above are the expensive part; nothing below
                        # may prevent them reaching the checkpoint. Each diagnostic
                        # is isolated: a failure records NaN and says so loudly
                        # rather than discarding hours of sampling.
                        row = {
                            "model": model,
                            "training_size": percent,
                            "sensitivity": sensitivity,
                            "distribution": distribution,
                            "solver": solver,
                            "step_size": step_size,
                            "time": model_dict[model]["time"],
                        }
                        model_dict[model]["forward"] = model_dict[model]["backward"] = None
                        for key, fn in (
                            ("reversibility_error", lambda: compute_reversibility_error(
                                model_dict[model]["model"], initial_conditions, t=t_span)),
                            ("hamiltonian_error", lambda: compute_hamiltonian_error(
                                model_dict[model]["model"], initial_conditions, t=t_span,
                                log_prob_func=experiment_hyperparams[distribution]["log_prob"])),
                            # ESS cannot certify a sampler; this compares the
                            # sampled distribution against the exact chain
                            ("distribution_error", lambda: normalised_energy_distance(
                                model_dict[model]["samples"], true_samples)),
                            ("ess", lambda: _compute_ess(model_dict[model]["samples"])),
                        ):
                            try:
                                val = fn()
                                if key == "reversibility_error":
                                    val, fwd, bwd = val
                                    model_dict[model]["forward"] = fwd[:5]
                                    model_dict[model]["backward"] = bwd[:5]
                                row[key] = (val.detach().cpu().numpy()
                                            if torch.is_tensor(val) else val)
                            except Exception as exc:
                                print(f"!! DIAGNOSTIC FAILED  {key}  model={model}  "
                                      f"distribution={distribution}  percent={percent}\n"
                                      f"   {type(exc).__name__}: {exc}", flush=True)
                                row[key] = float("nan")
                        error_list.append(row)

                    pd.DataFrame(error_list).to_csv(
                        _out("experiments/diagnostic_results_analytic.csv"), index=False)
                    _safe_plots(model_dict, initial_positions, distribution,
                                experiment_hyperparams[distribution]["params_init"])

    pd.DataFrame(error_list).to_csv(_out("experiments/diagnostic_results_analytic.csv"), index=False)


# ── Riemannian-manifold HMC experiment ──────────────────────────────────────

rmhmc_experiment_hyperparams = {
    "banana": {
        "step_size": .05, "L": 5, "burn": 300, "N": 600,
        "params_init": torch.Tensor([0., 100.]),
        "log_prob": banana_log_prob,
        "softabs_const": 1e1,
    },
    "normal_normal": {
        "step_size": .1, "L": 5, "burn": 300, "N": 600,
        "params_init": torch.ones(2),
        "log_prob": lambda omega: normal_normal_conjugate(omega),
        "softabs_const": 1e1,
    },
    # calibrated: at eps=.1 the chain diverges (v reaches 38 against a true
    # range of about +/-9) and acceptance falls to 0.28; eps=.05 with a harder
    # softabs regularization gives 0.85
    "funnel": {
        "step_size": .05, "L": 5, "burn": 300, "N": 600,
        "params_init": torch.Tensor([0., 1.]),
        "log_prob": funnel_log_prob,
        "softabs_const": 1e3,
    },
}


def _augmented_flow(model, dim):
    """Adapt an augmented-state (q, p, q_cop, p_cop) neural ODE to the
    (q, p) interface the diagnostics expect."""
    def flow(x, t):
        t_eval, traj = model(torch.cat([x, x], -1), t)
        return t_eval, traj[..., :2 * dim]
    return flow


def run_rmhmc_experiment(model_type, distribution, percent=1, device="cuda"):
    hamiltorch.set_random_seed(123)
    print(f"Running RMHMC experiment for: distribution: {distribution}, model: {model_type}")
    hp = rmhmc_experiment_hyperparams[distribution]
    log_prob = hp["log_prob"]
    params_init = hp["params_init"].to(device)
    dim = params_init.shape[0]
    step_size, L = hp["step_size"], hp["L"]
    burn, N = (12, 24) if SMOKE else (hp["burn"], hp["N"])
    base_sampler = RMHMC(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim,
                          softabs_const=hp["softabs_const"])

    if model_type == "RMHMC":
        params_traj, _, _, _ = base_sampler.sample(q_init=params_init, num_samples=int(burn * percent))
        params_traj, _, _, _ = base_sampler.sample(q_init=params_traj[-1, -1, :], num_samples=N - int(burn * percent))

        def model_func(x, t):
            # Tao integration of the exact Hamiltonian is not batched: loop rows
            rows = []
            for row in x:
                try:
                    qs, ps, _, _, _ = base_sampler.step(row[..., :dim], row[..., dim:])
                    rows.append(torch.cat([qs, ps], -1))
                except hamiltorch.util.LogProbError:
                    # diverged trajectory: freeze the row at its initial state
                    rows.append(row[None, :].expand(base_sampler.L + 1, -1).clone())
            return None, torch.stack(rows, axis=1)  # (T, B, 2D), time-first

        return params_traj, model_func

    elif model_type in ("NNODEgRMHMC", "Explicit NNODEgRMHMC"):
        sampler = SurrogateNeuralODERMHMC(
            step_size=step_size, L=L, log_prob_func=log_prob, dim=dim,
            base_sampler=base_sampler,
            model_type="explicit_hamiltonian" if model_type.startswith("Explicit") else "",
        )
        sampler.create_surrogate(q_init=params_init, burn=int(burn * percent), epochs=ODE_EPOCHS)
        params_out, _, _, _ = sampler.sample(q_init=None, num_samples=N - int(burn * percent))
        return params_out, _augmented_flow(sampler.model, dim)

    elif model_type in ("GSymplecticNNgRMHMC", "GradGSymplecticNNgRMHMC"):
        sampler = SymplecticRMHMC(step_size=step_size, L=L, log_prob_func=log_prob,
                                   dim=dim, base_sampler=base_sampler, model_type="GSymp")
        # Grad variant consumes the full (dq/dt, dp/dt) field stored by RMHMC.step
        sampler.create_surrogate(q_init=params_init, burn=int(burn * percent), epochs=SNN_EPOCHS,
                                 use_gradient=model_type.startswith("Grad"))
        params_out, _, _, _ = sampler.sample(num_samples=N - int(burn * percent), q_init=None)
        return params_out, sampler.model


def rmhmc_experiment(device: str = "cuda"):
    """Compare exact RMHMC with surrogate approximations trained on its burn-in."""
    distributions = list(rmhmc_experiment_hyperparams.keys())
    models = ["RMHMC", "NNODEgRMHMC", "Explicit NNODEgRMHMC",
              "GSymplecticNNgRMHMC", "GradGSymplecticNNgRMHMC"]
    percent_of_warmup = [1.0] if SMOKE else np.linspace(0.25, 1, 4)
    error_list = []

    for percent in percent_of_warmup:
        for distribution in distributions:
            hp = rmhmc_experiment_hyperparams[distribution]
            model_dict = {}
            for model_type in models:
                start = time.time()
                experiment_samples, experiment_model = run_rmhmc_experiment(
                    model_type, distribution, percent, device=device
                )
                model_dict[model_type] = {
                    "samples": experiment_samples[:, -1, :].detach(),
                    "model": experiment_model,
                    "time": time.time() - start,
                }

            true_samples = model_dict["RMHMC"]["samples"]
            hamiltorch.set_random_seed(1)
            # exact-RMHMC reversibility costs a Hessian per integrator stage per
            # row: keep the evaluation batch small
            num_eval = min(20, true_samples.shape[0])
            initial_positions = true_samples[
                torch.multinomial(torch.ones(true_samples.shape[0]),
                                  num_samples=num_eval, replacement=False), :
            ]
            # momenta must come from the Riemannian kinetic distribution:
            # N(0, I) momenta are out-of-distribution under the metric and can
            # blow up the exact dynamics
            diag_sampler = RMHMC(step_size=hp["step_size"], L=hp["L"],
                                 log_prob_func=hp["log_prob"],
                                 dim=true_samples.shape[-1],
                                 softabs_const=hp["softabs_const"])
            # the diagnostics evaluate maps on (q, p), so momenta come from the
            # original conditional p ~ N(0, G(q)), not the extended-state draw
            initial_momentum = torch.stack(
                [diag_sampler.gibbs_marginal(q) for q in initial_positions]
            )
            initial_conditions = torch.cat([initial_positions, initial_momentum], -1)

            for model_type in model_dict:
                t_span = torch.linspace(0, hp["L"] * hp["step_size"], hp["L"] + 1)
                error, forward_traj, backward_traj = compute_reversibility_error(
                    model_dict[model_type]["model"], initial_conditions, t=t_span
                )
                rm_h_error = compute_rm_hamiltonian_error(
                    model_dict[model_type]["model"], initial_conditions, t=t_span,
                    rm_hamiltonian_func=diag_sampler.hamiltonian,
                )
                model_dict[model_type]["forward"] = forward_traj[:5]
                model_dict[model_type]["backward"] = backward_traj[:5]
                error_list.append({
                    "model": model_type,
                    "training_size": percent,
                    "distribution": distribution,
                    "step_size": hp["step_size"],
                    "reversibility_error": error.detach().cpu().numpy(),
                    "rm_hamiltonian_error": float(rm_h_error.cpu()),
                    "distribution_error": normalised_energy_distance(
                        model_dict[model_type]["samples"], true_samples),
                    "time": model_dict[model_type]["time"],
                    "ess": _compute_ess(model_dict[model_type]["samples"]),
                })

            plot_samples(
                model_dict,
                mean=hp["params_init"],
                distribution_name=f"rmhmc_{distribution}",
            )
            plot_reversibility(model_dict, initial_positions, distribution=f"rmhmc_{distribution}")

    pd.DataFrame(error_list).to_csv(_out("experiments/rmhmc_results.csv"), index=False)
    print("RMHMC experiment results saved to experiments/rmhmc_results.csv")


# ── Symmetrization control: is the gain from the symmetry or just depth? ─────

def run_symmetry_arm(arm, distribution, percent, device="cuda"):
    """arm = (label, model_type, n_blocks)."""
    label, model_type, n_blocks = arm
    hamiltorch.set_random_seed(123)
    hp = experiment_hyperparams[distribution]
    log_prob = hp["log_prob"]
    params_init = hp["params_init"].to(device)
    dim = params_init.shape[0]
    step_size, L = hp["step_size"], hp["L"]
    burn, N = _chain_lengths(hp)
    base = HMC(step_size=step_size, L=L, log_prob_func=log_prob, dim=dim)
    if model_type == "HMC":
        traj, _, _, _ = base.sample(q_init=params_init, num_samples=int(burn * percent))
        traj, _, _, _ = base.sample(q_init=traj[-1, -1, :], num_samples=N - int(burn * percent))
        def model_func(x, t):
            r = base.step(x[..., :dim], x[..., dim:])
            return (None, torch.cat([r[0], r[1]], -1))
        return traj, model_func, 0
    sampler = SymplecticHMC(step_size=step_size, L=L, log_prob_func=log_prob,
                            dim=dim, base_sampler=base, model_type=model_type)
    sampler.create_surrogate(q_init=params_init, burn=int(burn * percent),
                             epochs=SNN_EPOCHS, n_blocks=n_blocks)
    out, _, _, _ = sampler.sample(num_samples=N - int(burn * percent), q_init=None)
    n_par = sum(p.numel() for p in sampler.model.parameters())
    return out, sampler.model, n_par


def symmetrization_control_experiment(device: str = "cuda"):
    """Disentangle the time-symmetric wrapper's symmetry from its extra depth.

    Psi = R . Phi^{-1} . R . Phi applies Phi twice, so Rev(n) has the same
    effective depth as plain(2n) with half the parameters. Comparing
    Rev(n) against plain(2n) isolates the symmetry; comparing Rev(n) against
    plain(n) confounds it with depth.
    """
    arms = [
        ("HMC",        "HMC",       0),
        ("plain-4",    "GSymp",     4),
        ("plain-8",    "GSymp",     8),
        ("plain-16",   "GSymp",    16),
        ("Rev-4",      "RevGSymp",  4),   # effective depth 8  -> compare vs plain-8
        ("Rev-8",      "RevGSymp",  8),   # effective depth 16 -> compare vs plain-16
    ]
    distributions = ["gaussian", "normal_normal", "high_dimensional_gaussian"]
    percents = [1.0] if SMOKE else [0.325, 0.55, 1.0]
    rows = []
    for percent in percents:
        for distribution in distributions:
            hp = experiment_hyperparams[distribution]
            results = {}
            for arm in arms:
                start = time.time()
                samples, model, n_par = run_symmetry_arm(arm, distribution, percent, device=device)
                results[arm[0]] = {"samples": samples[:, -1, :].detach(), "model": model,
                                   "time": time.time() - start, "params": n_par}
            true_samples = results["HMC"]["samples"]
            hamiltorch.set_random_seed(1)
            n_eval = min(100, true_samples.shape[0])
            mom = torch.distributions.Normal(0, 1).sample(sample_shape=(n_eval, true_samples.shape[-1]))
            pos = true_samples[torch.multinomial(torch.ones(true_samples.shape[0]),
                                                 num_samples=n_eval, replacement=False), :]
            init = torch.cat([pos, mom], -1)
            t_span = torch.linspace(0, hp["L"] * hp["step_size"], hp["L"] + 1)
            for label in results:
                err, _, _ = compute_reversibility_error(results[label]["model"], init, t=t_span)
                rows.append({
                    "arm": label,
                    "distribution_error": normalised_energy_distance(
                        results[label]["samples"], true_samples),
                    "effective_depth": {"HMC": 0, "plain-4": 4, "plain-8": 8, "plain-16": 16,
                                        "Rev-4": 8, "Rev-8": 16}[label],
                    "params": results[label]["params"],
                    "distribution": distribution,
                    "training_size": percent,
                    "reversibility_error": err.detach().cpu().numpy(),
                    "ess": _compute_ess(results[label]["samples"]),
                    "time": results[label]["time"],
                })
    pd.DataFrame(rows).to_csv(_out("experiments/symmetrization_control.csv"), index=False)
    print("Symmetrization control saved to experiments/symmetrization_control.csv")


# ── Cost scaling: pair construction and trajectory length ───────────────────

def pair_mode_experiment(device: str = "cuda"):
    """Is the O(L^2) pair construction necessary?

    The sampler queries the flow map only at tau = L*eps, so in principle
    training needs only the (x_0, x_L) pair per trajectory. This measures what
    the extra offsets buy, across trajectory lengths where the distinction
    matters for cost.
    """
    lengths = [5] if SMOKE else [5, 25, 50]
    modes = ["all", "from_start", "endpoint"]
    distribution = "gaussian"
    hp = dict(experiment_hyperparams[distribution])
    rows = []
    for L in lengths:
        for mode in modes:
            hamiltorch.set_random_seed(123)
            log_prob = hp["log_prob"]
            params_init = hp["params_init"].to(device)
            dim = params_init.shape[0]
            eps = hp["step_size"]
            burn, N = (20, 40) if SMOKE else (600, 1200)
            base = HMC(step_size=eps, L=L, log_prob_func=log_prob, dim=dim)
            sampler = SymplecticHMC(step_size=eps, L=L, log_prob_func=log_prob,
                                    dim=dim, base_sampler=base, model_type="RevGSymp")
            t0 = time.time()
            sampler.create_surrogate(q_init=params_init, burn=burn, epochs=SNN_EPOCHS,
                                     pair_mode=mode)
            train_time = time.time() - t0
            t1 = time.time()
            out, _, _, _ = sampler.sample(num_samples=N - burn, q_init=None)
            sample_time = time.time() - t1
            samples = out[:, -1, :].detach()
            hamiltorch.set_random_seed(1)
            n_eval = min(50, samples.shape[0])
            init = torch.cat([
                samples[torch.multinomial(torch.ones(samples.shape[0]), n_eval, replacement=False)],
                torch.distributions.Normal(0, 1).sample(sample_shape=(n_eval, dim))], -1)
            err, _, _ = compute_reversibility_error(
                sampler.model, init, t=torch.linspace(0, L * eps, L + 1))
            pairs_per_traj = {"all": L * (L + 1) // 2, "from_start": L, "endpoint": 1}[mode]
            rows.append({
                "L": L, "pair_mode": mode,
                "pairs_per_traj": pairs_per_traj,
                "total_pairs": pairs_per_traj * burn,
                "train_time": train_time, "sample_time": sample_time,
                "reversibility_error": err.detach().cpu().numpy(),
                "ess": _compute_ess(samples),
            })
            print(f"  L={L} mode={mode}: {pairs_per_traj} pairs/traj, "
                  f"train {train_time:.1f}s, ESS {rows[-1]['ess']:.1f}")
    pd.DataFrame(rows).to_csv(_out("experiments/pair_mode.csv"), index=False)
    print("Pair-mode results saved to experiments/pair_mode.csv")


def trajectory_length_experiment(device: str = "cuda"):
    """Per-proposal cost of a flow-map surrogate is O(1) in L; integrator-based
    surrogates are O(L). This measures ESS/s for both as L grows."""
    lengths = [5] if SMOKE else [5, 25, 50, 100]
    # (label, model, pair_mode) — the flow map is run under both pair
    # constructions, since the O(L^2) one both costs more and fits worse
    arms = [("HMC", "HMC", "all"),
            ("NNODEgHMC", "NNODEgHMC", "all"),
            ("RevSympNet[all]", "RevGSymplecticNNgHMC", "all"),
            ("RevSympNet[endpoint]", "RevGSymplecticNNgHMC", "endpoint")]
    distribution = "gaussian"
    rows = []
    for L in lengths:
        # temporarily override L for this target
        saved = experiment_hyperparams[distribution]["L"]
        experiment_hyperparams[distribution]["L"] = L
        try:
            for label, model, pmode in arms:
                start = time.time()
                samples, _, _ = run_experiment(model, "autograd", distribution,
                                               SynchronousLeapfrog(), 1.0, device=device,
                                               pair_mode=pmode)
                elapsed = time.time() - start
                e = _compute_ess(samples[:, -1, :].detach())
                rows.append({"L": L, "model": label, "pair_mode": pmode, "ess": e,
                             "time": elapsed, "ess_per_sec": e / max(elapsed, 1e-9)})
                print(f"  L={L} {label}: ESS {e:.1f}, {elapsed:.1f}s, ESS/s {e/max(elapsed,1e-9):.2f}")
        finally:
            experiment_hyperparams[distribution]["L"] = saved
    pd.DataFrame(rows).to_csv(_out("experiments/trajectory_length.csv"), index=False)
    print("Trajectory-length results saved to experiments/trajectory_length.csv")
