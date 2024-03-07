import torch
import hamiltorch
from hamiltorch.samplers import leapfrog_hmc
from hamiltorch.ode import SynchronousLeapfrog
from torch.autograd import grad
from hamiltorch.plot_utils import plot_results, plot_reversibility, plot_samples
from hamiltorch.experiment_utils import banana_log_prob, gaussian_log_prob, ill_conidtioned_gaussian_log_prob, compute_reversibility_error, params_grad

experiment_hyperparams = {
    "banana": {"step_size": .1, "L":5 , "burn": 3000, "N": 6000 , "params_init": torch.Tensor([0.,100.
                                                                                                ]), "log_prob": banana_log_prob,
                                                                                                "grad_func": lambda p: params_grad(p, banana_log_prob)},
        "gaussian": {"step_size":.3, "L":5, "burn": 1000, "N": 2000, "params_init": torch.zeros(3), "log_prob": gaussian_log_prob,
                     "grad_func": lambda p: params_grad(p, gaussian_log_prob)
                      },
        "ill_conditioned_gaussian": {"step_size": .5, "L":100 , "burn": 3000 , "N": 6000 , "D": 30, "params_init": torch.zeros(30), 
                                     "log_prob": ill_conidtioned_gaussian_log_prob,
                                     "grad_func": lambda p: params_grad(p, ill_conidtioned_gaussian_log_prob)}
}


def run_experiment(model_type, sensitivity, distribution, solver):
    hamiltorch.set_random_seed(123)
    experiment_params = experiment_hyperparams[distribution]
    log_prob = experiment_params["log_prob"]
    params_init = experiment_params["params_init"]
    dim = params_init.shape[0]
    step_size = experiment_params["step_size"]
    L = experiment_params["L"]
    N = experiment_params["N"]
    burn = experiment_params["burn"]
    if model_type == "HMC":
        params_hmc = hamiltorch.sample(log_prob_func=log_prob, params_init=params_init, num_samples=N,
                               step_size=step_size, num_steps_per_sample=L, burn = burn)

        model = lambda x, t: leapfrog_hmc(x[..., :dim], x[..., dim:], log_prob_func=log_prob,steps = t.shape[0], step_size=step_size)
        gradient_func = experiment_params["grad_func"]
    elif model_type == "NNgHMC":
        params_hmc_surrogate, surrogate_model = hamiltorch.sample_surrogate_hmc(log_prob_func=log_prob, params_init=params_init,
                                                  num_samples=N,step_size=step_size,num_steps_per_sample=L,burn=burn,
                                                  desired_accept_rate=0.8)
        model = lambda x, t: leapfrog_hmc(x[..., :dim], x[..., dim:], log_prob_func=log_prob,steps = t.shape[0], step_size=step_size, pass_grad=surrogate_model)
        gradient_func = surrogate_model
    elif model_type == "NNODEgHMC":
        params_hmc_surrogate_ode_nnghmc, surrogate_model_ode_nnghmc = hamiltorch.sample_neural_ode_surrogate_hmc(log_prob_func=log_prob, params_init=params_init,
                                                  num_samples=N,step_size=step_size,num_steps_per_sample=L,burn=burn, model_type = ""
                                                  )
        gradient_func = surrogate_model_ode_nnghmc.odefunc
    elif model_type == "Explicit NNODEgHMC":
        params_hmc_surrogate_ode_explicit, surrogate_model_ode_explicit = hamiltorch.sample_neural_ode_surrogate_hmc(log_prob_func=log_prob, params_init=params_init,
                                                  num_samples=N,step_size=step_size,num_steps_per_sample=L,burn=burn, model_type = "explicit_hamiltonian"
                                                  )
        gradient_func = surrogate_model_ode_explicit.odefunc




def surrogate_neural_ode_hmc_experiment():
    distributions = ["banana", "gaussian", "ill_conditioned_gaussian"]
    sensitivities = ["adjoint", "autograd"]
    solvers = ["SynchronousLeapfrog", "dopri5"]
    models = ["HMC", "NNgHMC", "Explicit NNODEgHMC", "NNODEgHMC"]

    for model in models:
        for sensitivity in sensitivities:
            for distribution in distributions:
                for solver in solvers:
                    run_experiment(model, sensitivity, distribution, solver)





