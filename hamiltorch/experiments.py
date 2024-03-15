import torch
import numpy as np
import hamiltorch
import arviz as az
from hamiltorch.samplers import leapfrog
from hamiltorch.ode import SynchronousLeapfrog
from hamiltorch.plot_utils import plot_results, plot_reversibility, plot_samples
from hamiltorch.experiment_utils import banana_log_prob, gaussian_log_prob, high_dimensional_gaussian_log_prob, compute_reversibility_error, params_grad, normal_normal_conjugate
from arviz import ess, autocorr
import pandas as pd
import time



hamiltorch.set_random_seed(13)

experiment_hyperparams = {
    "banana": {"step_size": .1, "L":5 , "burn": 3000, "N": 6000 , "params_init": torch.Tensor([0.,100.
                                                                                                ]), "log_prob": banana_log_prob,
                                                                                                "grad_func": lambda p: params_grad(p, banana_log_prob)},
        "gaussian": {"step_size":.3, "L":5, "burn": 1000, "N": 2000, "params_init": torch.zeros(3), "log_prob": gaussian_log_prob,
                     "grad_func": lambda p: params_grad(p, gaussian_log_prob)
                      },
        "high_dimensional_gaussian": {"step_size": .1, "L":5 , "burn": 3000 , "N": 6000 , "D": 30, "params_init": torch.randn(30), 
                                     "log_prob": lambda omega: high_dimensional_gaussian_log_prob(omega, D = 30),
                                     "grad_func": lambda p: params_grad(p, high_dimensional_gaussian_log_prob)},
        "normal_normal": {"step_size": .1, "L":5 , "burn": 3000 , "N": 6000 , "params_init": torch.ones(2), 
                                     "log_prob": lambda omega: normal_normal_conjugate(omega),
                                     "grad_func": lambda p: params_grad(p, normal_normal_conjugate)}
}


def run_experiment(model_type, sensitivity, distribution, solver, percent = 1):
    hamiltorch.set_random_seed(123)
    print(f"Running experiment for: solver: {solver}, sensitivity: {sensitivity}, distribution: {distribution}, model: {model_type}")
    experiment_params = experiment_hyperparams[distribution]
    log_prob = experiment_params["log_prob"]
    params_init = experiment_params["params_init"]
    dim = params_init.shape[0]
    step_size = experiment_params["step_size"]
    L = experiment_params["L"]
    N = experiment_params["N"]
    burn = experiment_params["burn"]
    if solver == "SynchronousLeapfrog":
        solver = SynchronousLeapfrog()
    if model_type == "HMC":
        params_hmc = hamiltorch.sample(log_prob_func=log_prob, params_init=params_init, num_samples=N,
                               step_size=step_size, num_steps_per_sample=L, burn = burn)

        model = lambda x, t: (None, torch.cat([torch.stack(item,0) for item in 
                                               leapfrog(x[..., :dim], x[..., dim:], log_prob_func=log_prob,steps = t.shape[0], 
                                                        step_size=step_size)], -1))
        gradient_func = experiment_params["grad_func"]
        return params_hmc, model, gradient_func
    elif model_type == "NNgHMC":
        params_hmc_surrogate, surrogate_model = hamiltorch.sample_surrogate_hmc(log_prob_func=log_prob, params_init=params_init,
                                                  num_samples=N,step_size=step_size,num_steps_per_sample=L,burn=int(burn * percent),
                                                  desired_accept_rate=0.8)
        model = lambda x, t: (None, torch.cat([torch.stack(item,0) for item in 
                                               leapfrog(x[..., :dim], x[..., dim:], log_prob_func=log_prob,steps = t.shape[0], 
                                                        step_size=step_size, pass_grad=surrogate_model)], -1))
        return params_hmc_surrogate, model, surrogate_model
    elif model_type == "NNODEgHMC":
        params_hmc_surrogate_ode_nnghmc, surrogate_model_ode_nnghmc = hamiltorch.sample_neural_ode_surrogate_hmc(log_prob_func=log_prob, params_init=params_init,
                                                  num_samples=N,step_size=step_size,num_steps_per_sample=L,burn=int(burn * percent), model_type = "", sensitivity=sensitivity, solver = solver
                                                  )
        gradient_func = surrogate_model_ode_nnghmc.odefunc
        return params_hmc_surrogate_ode_nnghmc, surrogate_model_ode_nnghmc, gradient_func
    elif model_type == "Explicit NNODEgHMC":
        params_hmc_surrogate_ode_explicit, surrogate_model_ode_explicit = hamiltorch.sample_neural_ode_surrogate_hmc(log_prob_func=log_prob, params_init=params_init,
                                                  num_samples=N,step_size=step_size,num_steps_per_sample=L,burn=int(burn*percent), model_type = "explicit_hamiltonian", 
                                                  sensitivity=sensitivity, solver = solver
                                                  )
        gradient_func = surrogate_model_ode_explicit.odefunc

        return params_hmc_surrogate_ode_explicit, surrogate_model_ode_explicit, gradient_func




def surrogate_neural_ode_hmc_experiment():
    distributions = ["banana", "gaussian", "high_dimensional_gaussian", "normal_normal"]
    sensitivities = ["autograd"]
    solvers = ["SynchronousLeapfrog"]
    models = ["HMC", "NNgHMC", "Explicit NNODEgHMC", "NNODEgHMC"]
    error_list = []
    for distribution in distributions:
        for sensitivity in sensitivities:
            for solver in solvers:
                model_dict = {}
                for model in models:
                    
                    start = time.time()
                    
                    experiment_samples, experiment_model, experiment_grad_func = run_experiment(model, sensitivity, distribution, solver)

                    end = time.time()
                    model_dict[model] = {"samples":experiment_samples, "model": experiment_model, "time": end - start}
                    
                true_samples = torch.stack(model_dict["HMC"]["samples"], 0)
                
                hamiltorch.set_random_seed(1)
                num_samples = 100
                initial_momentum = torch.distributions.Normal(0,1).sample(sample_shape = (num_samples, true_samples.shape[-1]))
                initial_positions = true_samples[torch.multinomial(torch.ones(true_samples.shape[0]), num_samples = 100, replacement=False), :]
                initial_conditions = torch.cat([initial_positions, initial_momentum], -1)
                
                for model in model_dict:
                    error_dict = {}
                    step_size = experiment_hyperparams[distribution]["step_size"] 
                    L = experiment_hyperparams[distribution]["L"] 
                    error, forward_traj, backward_traj = compute_reversibility_error(model_dict[model]["model"], initial_conditions,
                                                        t = torch.linspace(0, L * step_size, L ))
                    model_dict[model]["forward"] = forward_traj[0:5, :]
                    model_dict[model]["backward"] = backward_traj[0:5, :]

                    error_dict["model"] = model
                    error_dict["sensitivity"] = sensitivity
                    error_dict["distribution"] = distribution
                    error_dict["solver"] = solver
                    error_dict["reversibility_error"] = error
                    error_dict["time"] = model_dict[model]["time"]
                    # error_dict["acf"] = autocorr(torch.stack(model_dict[model]["samples"],0).numpy()[None, :, :])
                    error_dict["ess"] = ess(az.convert_to_inference_data(torch.stack(model_dict[model]["samples"],0).numpy()[None, : ,: ])).x.mean().values
                    error_list.append(error_dict)

                plot_samples(model_dict, mean = experiment_hyperparams[distribution]["params_init"], distribution_name=distribution)
                plot_reversibility(model_dict, initial_positions,
                                        distribution=distribution)
    pd.DataFrame(error_list).to_csv("../experiments/diagnostic_results.csv", index = False)
    

def surrogate_neural_ode_hmc_sample_size_experiment():
    distributions = ["banana", "gaussian", "high_dimensional_gaussian", "normal_normal"]
    sensitivities = ["autograd"]
    solvers = ["SynchronousLeapfrog"]
    models = ["HMC", "NNgHMC", "Explicit NNODEgHMC", "NNODEgHMC"]
    percent_of_warmup = np.linspace(0.1, 1, 10)
    error_list = []
    for percent in percent_of_warmup:
        for distribution in distributions:
            for sensitivity in sensitivities:
                for solver in solvers:
                    model_dict = {}
                    for model in models:
                        
                        start = time.time()
                        
                        experiment_samples, experiment_model, experiment_grad_func = run_experiment(model, sensitivity, distribution, solver, percent)

                        end = time.time()
                        model_dict[model] = {"samples":experiment_samples, "model": experiment_model, "time": end - start}
                        
                    true_samples = torch.stack(model_dict["HMC"]["samples"], 0)
                    
                    hamiltorch.set_random_seed(1)
                    num_samples = 100
                    initial_momentum = torch.distributions.Normal(0,1).sample(sample_shape = (num_samples, true_samples.shape[-1]))
                    initial_positions = true_samples[torch.multinomial(torch.ones(true_samples.shape[0]), num_samples = 100, replacement=False), :]
                    initial_conditions = torch.cat([initial_positions, initial_momentum], -1)
                    
                    for model in model_dict:
                        error_dict = {}
                        step_size = experiment_hyperparams[distribution]["step_size"] 
                        L = experiment_hyperparams[distribution]["L"] 
                        error, forward_traj, backward_traj = compute_reversibility_error(model_dict[model]["model"], initial_conditions,
                                                            t = torch.linspace(0, L * step_size, L ))
                        model_dict[model]["forward"] = forward_traj[0:5, :]
                        model_dict[model]["backward"] = backward_traj[0:5, :]

                        error_dict["model"] = model
                        error_dict["training_size"] = percent
                        error_dict["sensitivity"] = sensitivity
                        error_dict["distribution"] = distribution
                        error_dict["solver"] = solver
                        error_dict["reversibility_error"] = error
                        error_dict["time"] = model_dict[model]["time"]
                        # error_dict["acf"] = autocorr(torch.stack(model_dict[model]["samples"],0).numpy()[None, :, :])
                        error_dict["ess"] = ess(az.convert_to_inference_data(torch.stack(model_dict[model]["samples"],0).numpy()[None, : ,: ])).x.mean().values
                        error_list.append(error_dict)

                    plot_samples(model_dict, mean = experiment_hyperparams[distribution]["params_init"], distribution_name=distribution)
                    plot_reversibility(model_dict, initial_positions,
                                        distribution=distribution)
    pd.DataFrame(error_list).to_csv("../experiments/diagnostic_results.csv", index = False)

