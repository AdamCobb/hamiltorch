import torch
import torch.nn as nn
from enum import Enum

from numpy import pi
from . import util


class Sampler(Enum):
    HMC = 1
    RMHMC = 2
    HMC_NUTS = 3
    # IMPORTANCE = 3
    # MH = 4


class Integrator(Enum):
    EXPLICIT = 1
    IMPLICIT = 2
    S3       = 3


class Metric(Enum):
    HESSIAN = 1
    SOFTABS = 2
    JACOBIAN_DIAG = 3

def collect_gradients(log_prob, params):
    if isinstance(log_prob, tuple):
        log_prob[0].backward()
        params_list = list(log_prob[1])
        # params = util.flatten(params_list)
        params = torch.cat([p.flatten() for p in params_list])
        params.grad = torch.cat([p.grad.flatten() for p in params_list])
    else:
        params.grad = torch.autograd.grad(log_prob,params)[0]
        # log_prob.backward()
        # import pdb; pdb.set_trace()
    return params


def fisher(params, log_prob_func=None, jitter=None, normalizing_const=1., softabs_const=1e6, metric=Metric.HESSIAN):
    log_prob = log_prob_func(params)
    if util.has_nan_or_inf(log_prob):
        print('Invalid log_prob: {}, params: {}'.format(log_prob, params))
        raise util.LogProbError()
    if metric == Metric.JACOBIAN_DIAG:
        # raise NotImplementedError()
        # import pdb; pdb.set_trace()
        jac = util.jacobian(log_prob, params, create_graph=True, return_inputs=False)
        jac = torch.cat([j.flatten() for j in jac])
        # util.flatten(jac).view(1,-1)
        fish = torch.matmul(jac.view(-1,1),jac.view(1,-1)).diag().diag()/ normalizing_const #.diag().diag() / normalizing_const
    else:
        hess = util.hessian(log_prob.float(), params, create_graph=True, return_inputs=False)
        fish = - hess / normalizing_const
    if util.has_nan_or_inf(fish):
        print('Invalid hessian: {}, params: {}'.format(fish, params))
        raise util.LogProbError()
    if jitter is not None:
        params_n_elements = fish.shape[0]
        fish += (torch.eye(params_n_elements) * torch.rand(params_n_elements) * jitter).to(fish.device)
    if (metric is Metric.HESSIAN) or (metric is Metric.JACOBIAN_DIAG):
        return fish, None
    elif metric == Metric.SOFTABS:
        eigenvalues, eigenvectors = torch.symeig(fish, eigenvectors=True)
        abs_eigenvalues = (1./torch.tanh(softabs_const * eigenvalues)) * eigenvalues
        fish = torch.matmul(eigenvectors, torch.matmul(abs_eigenvalues.diag(), eigenvectors.t()))
        return fish, abs_eigenvalues
    else:
            # if metric == Metric.JACOBIAN:
            #     jac = jacobian(log_prob, params, create_graph=True)
            #     fish = torch.matmul(jac.t(),jac) / normalizing_const
        raise ValueError('Unknown metric: {}'.format(metric))


def cholesky_inverse(fish, momentum):
    lower = torch.cholesky(fish)
    y = torch.triangular_solve(momentum.view(-1, 1), lower, upper=False, transpose=False, unitriangular=False)[0]
    fish_inv_p = torch.triangular_solve(y, lower.t(), upper=True, transpose=False, unitriangular=False)[0]
    return fish_inv_p


def gibbs(params, sampler=Sampler.HMC, log_prob_func=None, jitter=None, normalizing_const=1., softabs_const=None, mass=None, metric=Metric.HESSIAN):
    if sampler == Sampler.RMHMC:
        dist = torch.distributions.MultivariateNormal(torch.zeros_like(params), fisher(params, log_prob_func, jitter, normalizing_const, softabs_const, metric)[0])
    elif mass is None:
        dist = torch.distributions.Normal(torch.zeros_like(params), torch.ones_like(params))
    else:
        if len(mass.shape) == 2:
            dist = torch.distributions.MultivariateNormal(torch.zeros_like(params), mass)
        elif len(mass.shape) == 1:
            dist = torch.distributions.Normal(torch.zeros_like(params), mass)
    return dist.sample()


def leapfrog(params, momentum, log_prob_func, steps=10, step_size=0.1, jitter=0.01, normalizing_const=1., softabs_const=1e6, explicit_binding_const=100, fixed_point_threshold=1e-20, fixed_point_max_iterations=6, jitter_max_tries=10, inv_mass=None, ham_func=None, sampler=Sampler.HMC, integrator=Integrator.IMPLICIT, metric=Metric.HESSIAN, debug=False):
    if sampler == Sampler.HMC:
        def params_grad(p):
            p = p.detach().requires_grad_()
            log_prob = log_prob_func(p)
            # log_prob.backward()
            p = collect_gradients(log_prob, p)
            return p.grad
        ret_params = []
        ret_momenta = []
        momentum += 0.5 * step_size * params_grad(params)
        for n in range(steps):
            if inv_mass is None:
                params = params + step_size * momentum
            else:
                #Assum G is diag here so 1/Mass = G inverse
                if len(inv_mass.shape) == 2:
                    params = params + step_size * torch.matmul(inv_mass,momentum.view(-1,1)).view(-1)
                else:
                    params = params + step_size * inv_mass * momentum
            p_grad = params_grad(params)
            momentum += step_size * p_grad
            ret_params.append(params.clone())
            ret_momenta.append(momentum.clone())
        # only need last for Hamiltoninian check (see p.14) https://arxiv.org/pdf/1206.1901.pdf
        ret_momenta[-1] = ret_momenta[-1] - 0.5 * step_size * p_grad.clone()
            # import pdb; pdb.set_trace()
        return ret_params, ret_momenta
    elif sampler == Sampler.RMHMC and (integrator == Integrator.IMPLICIT or integrator == Integrator.S3):
        if integrator is not Integrator.S3:
            ham_func = None
            # Else we are doing semi sep and need auxiliary for Riemann version.

        def fixed_point_momentum(params, momentum):
            momentum_old = momentum.clone()
            # print('s')
            for i in range(fixed_point_max_iterations):
                momentum_prev = momentum.clone()
                params = params.detach().requires_grad_()
                ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, normalizing_const=normalizing_const, ham_func=ham_func, sampler=sampler, integrator=integrator, metric=metric)
                params = collect_gradients(ham, params)

                # draw the jitter on the diagonal of Fisher again (probably a better place to do this)
                tries = 0
                while util.has_nan_or_inf(params.grad):
                    params = params.detach().requires_grad_()
                    ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, normalizing_const=normalizing_const, ham_func=ham_func, sampler=sampler, integrator=integrator, metric=metric)
                    params = collect_gradients(ham, params)
                    tries += 1
                    if tries > jitter_max_tries:
                        print('Warning: reached jitter_max_tries {}'.format(jitter_max_tries))
                        # import pdb; pdb.set_trace()
                        raise util.LogProbError()
                        # import pdb; pdb.set_trace()
                        # break

                momentum = momentum_old - 0.5 * step_size * params.grad
                momenta_diff = torch.max((momentum_prev-momentum)**2)
                if momenta_diff < fixed_point_threshold:
                    break
            if debug:
                print('Converged (momentum), iterations: {}, momenta_diff: {}'.format(i, momenta_diff))
            return momentum

        def fixed_point_params(params, momentum):
            params_old = params.clone()
            momentum = momentum.detach().requires_grad_()
            ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, normalizing_const=normalizing_const, ham_func=ham_func, sampler=sampler, integrator=integrator, metric=metric)
            momentum = collect_gradients(ham,momentum)
            momentum_grad_old = momentum.grad.clone()
            for i in range(fixed_point_max_iterations):
                params_prev = params.clone()
                momentum = momentum.detach().requires_grad_()
                ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, normalizing_const=normalizing_const, ham_func=ham_func, sampler=sampler, integrator=integrator, metric=metric)
                momentum = collect_gradients(ham,momentum)#collect_gradients(ham, params)
                params = params_old + 0.5 * step_size * momentum.grad + 0.5 * step_size * momentum_grad_old
                params_diff = torch.max((params_prev-params)**2)
                if params_diff < fixed_point_threshold:
                    break
            if debug:
                print('Converged (params), iterations: {}, params_diff: {}'.format(i, params_diff))
            return params
        ret_params = []
        ret_momenta = []
        for n in range(steps):
            # import pdb; pdb.set_trace()
            momentum = fixed_point_momentum(params, momentum)
            params = fixed_point_params(params, momentum)

            params = params.detach().requires_grad_()
            ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, normalizing_const=normalizing_const, ham_func=ham_func, sampler=sampler, integrator=integrator, metric=metric)
            params = collect_gradients(ham, params)

            # draw the jitter on the diagonal of Fisher again (probably a better place to do this)
            tries = 0
            while util.has_nan_or_inf(params.grad):
                params = params.detach().requires_grad_()
                ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, normalizing_const=normalizing_const, ham_func=ham_func, sampler=sampler, integrator=integrator, metric=metric)
                params = collect_gradients(ham, params)
                tries += 1
                if tries > jitter_max_tries:
                    print('Warning: reached jitter_max_tries {}'.format(jitter_max_tries))
                    raise util.LogProbError()
                    # break
            momentum -= 0.5 * step_size * params.grad

            ret_params.append(params)
            ret_momenta.append(momentum)
        return ret_params, ret_momenta

    elif sampler == Sampler.RMHMC and integrator == Integrator.EXPLICIT:
        #During leapfrog define integrator as implict when passing into riemannian_hamiltonian
        leapfrog_hamiltonian_flag = Integrator.IMPLICIT
        def hamAB_grad_params(params,momentum):
            params = params.detach().requires_grad_()
            ham = hamiltonian(params, momentum.detach(), log_prob_func, jitter=jitter, normalizing_const=normalizing_const, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, sampler=sampler, integrator=leapfrog_hamiltonian_flag, metric=metric)
            params = collect_gradients(ham, params)

            # draw the jitter on the diagonal of Fisher again (probably a better place to do this)
            tries = 0
            while util.has_nan_or_inf(params.grad):
                # import pdb; pdb.set_trace()
                params = params.detach().requires_grad_()
                ham = hamiltonian(params, momentum.detach(), log_prob_func, jitter=jitter, normalizing_const=normalizing_const, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, sampler=sampler, integrator=leapfrog_hamiltonian_flag, metric=metric)
                params = collect_gradients(ham, params)
                tries += 1
                if tries > jitter_max_tries:
                    print('Warning: reached jitter_max_tries {}'.format(jitter_max_tries))
                    raise util.LogProbError()
                    # import pdb; pdb.set_trace()
                    # break

            return params.grad
        def hamAB_grad_momentum(params,momentum):
            momentum = momentum.detach().requires_grad_()
            params = params.detach().requires_grad_()
            # Can't detach p as we still need grad to do derivatives
            ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, normalizing_const=normalizing_const, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, sampler=sampler, integrator=leapfrog_hamiltonian_flag, metric=metric)
            # import pdb; pdb.set_trace()
            momentum = collect_gradients(ham,momentum)
            return momentum.grad
        ret_params = []
        ret_momenta = []
        params_copy = params.clone()
        momentum_copy = momentum.clone()
        for n in range(steps):
            # \phi_{H_A}
            momentum = momentum - 0.5 * step_size * hamAB_grad_params(params,momentum_copy)
            params_copy = params_copy + 0.5 * step_size * hamAB_grad_momentum(params,momentum_copy)
            # \phi_{H_B}
            params = params + 0.5 * step_size * hamAB_grad_momentum(params_copy,momentum)
            momentum_copy = momentum_copy - 0.5 * step_size * hamAB_grad_params(params_copy,momentum)
            # \phi_{H_C}
            c = torch.cos(torch.FloatTensor([2* explicit_binding_const * step_size])).to(params.device)
            s = torch.sin(torch.FloatTensor([2* explicit_binding_const * step_size])).to(params.device)
            # params_add = params + params_copy
            # params_sub = params - params_copy
            # momentum_add = momentum + momentum_copy
            # momentum_sub = momentum - momentum_copy
            # ### CHECK IF THE VALUES ON THE RIGHT NEED TO BE THE OLD OR UPDATED ones
            # ### INSTINCT IS THAT USING UPDATED ONES IS BETTER
            # params = 0.5 * ((params_add) + c*(params_sub) + s*(momentum_sub))
            # momentum = 0.5 * ((momentum_add) - s*(params_sub) + c*(momentum_sub))
            # params_copy = 0.5 * ((params_add) - c*(params_sub) - s*(momentum_sub))
            # momentum_copy = 0.5 * ((momentum_add) + s*(params_sub) - c*(momentum_sub))
            params = 0.5 * ((params+params_copy) + c*(params-params_copy) + s*(momentum-momentum_copy))
            momentum = 0.5 * ((momentum+momentum_copy) - s*(params-params_copy) + c*(momentum-momentum_copy))
            params_copy = 0.5 * ((params+params_copy) - c*(params-params_copy) - s*(momentum-momentum_copy))
            momentum_copy = 0.5 * ((momentum+momentum_copy) + s*(params-params_copy) - c*(momentum-momentum_copy))


            # \phi_{H_B}
            params = params + 0.5 * step_size * hamAB_grad_momentum(params_copy,momentum)
            momentum_copy = momentum_copy - 0.5 * step_size * hamAB_grad_params(params_copy,momentum)
            # \phi_{H_A}
            momentum = momentum - 0.5 * step_size * hamAB_grad_params(params,momentum_copy)
            params_copy = params_copy + 0.5 * step_size * hamAB_grad_momentum(params,momentum_copy)

            ret_params.append(params.clone())
            ret_momenta.append(momentum.clone())
        return [ret_params,params_copy], [ret_momenta, momentum_copy]


    else:
        raise NotImplementedError()


def acceptance(h_old, h_new):
    # if isinstance(h_old, tuple):
    #     return float(-torch.log(h_new[0]) + torch.log(h_old[0]))
    # else:
    # return float(-torch.log(h_new) + torch.log(h_old))
    return float(-h_new + h_old)

# Adaptation p.15 No-U-Turn samplers Algo 5
def adaptation(rho, t, step_size_init, H_t, eps_bar, desired_accept_rate=0.8):
    # rho is current acceptance ratio
    # t is current iteration
    t = t + 1
    if util.has_nan_or_inf(torch.tensor([rho])):
        alpha = 0 # Acceptance rate is zero if nan.
    else:
        alpha = min(1.,float(torch.exp(torch.FloatTensor([rho]))))
    mu = float(torch.log(10*torch.FloatTensor([step_size_init])))
    gamma = 0.05
    t0 = 10
    kappa = 0.75
    H_t = (1-(1/(t+t0)))*H_t + (1/(t+t0))*(desired_accept_rate - alpha)
    x_new = mu - (t**0.5)/gamma * H_t
    step_size = float(torch.exp(torch.FloatTensor([x_new])))
    x_new_bar = t**-kappa * x_new +  (1 - t**-kappa) * torch.log(torch.FloatTensor([eps_bar]))
    eps_bar = float(torch.exp(x_new_bar))
    # import pdb; pdb.set_trace()
    # print('rho: ',rho)
    # print('alpha: ',alpha)
    # print('step_size: ',step_size)
    # adapt_stepsize_list.append(torch.exp(x_new_bar))
    return step_size, eps_bar, H_t


def rm_hamiltonian(params, momentum, log_prob_func, jitter, normalizing_const, softabs_const=1e6, sampler=Sampler.HMC, integrator=Integrator.EXPLICIT, metric=Metric.HESSIAN):
    log_prob = log_prob_func(params)
    ndim = params.nelement()
    pi_term = ndim * torch.log(2.*torch.tensor(pi))

    fish, abs_eigenvalues = fisher(params, log_prob_func, jitter=jitter, normalizing_const=normalizing_const, softabs_const=softabs_const, metric=metric)

    if abs_eigenvalues is not None:
        if util.has_nan_or_inf(fish) or util.has_nan_or_inf(abs_eigenvalues):
            print('Invalid Fisher: {} , abs_eigenvalues: {}, params: {}'.format(fish, abs_eigenvalues, params))
            raise util.LogProbError()
    else:
        if util.has_nan_or_inf(fish):
            print('Invalid Fisher: {}, params: {}'.format(fish, params))
            raise util.LogProbError()

    if metric == Metric.SOFTABS:
        log_det_abs = abs_eigenvalues.log().sum()
    else:
        log_det_abs = torch.slogdet(fish)[1]
    fish_inverse_momentum = cholesky_inverse(fish, momentum)
    quadratic_term = torch.matmul(momentum.view(1, -1), fish_inverse_momentum)
    hamiltonian = - log_prob + 0.5 * pi_term + 0.5 * log_det_abs + 0.5 * quadratic_term
    if util.has_nan_or_inf(hamiltonian):
        print('Invalid hamiltonian, log_prob: {}, params: {}, momentum: {}'.format(log_prob, params, momentum))
        raise util.LogProbError()
    # if NN_flag:
    #     return hamiltonian, params
    # else:
    return hamiltonian

def hamiltonian(params, momentum, log_prob_func, jitter=0.01, normalizing_const=1., softabs_const=1e6, explicit_binding_const=100, inv_mass=None, ham_func=None, sampler=Sampler.HMC, integrator=Integrator.EXPLICIT, metric=Metric.HESSIAN):

    if sampler == Sampler.HMC:
        log_prob = log_prob_func(params)

        if util.has_nan_or_inf(log_prob):
            print('Invalid log_prob: {}, params: {}'.format(log_prob, params))
            raise util.LogProbError()

        potential = -log_prob
        if inv_mass is None:
            kinetic = 0.5 * torch.dot(momentum, momentum)
        else:
            if len(inv_mass.shape) == 2:
                kinetic = 0.5 * torch.matmul(momentum.view(1,-1),torch.matmul(inv_mass,momentum.view(-1,1))).view(-1)
            else:
                kinetic = 0.5 * torch.dot(momentum, inv_mass * momentum)
        hamiltonian = potential + kinetic
        # hamiltonian = hamiltonian
    elif sampler == Sampler.RMHMC and integrator == Integrator.IMPLICIT:
        hamiltonian = rm_hamiltonian(params, momentum, log_prob_func, jitter, normalizing_const, softabs_const=softabs_const, sampler=sampler, integrator=integrator, metric=metric)
    elif sampler == Sampler.RMHMC and integrator == Integrator.EXPLICIT:
        if type(params) is not list:
            # Therefore first instance of sampler before leapfrog_params
            hamiltonian = 2 * rm_hamiltonian(params, momentum, log_prob_func, jitter, normalizing_const, softabs_const=softabs_const, sampler=sampler, integrator=integrator, metric=metric)
        else:
            # Note that in this case hamiltonian expects a list of two tensors:
            # params = [params_orig,params_copy]; momentum = [momentum_orig,momentum_copy]
            HA = rm_hamiltonian(params[0], momentum[1], log_prob_func, jitter, normalizing_const, softabs_const=softabs_const, sampler=sampler, integrator=integrator, metric=metric)
            HB = rm_hamiltonian(params[1], momentum[0], log_prob_func, jitter, normalizing_const, softabs_const=softabs_const, sampler=sampler, integrator=integrator, metric=metric)
            HC = (0.5 * torch.sum((params[0]-params[1])**2) + 0.5 * torch.sum((momentum[0]-momentum[1])**2))
            hamiltonian = HA + HB + explicit_binding_const * HC
    elif sampler == Sampler.RMHMC and integrator == Integrator.S3: # CURRENTLY ASSUMING DIAGONAL
        log_prob = log_prob_func(params)
        ndim = params.nelement()
        pi_term = ndim * torch.log(2.*torch.tensor(pi))
        fish, abs_eigenvalues = fisher(params, log_prob_func, jitter=jitter, normalizing_const=normalizing_const, softabs_const=softabs_const, metric=metric)
        fish_inverse_momentum = cholesky_inverse(fish, momentum)
        quadratic_term = torch.matmul(momentum.view(1, -1), fish_inverse_momentum)
        # print((momentum ** 2 *  fish.diag() ** -1).sum() - quadratic_term)
        hamiltonian = - log_prob + 0.5 * quadratic_term + ham_func(params)
        # import pdb; pdb.set_trace()
        if util.has_nan_or_inf(hamiltonian):
            print('Invalid hamiltonian, log_prob: {}, params: {}, momentum: {}'.format(log_prob, params, momentum))
            raise util.LogProbError()
    else:
        raise NotImplementedError()
    # if not tup:
    return hamiltonian
    # else:
    #     model_parameters = hamiltonian[1]
    #     return hamiltonian[0], model_parameters


def sample(log_prob_func, params_init, num_samples=10, num_steps_per_sample=10, step_size=0.1, burn=0, jitter=None, inv_mass=None, normalizing_const=1., softabs_const=None, explicit_binding_const=100, fixed_point_threshold=1e-5, fixed_point_max_iterations=1000, jitter_max_tries=10, sampler=Sampler.HMC, integrator=Integrator.IMPLICIT, metric=Metric.HESSIAN, debug=False, desired_accept_rate=0.8):

    if params_init.dim() != 1:
        raise RuntimeError('params_init must be a 1d tensor.')

    if burn >= num_samples:
        raise RuntimeError('burn must be less than num_samples.')

    NUTS = False
    if sampler == Sampler.HMC_NUTS:
        if burn == 0:
            raise RuntimeError('burn must be greater than 0 for NUTS.')
        sampler = Sampler.HMC
        NUTS = True
        step_size_init = step_size
        H_t = 0.
        eps_bar = 1.

    # Invert mass matrix once (As mass is used in Gibbs resampling step)
    mass = None
    if inv_mass is not None:
        if len(inv_mass.shape) == 2:
            mass = torch.inverse(inv_mass)
        elif len(inv_mass.shape) == 1:
            mass = 1/inv_mass

    params = params_init.clone().requires_grad_()
    ret_params = [params.clone()]
    num_rejected = 0
    # if sampler == Sampler.HMC:
    util.progress_bar_init('Sampling ({}; {})'.format(sampler, integrator), num_samples, 'Samples')
    for n in range(num_samples):
        util.progress_bar_update(n)
        try:
            momentum = gibbs(params, sampler=sampler, log_prob_func=log_prob_func, jitter=jitter, normalizing_const=normalizing_const, softabs_const=softabs_const, metric=metric, mass=mass)

            ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, normalizing_const=normalizing_const, sampler=sampler, integrator=integrator, metric=metric, inv_mass=inv_mass)

            leapfrog_params, leapfrog_momenta = leapfrog(params, momentum, log_prob_func, sampler=sampler, integrator=integrator, steps=num_steps_per_sample, step_size=step_size, inv_mass=inv_mass, jitter=jitter, jitter_max_tries=jitter_max_tries, fixed_point_threshold=fixed_point_threshold, fixed_point_max_iterations=fixed_point_max_iterations, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, metric=metric, debug=debug)
            if sampler == Sampler.RMHMC and integrator == Integrator.EXPLICIT:

                # Step required to remove bias by comparing to Hamiltonian that is not augmented:
                ham = ham/2 # Original RMHMC

                params = leapfrog_params[0][-1].detach().requires_grad_()
                params_copy = leapfrog_params[-1].detach().requires_grad_()
                params_copy = params_copy.detach().requires_grad_()
                momentum = leapfrog_momenta[0][-1]
                momentum_copy = leapfrog_momenta[-1]

                leapfrog_params = leapfrog_params[0]
                leapfrog_momenta = leapfrog_momenta[0]

                # This is trying the new (unbiased) version:
                new_ham = rm_hamiltonian(params, momentum, log_prob_func, jitter, normalizing_const, softabs_const=softabs_const, sampler=sampler, integrator=integrator, metric=metric) # In rm sampler so no need for inv_mass
                # new_ham = hamiltonian([params,params_copy] , [momentum,momentum_copy], log_prob_func, jitter=jitter, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, normalizing_const=normalizing_const, sampler=sampler, integrator=integrator, metric=metric)

            else:
                params = leapfrog_params[-1].detach().requires_grad_()
                momentum = leapfrog_momenta[-1]
                new_ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, normalizing_const=normalizing_const, sampler=sampler, integrator=integrator, metric=metric, inv_mass=inv_mass)



            # new_ham = hamiltonian(params, momentum, log_prob_func, jitter=jitter, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, normalizing_const=normalizing_const, sampler=sampler, integrator=integrator, metric=metric)
            rho = min(0., acceptance(ham, new_ham))
            if debug:
                print('Current Hamiltoninian: {}, Proposed Hamiltoninian: {}'.format(ham,new_ham))

            if rho >= torch.log(torch.rand(1)):
                if debug:
                    print('Accept rho: {}'.format(rho))
                # ret_params.append(params)
                if n > burn:
                    ret_params.extend(leapfrog_params)
            else:
                num_rejected += 1
                params = ret_params[-1]
                if n > burn:
                    leapfrog_params = ret_params[-num_steps_per_sample:] ### Might want to remove grad as wastes memory
                    ret_params.extend(leapfrog_params) # append the current sample to the chain
                if debug:
                    print('REJECT')

            if NUTS and n <= burn:
                if n < burn:
                    step_size, eps_bar, H_t = adaptation(rho, n, step_size_init, H_t, eps_bar, desired_accept_rate=desired_accept_rate)
                if n  == burn:
                    step_size = eps_bar
                    print('Final Adapted Step Size: ',step_size)

        except util.LogProbError:
            num_rejected += 1
            params = ret_params[-1]
            if n > burn:
                leapfrog_params = ret_params[-num_steps_per_sample:] ### Might want to remove grad as wastes memory
                ret_params.extend(leapfrog_params)
            if debug:
                print('REJECT')
            if NUTS and n <= burn:
                # print('hi')
                rho = float('nan') # Acceptance rate = 0
                # print(rho)
                step_size, eps_bar, H_t = adaptation(rho, n, step_size_init, H_t, eps_bar, desired_accept_rate=desired_accept_rate)
            if NUTS and n  == burn:
                step_size = eps_bar
                print('Final Adapted Step Size: ',step_size)

        # gc.collect()

    util.progress_bar_end('Acceptance Rate {:.2f}'.format(1 - num_rejected/num_samples)) #need to adapt for burn
    if NUTS and debug:
        return list(map(lambda t: t.detach(), ret_params)), step_size
    else:
        return list(map(lambda t: t.detach(), ret_params))

def define_model_log_prob(model, model_loss, x, y, params_flattened_list, params_shape_list, tau_list, tau_out, predict=False):
    fmodel = util.make_functional(model)
    dist_list = []
    for tau in tau_list:
        dist_list.append(torch.distributions.Normal(torch.zeros_like(tau_list[0]), tau**-0.5))

    def log_prob_func(params):
        # model.zero_grad()
        # params is flat
        # Below we update the network weights to be params
        params_unflattened = util.unflatten(model, params)

        i_prev = 0
        l_prior = torch.zeros_like( params[0], requires_grad=True) # Set l2_reg to be on the same device as params
        for weights, index, shape, dist in zip(model.parameters(), params_flattened_list, params_shape_list, dist_list):
            # weights.data = params[i_prev:index+i_prev].reshape(shape)
            w = params[i_prev:index+i_prev]
            l_prior = dist.log_prob(w).sum() + l_prior
            i_prev += index

        # Sample prior if no data
        if x is None:
            # print('hi')
            return l_prior#/y.shape[0]


        output = fmodel(x,params=params_unflattened)

        if model_loss is 'binary_class':
            crit = nn.BCEWithLogitsLoss(reduction='sum')
            ll = - tau_out *(crit(output, y))
        elif model_loss is 'multi_class_linear_output':
    #         crit = nn.MSELoss(reduction='mean')
            crit = nn.CrossEntropyLoss(reduction='sum')
    #         crit = nn.BCEWithLogitsLoss(reduction='sum')
            ll = - tau_out *(crit(output, y.long().view(-1)))
            # ll = - tau_out *(torch.nn.functional.nll_loss(output, y.long().view(-1)))
        elif model_loss is 'multi_class_log_softmax_output':
            ll = - tau_out *(torch.nn.functional.nll_loss(output, y.long().view(-1)))

        elif model_loss is 'regression':
            # crit = nn.MSELoss(reduction='sum')
            ll = - 0.5 * tau_out * ((output - y) ** 2).sum(0)#sum(0)

        elif callable(model_loss):
            # Assume defined custom log-likelihood.
            ll = - model_loss(output, y).sum(0)
        else:
            raise NotImplementedError()
        if predict:
            return ll + l_prior, output
        else:
            return ll + l_prior

    return log_prob_func


def sample_model(model, x, y, params_init, model_loss='multi_class_linear_output' ,num_samples=10, num_steps_per_sample=10, step_size=0.1, burn=0, inv_mass=None, jitter=None, normalizing_const=1., softabs_const=None, explicit_binding_const=100, fixed_point_threshold=1e-5, fixed_point_max_iterations=1000, jitter_max_tries=10, sampler=Sampler.HMC, integrator=Integrator.IMPLICIT, metric=Metric.HESSIAN, debug=False, tau_out=1.,tau_list=None):
    params_shape_list = []
    params_flattened_list = []
    build_tau = False
    if tau_list is None:
        tau_list = []
        build_tau = True
    for weights in model.parameters():
        params_shape_list.append(weights.shape)
        params_flattened_list.append(weights.nelement())
        if build_tau:
            tau_list.append(1.)

    log_prob_func = define_model_log_prob(model, model_loss, x, y, params_flattened_list, params_shape_list, tau_list, tau_out)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return sample(log_prob_func, params_init, num_samples=num_samples, num_steps_per_sample=num_steps_per_sample, step_size=step_size, burn=burn, jitter=jitter, inv_mass=inv_mass, normalizing_const=normalizing_const, softabs_const=softabs_const, explicit_binding_const=explicit_binding_const, fixed_point_threshold=fixed_point_threshold, fixed_point_max_iterations=fixed_point_max_iterations, jitter_max_tries=jitter_max_tries, sampler=sampler, integrator=integrator, metric=metric, debug=debug, desired_accept_rate=0.8)

def predict_model(model, x, y, samples, model_loss='multi_class_linear_output', tau_out=1., tau_list=None):
    params_shape_list = []
    params_flattened_list = []
    build_tau = False
    if tau_list is None:
        tau_list = []
        build_tau = True
    for weights in model.parameters():
        params_shape_list.append(weights.shape)
        params_flattened_list.append(weights.nelement())
        if build_tau:
            tau_list.append(1.)

    log_prob_func = define_model_log_prob(model, model_loss, x, y, params_flattened_list, params_shape_list, tau_list, tau_out, predict=True)

    pred_log_prob_list = []
    pred_list = []
    for s in samples:
        lp, pred = log_prob_func(s)
        pred_log_prob_list.append(lp.detach()) # Side effect is to update weights to be s
        pred_list.append(pred.detach())

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return torch.stack(pred_list), pred_log_prob_list
