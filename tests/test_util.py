import unittest
import torch.nn as nn
from torch.nn import functional as F
import hamiltorch.util
import torch


class UtilTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_flatten_unflatten(self):
        model = nn.Linear(4, 4)
        flattened_params = hamiltorch.util.flatten(model)

        new_model = nn.Linear(4, 4)
        unflattened_params = hamiltorch.util.unflatten(new_model, flattened_params)
        hamiltorch.util.update_model_params_in_place(new_model, unflattened_params)

        new_model_flattened_params = hamiltorch.util.flatten(new_model)

        hamiltorch.util.eval_print('flattened_params', 'new_model_flattened_params')

        self.assertTrue(torch.all(torch.eq(flattened_params, new_model_flattened_params)))

    def test_model_functional(self):
        model = nn.Linear(4, 4)
        fmodel = hamiltorch.util.make_functional(model)

        x = torch.zeros(1, 4)
        ymodel = model.forward(x)
        params = list(model.parameters())
        yfunctional = fmodel(x, params=params)

        hamiltorch.util.eval_print('ymodel', 'yfunctional')

        self.assertTrue(torch.all(torch.eq(ymodel, yfunctional)))

    def test_differentiable_model_functional(self):
        model = nn.Linear(2, 2)
        params = list(model.parameters())
        params_flattened = hamiltorch.util.flatten(model)
        fmodel = hamiltorch.util.make_functional(model)

        model.zero_grad()
        xmodel = torch.randn(1, 2, requires_grad=True)
        ymodel = model.forward(xmodel).sum()
        ymodel.backward()
        xmodelgrad = xmodel.grad
        pmodelgrad = torch.cat([p.grad.flatten() for p in params])

        model.zero_grad()
        xfunctional = xmodel.detach().clone().requires_grad_()
        yfunctional = fmodel(xfunctional, params=params).sum()
        yfunctional.backward()
        xfunctionalgrad = xfunctional.grad
        pfunctionalgrad = torch.cat([p.grad.flatten() for p in params])

        hamiltorch.util.eval_print('params_flattened', 'xmodel', 'ymodel', 'xmodelgrad', 'xfunctional', 'yfunctional', 'xfunctionalgrad', 'pmodelgrad', 'pfunctionalgrad')

        self.assertTrue(torch.all(torch.eq(xmodelgrad, xfunctionalgrad)))
        self.assertTrue(torch.all(torch.eq(ymodel, yfunctional)))

    def test_conv_model_functional(self):
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.layers = nn.Sequential(
                        nn.Conv2d(1, 10, kernel_size=5),
                        nn.MaxPool2d(2),
                        nn.ReLU(),
                        nn.Conv2d(10, 20, kernel_size=5),
                        nn.MaxPool2d(2),
                        nn.ReLU())
                self.fc1 = nn.Linear(320, 50)
                self.fc2 = nn.Linear(50, 10)

            def forward(self, x):
                x = self.layers(x)
                x = x.view(-1, 320)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return F.log_softmax(x, dim=1)

        model = Net()
        model.eval()
        eval_fmodel = hamiltorch.util.make_functional(model)
        model.train()
        train_fmodel = hamiltorch.util.make_functional(model)
        # Verify correctness in eval mode
        model.eval()
        params = list(model.parameters())
        x = torch.randn(10, 1, 28, 28)

        self.assertTrue(torch.all(torch.eq(model(x).sum(), train_fmodel(x, params=params).sum())))

    def test_vanilla_HMC_reversible(self):
        def log_prob(omega):
            mean = torch.zeros(2)
            var = torch.tensor([.10,.10])
            return torch.distributions.MultivariateNormal(mean, torch.diag(var)).log_prob(omega).sum()

        params_init = torch.tensor([1.,1.])
        momentum_init = torch.tensor([1.,1.])
        inv_mass = torch.tensor([1.,1.])
        p,m = hamiltorch.samplers.leapfrog(params_init, momentum_init, log_prob, steps=100, step_size=0.1, jitter=None, normalizing_const=1., softabs_const=1e6, explicit_binding_const=100, fixed_point_threshold=1e-20, fixed_point_max_iterations=6, jitter_max_tries=10, inv_mass=inv_mass, ham_func=None, sampler=hamiltorch.Sampler.HMC, integrator=hamiltorch.Integrator.EXPLICIT, metric=hamiltorch.Metric.HESSIAN, debug=False)
        momentum_reversed = - m[-1].clone()
        p,m = hamiltorch.samplers.leapfrog(p[-1], momentum_reversed, log_prob, steps=100, step_size=0.1, jitter=None, normalizing_const=1., softabs_const=1e6, explicit_binding_const=100, fixed_point_threshold=1e-20, fixed_point_max_iterations=6, jitter_max_tries=10, inv_mass=inv_mass, ham_func=None, sampler=hamiltorch.Sampler.HMC, integrator=hamiltorch.Integrator.EXPLICIT, metric=hamiltorch.Metric.HESSIAN, debug=False)

        self.assertTrue(torch.all(torch.eq(p[-1], params_init)))

if __name__ == '__main__':
    unittest.main(verbosity=2)
